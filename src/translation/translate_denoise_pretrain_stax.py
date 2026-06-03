"""
Stage 1: Denoising Pretraining for Chinese-English Translation
=============================================================

BART-style text denoising autoencoder on monolingual Chinese data.

The Encoder receives corrupted Chinese text (masked + deleted tokens),
the Decoder reconstructs the original text. This pretrains:
  - Decoder: learns to generate fluent Chinese (fully transferred to Stage 2)
  - Encoder blocks: learn general sequence representations (transferred)
  - Encoder embedding: learns Chinese token semantics (replaced in Stage 2)

Usage:
    python src/translation/translate_denoise_pretrain_stax.py

Output:
    ckpt/translate_denoise_pretrain.pkl   — full model weights for Stage 2
    translate_denoise_pretrain_history.npz — training curves
    translate_denoise_pretrain_curve.png   — loss plot

Reference:
    Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training
    for Natural Language Generation, Translation, and Comprehension" (2019)
"""

import os, sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if '--xla_gpu_enable_command_buffer' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_command_buffer='

import time, pickle
import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import re
import jieba
from collections import Counter
import logging
import json
from torch.utils.data import Dataset, DataLoader
from functools import partial

from transformers.transformer_flash_v1 import Transformer

# ═══════════════════════════════════════════════════════════════════
# Paths & Logging
# ═══════════════════════════════════════════════════════════════════

PRETRAIN_DIR = "./translate_denoise_pretrain"
os.makedirs(PRETRAIN_DIR, exist_ok=True)
os.makedirs("ckpt", exist_ok=True)

logging.basicConfig(
    filename=os.path.join(PRETRAIN_DIR, 'pretrain.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


# ═══════════════════════════════════════════════════════════════════
# Special Tokens
# ═══════════════════════════════════════════════════════════════════

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"          # BART-style masking token


# ═══════════════════════════════════════════════════════════════════
# Vocabulary (with <mask> token)
# ═══════════════════════════════════════════════════════════════════

class Vocabulary:
    """Vocabulary with <mask> token for denoising pretraining."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]:
            if token not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[token] = index
                self.idx2word[index] = token
                self.word_count[token] = 0

    def build_from_sentences(self, sentences, min_freq=1, max_vocab_size=55000):
        self.word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                self.word_count[word.lower()] += 1
        self._add_special_tokens()
        non_special = [w for w in self.word_count
                       if w not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]]
        for word in sorted(non_special, key=lambda w: self.word_count[w], reverse=True):
            if self.word_count[word] >= min_freq and len(self.word2idx) < max_vocab_size:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        logger.info(f"词汇表大小: {len(self.word2idx)} (含 <mask>={self.word2idx[MASK_TOKEN]})")
        return self

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence, add_special_tokens=True):
        """Encode token list to numpy array of ids."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.word2idx[SOS_TOKEN])
        for word in sentence:
            w = word.lower()
            tokens.append(self.word2idx.get(w, self.word2idx[UNK_TOKEN]))
        if add_special_tokens:
            tokens.append(self.word2idx[EOS_TOKEN])
        return np.array(tokens, dtype=np.int32)

    def decode(self, index_list, remove_special=True):
        """Decode ids back to token strings."""
        tokens = [self.idx2word.get(int(idx), UNK_TOKEN) for idx in index_list]
        if remove_special:
            return [t for t in tokens if t not in
                    [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN]]
        return tokens

    def save(self, file_path):
        data = {'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'word_count': dict(self.word_count)}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_count = Counter(data['word_count'])
        return vocab


# ═══════════════════════════════════════════════════════════════════
# Chinese Text Cleaning
# ═══════════════════════════════════════════════════════════════════

def clean_chinese_text(text):
    """Clean Chinese text for tokenization.

    Keeps CJK chars, Latin letters, digits, basic punctuation.
    """
    text = re.sub(r"[^一-龥a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([一-龥])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([一-龥])', r'\1 \2', text)
    text = text.replace(' . . .', ' ...')
    text = re.sub(r' ([.,?!])', r'\1', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# BART-style Text Corruption
# ═══════════════════════════════════════════════════════════════════

def corrupt_sentence(tokens, mask_token_str, mask_ratio=0.30, delete_ratio=0.20,
                     rng=None):
    """Apply BART-style denoising corruption to a token list.

    Two corruption types (mutually exclusive per position):
      1. Token Masking: replace token with <mask> (mask_ratio of positions)
      2. Token Deletion:  remove token entirely   (delete_ratio of positions)

    The remaining (1 - mask_ratio - delete_ratio) positions are kept as-is.

    Args:
        tokens: list of str — raw tokens (no special tokens)
        mask_token_str: str — the mask token string (e.g. "<mask>")
        mask_ratio: float — fraction of tokens to mask
        delete_ratio: float — fraction of tokens to delete
        rng: np.random.RandomState or None

    Returns:
        corrupted: list of str — corrupted token sequence (may be shorter)
        original:  list of str — uncorrupted copy of input tokens
    """
    if rng is None:
        rng = np.random.RandomState()

    n = len(tokens)
    if n == 0:
        return [], []

    decisions = rng.rand(n)
    corrupted = []

    for i, token in enumerate(tokens):
        if decisions[i] < mask_ratio:
            corrupted.append(mask_token_str)         # mask
        elif decisions[i] < mask_ratio + delete_ratio:
            continue                                  # delete (skip)
        else:
            corrupted.append(token)                   # keep

    # Ensure at least one token remains (avoid empty sequence)
    if len(corrupted) == 0:
        # Keep at least one random token
        idx = rng.randint(0, n)
        corrupted = [tokens[idx]]

    return corrupted, list(tokens)


# ═══════════════════════════════════════════════════════════════════
# Monolingual Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_monolingual_chinese(file_path, max_length=64, max_sentences=None):
    """Extract Chinese sentences from AI Challenger TSV (en\\tzh format).

    Tokenises Chinese text with jieba, filters by length.

    Returns:
        sentences: list of list of str — tokenised Chinese sentences
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return []

    sentences = []
    skipped_short = 0
    skipped_long = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_sentences and len(sentences) >= max_sentences:
                break

            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            cn_text = parts[1].strip()
            if not cn_text:
                continue

            cn_clean = clean_chinese_text(cn_text)
            cn_tokens = [t for t in jieba.cut(cn_clean) if t.strip()]

            if not cn_tokens:
                continue

            # Filter by token count (excl. <sos>/<eos>)
            if len(cn_tokens) < 2:
                skipped_short += 1
                continue
            if len(cn_tokens) > max_length - 2:       # reserve room for <sos>/<eos>
                skipped_long += 1
                continue

            sentences.append(cn_tokens)

            if (i + 1) % 500000 == 0:
                logger.info(f"  已加载 {len(sentences):,} 句 (行 {i+1:,})")

    logger.info(f"加载单语中文: {len(sentences):,} 句 "
                f"(跳过: 过短{skipped_short:,}, 过长{skipped_long:,})")

    # Print sample
    if sentences:
        logger.info(f"样本: {' '.join(sentences[0])}")

    return sentences


# ═══════════════════════════════════════════════════════════════════
# Denoising Dataset
# ═══════════════════════════════════════════════════════════════════

class DenoisingDataset(Dataset):
    """Dataset that applies text corruption on-the-fly.

    Each __getitem__ call produces a differently corrupted version
    of the same original sentence (stochastic corruption).
    """

    def __init__(self, sentences, mask_token_str, mask_ratio=0.30, delete_ratio=0.20,
                 seed=42):
        self.sentences = sentences
        self.mask_token = mask_token_str
        self.mask_ratio = mask_ratio
        self.delete_ratio = delete_ratio
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        original = self.sentences[idx]
        # Use a fresh RNG per call for stochastic corruption
        rng = np.random.RandomState()
        corrupted, clean = corrupt_sentence(
            original, self.mask_token,
            mask_ratio=self.mask_ratio,
            delete_ratio=self.delete_ratio,
            rng=rng,
        )
        return corrupted, clean


def denoise_collate_fn(batch, vocab, max_length):
    """Collate (corrupted, original) pairs into padded arrays.

    Args:
        batch: list of (corrupted_tokens, original_tokens) — both list of str
        vocab: Vocabulary instance
        max_length: int

    Returns:
        src_mat: (batch, max_length) — corrupted (encoder input)
        tgt_mat: (batch, max_length) — original  (decoder target)
    """
    src_seqs, tgt_seqs = [], []
    for corrupted, original in batch:
        src_seqs.append(vocab.encode(corrupted, add_special_tokens=True))
        tgt_seqs.append(vocab.encode(original, add_special_tokens=True))

    pad_id = vocab.word2idx[PAD_TOKEN]
    src_mat = np.full((len(batch), max_length), pad_id, dtype=np.int32)
    tgt_mat = np.full((len(batch), max_length), pad_id, dtype=np.int32)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_mat[i, :min(len(s), max_length)] = s[:max_length]
        tgt_mat[i, :min(len(t), max_length)] = t[:max_length]

    return src_mat, tgt_mat


# ═══════════════════════════════════════════════════════════════════
# Training Step
# ═══════════════════════════════════════════════════════════════════

def create_padding_mask(sequences, vocab):
    """Create boolean padding mask (True = valid position)."""
    pad_id = vocab.word2idx[PAD_TOKEN]
    return sequences != pad_id


def make_pretrain_step(model_apply, optimizer, vocab):
    """Create JIT-compiled denoising pretraining step.

    Loss: cross-entropy over ALL tokens of the original sequence.
    Unlike MAE (which only computes loss on masked positions), BART
    trains the decoder to regenerate the full original text.
    """

    @jax.jit
    def train_step(params, batch, opt_state, rng):
        src_inputs, tgt_inputs = batch[0], batch[1]
        src_padding_mask = create_padding_mask(src_inputs, vocab)
        tgt_padding_mask = create_padding_mask(tgt_inputs, vocab)

        def loss_fn(params):
            logits = model_apply(params, (src_inputs, tgt_inputs),
                                 src_padding_mask=src_padding_mask,
                                 tgt_padding_mask=tgt_padding_mask,
                                 rng=rng)
            # Standard teacher-forcing: predict next token
            labels = tgt_inputs[:, 1:]          # shift right
            logits = logits[:, :-1, :]            # remove last prediction
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            # Mask out padding positions
            mask = (labels != vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
            loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)

        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, grad_norm

    return train_step


def make_pretrain_val_step(model_apply, vocab):
    """Validation step (no dropout, no grad)."""

    @jax.jit
    def val_step(params, batch):
        src_inputs, tgt_inputs = batch[0], batch[1]
        src_padding_mask = create_padding_mask(src_inputs, vocab)
        tgt_padding_mask = create_padding_mask(tgt_inputs, vocab)

        logits = model_apply(params, (src_inputs, tgt_inputs),
                             src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask,
                             is_training=False)
        labels = tgt_inputs[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask = (labels != vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    return val_step


def validate_epoch(val_step_fn, params, val_loader):
    """Run validation over the entire val set."""
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        total_loss += val_step_fn(params, batch)
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════
# Reconstruction Display
# ═══════════════════════════════════════════════════════════════════

def show_reconstructions(params, vocab, model_encode, model_decode, max_len,
                         real_sentences, num_examples=4):
    """Show corrupted → reconstructed examples during training.

    Uses real Chinese sentences from the dataset (not random word jumbles).
    """
    rng = np.random.RandomState(999)  # fixed for reproducibility
    mask_token_str = MASK_TOKEN

    # Pick real sentences from the dataset
    if len(real_sentences) == 0:
        logger.warning("  无可用的真实句子进行重建展示")
        return

    indices = rng.choice(len(real_sentences), size=min(num_examples, len(real_sentences)),
                        replace=False)

    for idx in indices:
        original = real_sentences[idx]
        corrupted, _ = corrupt_sentence(
            original, mask_token_str, mask_ratio=0.30, delete_ratio=0.20, rng=rng)

        # Encode
        src_ids = vocab.encode(corrupted, add_special_tokens=True)
        tgt_ids = vocab.encode(original, add_special_tokens=True)

        # Pad to max_len
        pad_id = vocab.word2idx[PAD_TOKEN]
        src_input = np.full((1, max_len), pad_id, dtype=np.int32)
        src_input[0, :min(len(src_ids), max_len)] = src_ids[:max_len]

        # Greedy decode
        tgt_input = np.full((1, max_len), pad_id, dtype=np.int32)
        tgt_input[0, 0] = vocab.word2idx[SOS_TOKEN]
        finished = np.zeros((1,), dtype=bool)

        src_padding_mask = create_padding_mask(src_input, vocab)
        encoder_output = model_encode(params, src_input,
                                      src_padding_mask=src_padding_mask)

        for step in range(1, max_len):
            if np.all(finished):
                break
            tgt_padding_mask = tgt_input != pad_id
            logits = model_decode(params, tgt_input, encoder_output,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_padding_mask)
            next_tokens = np.argmax(logits[0, step - 1, :], axis=-1)
            next_tokens = np.where(finished, pad_id, next_tokens)
            tgt_input[0, step] = next_tokens
            finished = finished | (next_tokens == vocab.word2idx[EOS_TOKEN])

        reconstructed = vocab.decode(tgt_input[0], remove_special=True)
        logger.info(f"  损坏: {' '.join(corrupted)}")
        logger.info(f"  原文: {' '.join(original)}")
        logger.info(f"  重建: {' '.join(reconstructed)}")
        logger.info("   ---")


# ═══════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(path, params, opt_state):
    """Save model params and optimizer state."""
    with open(path, "wb") as f:
        pickle.dump((params, opt_state), f)
    logger.info(f"检查点已保存: {path}")


def load_checkpoint(path):
    """Load checkpoint, compatible with old (params-only) and new format."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            params, opt_state = data
            logger.info(f"加载检查点 (含优化器状态): {path}")
            return params, opt_state
        else:
            logger.info(f"加载检查点 (旧格式，仅参数): {path}")
            return data, None
    except FileNotFoundError:
        logger.warning(f"检查点不存在: {path}")
        return None, None


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Hyperparameters ──────────────────────────────────────────
    n_heads = 8
    head_dim = 64
    embed_dim = n_heads * head_dim          # 512
    hidden_dim = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_length = 64
    block_size = 1024
    batch_size = 64

    # Corruption parameters
    mask_ratio = 0.30       # 30% of tokens → <mask>
    delete_ratio = 0.20     # 20% of tokens → deleted

    # Training schedule
    epochs = 200
    warmup_epochs = 10
    peak_lr = 3e-4
    weight_decay = 0.0001
    grad_clip = 1.0

    # Data
    dataset_path = "./data/ai_challenger_zh_en.tsv"
    val_ratio = 0.05          # 5% of monolingual data for validation (reconstruction)
    max_sentences = None      # None = use all sentences (up to 3M)

    # ── Seeds ────────────────────────────────────────────────────
    np.random.seed(42)
    random.seed(42)

    checkpoint_path = os.path.join(PRETRAIN_DIR, "pretrain_checkpoint.pkl")
    best_checkpoint_path = "ckpt/translate_denoise_pretrain.pkl"
    history_path = os.path.join(PRETRAIN_DIR, "pretrain_history.npz")
    curve_path = os.path.join(PRETRAIN_DIR, "pretrain_curve.png")

    # ── Load monolingual Chinese data ────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1: Denoising Pretraining (BART-style)")
    logger.info(f"Corruption: mask={mask_ratio}, delete={delete_ratio}")
    logger.info("=" * 60)

    logger.info("加载单语中文数据...")
    sentences = load_monolingual_chinese(
        dataset_path, max_length=max_length, max_sentences=max_sentences)
    if not sentences:
        logger.error("未找到中文数据，退出。")
        return

    # ── Build or load vocabulary ─────────────────────────────────
    vocab_path = os.path.join(PRETRAIN_DIR, "zh_vocab.json")
    if os.path.exists(vocab_path):
        logger.info("加载已有词汇表...")
        vocab = Vocabulary.load(vocab_path)
    else:
        logger.info("构建中文词汇表...")
        vocab = Vocabulary().build_from_sentences(
            sentences, min_freq=1, max_vocab_size=55000)
        vocab.save(vocab_path)

    vocab_size = len(vocab)
    logger.info(f"词汇表: {vocab_size} 词 (mask_id={vocab.word2idx[MASK_TOKEN]})")

    # ── Train/Val split ──────────────────────────────────────────
    n = len(sentences)
    rng_split = np.random.RandomState(42)
    indices = rng_split.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    logger.info(f"数据分割: 训练 {len(train_idx):,} + 验证 {len(val_idx):,} (总计 {n:,})")

    train_dataset = DenoisingDataset(
        [sentences[i] for i in train_idx],
        mask_token_str=MASK_TOKEN,
        mask_ratio=mask_ratio,
        delete_ratio=delete_ratio,
    )
    val_dataset = DenoisingDataset(
        [sentences[i] for i in val_idx],
        mask_token_str=MASK_TOKEN,
        mask_ratio=mask_ratio,
        delete_ratio=delete_ratio,
        seed=123,  # different seed for val
    )

    def collate_wrapper(b):
        return denoise_collate_fn(b, vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_wrapper)
    logger.info(f"DataLoader: train_batches={len(train_loader)}, "
                f"val_batches={len(val_loader)}")

    # ── Build model ──────────────────────────────────────────────
    logger.info("构建去噪预训练模型...")
    logger.info(f"  embed_dim={embed_dim}, heads={n_heads}, head_dim={head_dim}")
    logger.info(f"  enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}")
    logger.info(f"  mlp_dim={hidden_dim}, max_len={max_length}")

    model_init, model_apply, model_encode, model_decode = Transformer(
        src_vocab_size=vocab_size,        # shared vocab (Chinese)
        tgt_vocab_size=vocab_size,        # shared vocab (Chinese)
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        mlp_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_len=max_length,
        block_size=block_size,
    )

    # ── Initialize or load parameters ────────────────────────────
    loaded_params, loaded_opt_state = load_checkpoint(checkpoint_path)
    if loaded_params is not None:
        params = loaded_params
        logger.info("模型参数已从检查点恢复")
    else:
        rng = jax.random.PRNGKey(0)
        src_shape = (batch_size, max_length)
        tgt_shape = (batch_size, max_length)
        output_shape, params = model_init(rng, (src_shape, tgt_shape))
        logger.info(f"模型输出形状: {output_shape}")

    # Count parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info(f"模型参数总量: {total_params:,}")

    # ── Optimizer ────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=peak_lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    if loaded_opt_state is not None:
        opt_state = loaded_opt_state
        logger.info("优化器状态已恢复")
    else:
        opt_state = optimizer.init(params)
        logger.info("优化器状态已初始化")

    logger.info(f"LR: peak={peak_lr}, warmup={warmup_steps} steps, "
                f"cosine→{peak_lr*0.01:.1e}")
    logger.info(f"Total steps: {total_steps}, epochs: {epochs}")

    # ── Create step functions ────────────────────────────────────
    train_step_fn = make_pretrain_step(model_apply, optimizer, vocab)
    val_step_fn = make_pretrain_val_step(model_apply, vocab)

    # ── Training history ─────────────────────────────────────────
    train_loss_history, val_loss_history = [], []
    grad_norm_history = []
    val_epochs_logged = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    nan_recovery_count = 0
    start_epoch = 0

    validate_every = 5
    early_stop_patience = 5

    # Resume from history if available
    if os.path.exists(history_path):
        try:
            old = np.load(history_path)
            train_loss_history = list(old['train_loss'])
            val_loss_history = list(old['val_loss'])
            grad_norm_history = list(old['grad_norm'])
            val_epochs_logged = list(old['val_epochs'])
            start_epoch = len(train_loss_history)
            logger.info(f"加载已有训练历史，从 epoch {start_epoch + 1} 继续")
        except Exception as e:
            logger.warning(f"加载历史失败: {e}")

    logger.info(f"开始训练，共 {epochs} 周期")

    # ── Training Loop ────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0
        begin_time = time.time()
        grad_norms = []

        for batch_idx, (src_inputs, tgt_inputs) in enumerate(train_loader):
            rng = jax.random.PRNGKey(epoch * len(train_loader) + batch_idx)
            params, opt_state, loss, grad_norm = train_step_fn(
                params, (src_inputs, tgt_inputs), opt_state, rng)

            # NaN recovery
            if jnp.isnan(loss) or jnp.isinf(loss):
                logger.error(f"Epoch {epoch+1}, batch {batch_idx}: NaN/Inf loss，恢复检查点")
                nan_recovery_count += 1
                try:
                    with open(checkpoint_path, "rb") as f:
                        recovered = pickle.load(f)
                    if isinstance(recovered, tuple) and len(recovered) == 2:
                        params, opt_state = recovered
                    else:
                        params = recovered
                        opt_state = optimizer.init(params)
                except FileNotFoundError:
                    logger.error("无可用检查点，跳过此 batch")
                continue

            epoch_loss += loss
            num_batches += 1
            grad_norms.append(float(grad_norm))

            if batch_idx % 200 == 0:
                logger.info(f"  Epoch {epoch+1:4d}/{epochs} | Batch {batch_idx:4d} | "
                            f"loss={loss:.4f} | grad_norm={grad_norm:.4f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

        # ── Save checkpoints (every epoch) ───────────────────────
        # Recovery checkpoint: always save for crash-resume
        save_checkpoint(checkpoint_path, params, opt_state)
        # Latest weights: always save so Stage 2 can load even if training
        # is interrupted before any val improvement
        save_checkpoint(best_checkpoint_path, params, opt_state)

        # ── Validation (every validate_every epochs) ─────────────
        val_done = False
        if (epoch + 1) % validate_every == 0:
            val_loss = validate_epoch(val_step_fn, params, val_loader)
            val_done = True
            logger.info(f"Epoch {epoch+1:4d}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Best weights: keep a separate best-only copy
                save_checkpoint(best_checkpoint_path, params, opt_state)
                logger.info(f"  ★ 新最佳模型 (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                logger.info(f"早停: val_loss 连续 {early_stop_patience} 次未改善")
                break

        # ── Show reconstructions (every epoch) ───────────────────
        logger.info("──── 重建示例 ────")
        show_reconstructions(params, vocab, model_encode, model_decode, max_length,
                             val_dataset.sentences)

        elapsed = time.time() - begin_time
        logger.info(f"Epoch {epoch+1} 完成 | train_loss={avg_train_loss:.4f} | "
                    f"耗时={elapsed:.1f}s")
        logger.info("=" * 50)

        # ── Record history ───────────────────────────────────────
        train_loss_history.append(float(avg_train_loss))
        grad_norm_history.append(float(avg_grad_norm))
        if val_done:
            val_loss_history.append(float(val_loss))
            val_epochs_logged.append(epoch + 1)

        np.savez(history_path,
                 train_loss=np.array(train_loss_history),
                 val_loss=np.array(val_loss_history),
                 grad_norm=np.array(grad_norm_history),
                 val_epochs=np.array(val_epochs_logged))

        # ── Plot ─────────────────────────────────────────────────
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_arr = range(1, len(train_loss_history) + 1)

        axes[0].plot(epochs_arr, train_loss_history, 'b-', label='Train Loss', linewidth=0.8)
        if val_loss_history:
            axes[0].plot(val_epochs_logged, val_loss_history, 'rs-',
                         label='Val Loss', markersize=4, linewidth=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (Cross-Entropy)')
        axes[0].set_title('Denoising Pretraining — Reconstruction Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_arr, grad_norm_history, 'g-', linewidth=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norm')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(curve_path, dpi=150)
        plt.close(fig)

    # ── Done ─────────────────────────────────────────────────────
    logger.info(f"\n预训练完成。最佳 val_loss: {best_val_loss:.4f}, "
                f"NaN 恢复: {nan_recovery_count} 次")
    logger.info(f"预训练权重已保存到: {best_checkpoint_path}")
    logger.info(f"下一步: python src/translation/translate_denoise_finetune_stax.py")


if __name__ == "__main__":
    main()
