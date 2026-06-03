"""
Stage 2: Translation Fine-tuning (from Denoising Pretrained Weights)
=====================================================================

Loads Stage 1 denoising pretrained encoder + decoder weights,
replaces the encoder embedding for English input, and fine-tunes
on parallel EN→ZH data.

Weight Transfer:
    Encoder embedding  → NEW (English vocab, from scratch)
    Encoder PE         → transferred (same max_len)
    Encoder blocks     → transferred (same architecture)
    Decoder (all)      → transferred (same Chinese vocab)

Usage:
    python src/translation/translate_denoise_finetune_stax.py

Dependency:
    ckpt/translate_denoise_pretrain.pkl  — Stage 1 output

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

FINETUNE_DIR = "./translate_denoise_finetune"
os.makedirs(FINETUNE_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(FINETUNE_DIR, 'finetune.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


# ═══════════════════════════════════════════════════════════════════
# Special Tokens (same as original translate_stax_flash.py)
# ═══════════════════════════════════════════════════════════════════

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


# ═══════════════════════════════════════════════════════════════════
# Vocabulary (standard, without <mask> — mask is not used in fine-tuning)
# ═══════════════════════════════════════════════════════════════════

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            if token not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[token] = index
                self.idx2word[index] = token
                self.word_count[token] = 0

    def build_from_sentences(self, sentences, min_freq, max_vocab_size):
        self.word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                self.word_count[word.lower()] += 1
        self._add_special_tokens()
        non_special = [w for w in self.word_count
                       if w not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]]
        for word in sorted(non_special, key=lambda w: self.word_count[w], reverse=True):
            if self.word_count[word] >= min_freq and len(self.word2idx) < max_vocab_size:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        logger.info(f"词汇表: {len(self.word2idx)} 词")
        return self

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence, add_special_tokens=True):
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
        tokens = [self.idx2word.get(int(idx), UNK_TOKEN) for idx in index_list]
        if remove_special:
            return [t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
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
# Text Cleaning (same as translate_stax_flash.py)
# ═══════════════════════════════════════════════════════════════════

def clean_english_text(text):
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'([.,,!?()])', r' \1 ', text)
    text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1 '\2", text)
    text = text.replace('...', ' ... ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def clean_chinese_text(text):
    text = re.sub(r"[^一-龥a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([一-龥])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([一-龥])', r'\1 \2', text)
    text = text.replace(' . . .', ' ...')
    text = re.sub(r' ([.,?!])', r'\1', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Data Loading (reused from translate_stax_flash.py)
# ═══════════════════════════════════════════════════════════════════

def load_tatoeba_corpus(file_path, max_length):
    """Load parallel EN-ZH sentence pairs from TSV."""
    en_sentences, cn_sentences, skipped = [], [], 0
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                skipped += 1; continue
            en_text, cn_text = parts[0].strip(), parts[1].strip()
            if not en_text or not cn_text:
                skipped += 1; continue
            en_tokens = en_text.split()
            cn_tokens = [t for t in jieba.cut(cn_text) if t.strip()]
            if not en_tokens or not cn_tokens:
                skipped += 1; continue
            en_sentences.append(en_tokens)
            cn_sentences.append(cn_tokens)
    logger.info(f"加载: {len(en_sentences)} 句对, 跳过 {skipped}")
    return en_sentences, cn_sentences


def load_json_corpus(file_path):
    """Load parallel data from JSON format."""
    en_sentences, cn_sentences, skipped = [], [], 0
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for pair in data:
            if len(pair) < 2:
                skipped += 1; continue
            en_text = clean_english_text(pair[0].strip())
            cn_text = clean_chinese_text(pair[1].strip())
            if not en_text or not cn_text:
                skipped += 1; continue
            en_tokens = en_text.split()
            cn_tokens = [t for t in jieba.cut(cn_text) if t.strip()]
            if not en_tokens or not cn_tokens:
                skipped += 1; continue
            en_sentences.append(en_tokens)
            cn_sentences.append(cn_tokens)
    except Exception as e:
        logger.error(f"加载 JSON 出错: {e}")
        return [], []
    logger.info(f"加载: {len(en_sentences)} 句对, 跳过 {skipped}")
    return en_sentences, cn_sentences


class TranslationDataset:
    def __init__(self, src, tgt):
        self.source = src
        self.target = tgt

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


def collate_fn(batch, src_vocab, tgt_vocab, max_length):
    src_batch, tgt_batch = [], []
    for src_sent, tgt_sent in batch:
        src_batch.append(src_vocab.encode(src_sent))
        tgt_batch.append(tgt_vocab.encode(tgt_sent))
    src_mat = np.full((len(batch), max_length), src_vocab.word2idx[PAD_TOKEN], dtype=np.int32)
    tgt_mat = np.full((len(batch), max_length), tgt_vocab.word2idx[PAD_TOKEN], dtype=np.int32)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_mat[i, :min(len(s), max_length)] = s[:max_length]
        tgt_mat[i, :min(len(t), max_length)] = t[:max_length]
    return src_mat, tgt_mat


def load_dataset_and_vocab(dataset_path, vocab_dir, min_freq=1,
                           max_vocab_size_src=55000, max_vocab_size_tgt=55000,
                           max_length=50, batch_size=32, val_ratio=0.1):
    """Load parallel data, build/load vocabs, split train/val."""
    logger.info(f"加载数据集: {dataset_path}")
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == '.json':
        en_sents, cn_sents = load_json_corpus(dataset_path)
    elif ext in ['.txt', '.tsv']:
        en_sents, cn_sents = load_tatoeba_corpus(dataset_path, max_length)
    else:
        raise ValueError(f"不支持的类型: {ext}")

    if len(en_sents) == 0:
        return None, None, None, None

    os.makedirs(vocab_dir, exist_ok=True)
    src_vocab_path = os.path.join(vocab_dir, "src_vocab.json")
    tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.json")

    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        src_vocab = Vocabulary().build_from_sentences(en_sents, min_freq, max_vocab_size_src)
        tgt_vocab = Vocabulary().build_from_sentences(cn_sents, min_freq, max_vocab_size_tgt)
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)

    # Train/Val split
    n = len(en_sents)
    rng_split = np.random.RandomState(42)
    indices = rng_split.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    logger.info(f"数据分割: 训练 {len(train_idx)} + 验证 {len(val_idx)} (总计 {n})")

    train_dataset = TranslationDataset(
        [en_sents[i] for i in train_idx],
        [cn_sents[i] for i in train_idx],
    )
    val_dataset = TranslationDataset(
        [en_sents[i] for i in val_idx],
        [cn_sents[i] for i in val_idx],
    )

    def collate_wrapper(b):
        return collate_fn(b, src_vocab, tgt_vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_wrapper)
    logger.info(f"DataLoader: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    return src_vocab, tgt_vocab, train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════
# Pretrained Weight Transfer
# ═══════════════════════════════════════════════════════════════════

def load_pretrained_weights(pretrained_path):
    """Load Stage 1 pretrained checkpoint.

    Returns:
        pretrained_params: the 4-tuple (enc_params, enc_bn, dec_params, dec_bn)
        pretrained_zh_vocab_size: the Chinese vocab size from pretraining
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"预训练权重未找到: {pretrained_path}\n"
            f"请先运行 Stage 1: python src/translation/translate_denoise_pretrain_stax.py"
        )

    with open(pretrained_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        pretrained_params, _opt_state = data
    else:
        pretrained_params = data

    # Extract pretrained vocab size from the decoder embedding shape
    pt_enc_params, pt_enc_bn, pt_dec_params, pt_dec_bn = pretrained_params
    pt_enc_emb = pt_enc_params[0]       # (zh_vocab_size, embed_dim)
    pt_dec_emb = pt_dec_params[0]       # (zh_vocab_size, embed_dim)

    pt_zh_vocab_size = pt_dec_emb.shape[0]
    pt_embed_dim = pt_enc_emb.shape[1]

    logger.info(f"已加载预训练权重:")
    logger.info(f"  中文词表大小: {pt_zh_vocab_size}")
    logger.info(f"  嵌入维度: {pt_embed_dim}")
    logger.info(f"  来源: {pretrained_path}")

    return pretrained_params, pt_zh_vocab_size


def transfer_weights(pretrained_params, new_params, src_vocab, tgt_vocab):
    """Transfer pretrained weights into the fine-tuning model.

    Strategy:
        - Encoder embedding: KEEP new (English vocab, freshly initialized)
        - Encoder PE:         COPY from pretrained
        - Encoder blocks:     COPY from pretrained
        - Decoder (all):      COPY from pretrained (if Chinese vocab size matches)

    Args:
        pretrained_params: 4-tuple from Stage 1
        new_params: 4-tuple from fresh Transformer init (has correct shapes)
        src_vocab: English Vocabulary
        tgt_vocab: Chinese Vocabulary

    Returns:
        transferred_params: 4-tuple with mixed weights
    """
    pt_enc, pt_enc_bn, pt_dec, pt_dec_bn = pretrained_params
    new_enc, new_enc_bn, new_dec, new_dec_bn = new_params

    # ── Verify dimensions ────────────────────────────────────────
    pt_embed_dim = pt_enc[0].shape[1]       # embed_dim from pretrained
    new_embed_dim = new_enc[0].shape[1]     # embed_dim from new init

    if pt_embed_dim != new_embed_dim:
        raise ValueError(
            f"嵌入维度不匹配: pretrained={pt_embed_dim}, new={new_embed_dim}. "
            f"请确保两个阶段使用相同的 embed_dim。"
        )

    logger.info("权重转移方案:")
    logger.info(f"  Encoder embedding: NEW (英文, {len(src_vocab)} 词) "
                f"← pretrained 中文 {pt_enc[0].shape[0]} 词不可复用")
    logger.info(f"  Encoder PE:         TRANSFER ({pt_enc[1].shape})")
    logger.info(f"  Encoder blocks:     TRANSFER ({len(pt_enc[2])} 层)")
    logger.info(f"  Decoder (all):      TRANSFER "
                f"(中文 {pt_dec[0].shape[0]} 词)")

    # ── Verify decoder vocab match ───────────────────────────────
    pt_dec_vocab_size = pt_dec[0].shape[0]
    new_dec_vocab_size = new_dec[0].shape[0]

    if pt_dec_vocab_size != new_dec_vocab_size:
        logger.warning(
            f"⚠ 解码器词表大小不匹配: pretrained={pt_dec_vocab_size}, "
            f"new={new_dec_vocab_size}。"
        )
        logger.warning(
            f"  预训练使用中文词表 {pt_dec_vocab_size} 词，"
            f"微调使用 {new_dec_vocab_size} 词。"
        )
        logger.warning(
            f"  将只转移 Encoder PE + blocks，Decoder 从头训练。"
        )
        # Partial transfer: only encoder PE + blocks
        transferred_enc = [new_enc[0], pt_enc[1], pt_enc[2]]
        transferred_dec = new_dec  # Keep fresh decoder
    else:
        # Full transfer (except encoder embedding)
        transferred_enc = [new_enc[0], pt_enc[1], pt_enc[2]]
        transferred_dec = pt_dec   # Use pretrained decoder

    # bn_states are all None for this architecture; keep new ones
    transferred_params = (
        tuple(transferred_enc), new_enc_bn,
        transferred_dec, new_dec_bn,
    )

    # ── Count transferred vs fresh params ────────────────────────
    transferred_count = sum(
        p.size for p in jax.tree_util.tree_leaves(
            (transferred_enc[1], transferred_enc[2], transferred_dec)
        )
    )
    fresh_count = sum(
        p.size for p in jax.tree_util.tree_leaves(transferred_enc[0])
    )
    logger.info(f"参数统计: 迁移 {transferred_count:,} + 新初始化 {fresh_count:,} "
                f"= {transferred_count + fresh_count:,}")

    return transferred_params


# ═══════════════════════════════════════════════════════════════════
# Training & Validation Steps
# ═══════════════════════════════════════════════════════════════════

def create_padding_mask(sequences, vocab):
    pad_id = vocab.word2idx[PAD_TOKEN]
    return sequences != pad_id


def make_train_step(model_apply, optimizer, src_vocab, tgt_vocab):

    @jax.jit
    def train_step(params, batch, opt_state, rng):
        src_inputs, tgt_inputs = batch[0], batch[1]
        src_padding_mask = create_padding_mask(src_inputs, src_vocab)
        tgt_padding_mask = create_padding_mask(tgt_inputs, tgt_vocab)

        def loss_fn(params):
            logits = model_apply(params, (src_inputs, tgt_inputs),
                                 src_padding_mask=src_padding_mask,
                                 tgt_padding_mask=tgt_padding_mask,
                                 rng=rng)
            labels = tgt_inputs[:, 1:]
            logits = logits[:, :-1, :]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            mask = (labels != tgt_vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
            loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)

        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, grad_norm

    return train_step


def make_val_step(model_apply, src_vocab, tgt_vocab):

    @jax.jit
    def val_step(params, batch):
        src_inputs, tgt_inputs = batch[0], batch[1]
        src_padding_mask = create_padding_mask(src_inputs, src_vocab)
        tgt_padding_mask = create_padding_mask(tgt_inputs, tgt_vocab)

        logits = model_apply(params, (src_inputs, tgt_inputs),
                             src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask,
                             is_training=False)
        labels = tgt_inputs[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask = (labels != tgt_vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    return val_step


def validate_epoch(val_step_fn, params, val_loader):
    total_loss, num_batches = 0.0, 0
    for batch in val_loader:
        total_loss += val_step_fn(params, batch)
        num_batches += 1
    return total_loss / max(num_batches, 1)


# ═══════════════════════════════════════════════════════════════════
# Translation Inference
# ═══════════════════════════════════════════════════════════════════

def translate_batch(params, src_inputs, src_vocab, tgt_vocab,
                    model_encode, model_decode, max_len):
    batch_size = src_inputs.shape[0]
    pad_id = tgt_vocab.word2idx[PAD_TOKEN]
    eos_id = tgt_vocab.word2idx[EOS_TOKEN]
    tgt_inputs = jnp.full((batch_size, max_len), pad_id, dtype=jnp.int32)
    tgt_inputs = tgt_inputs.at[:, 0].set(tgt_vocab.word2idx[SOS_TOKEN])
    finished = jnp.zeros((batch_size,), dtype=bool)

    src_padding_mask = create_padding_mask(src_inputs, src_vocab)
    encoder_output = model_encode(params, src_inputs, src_padding_mask=src_padding_mask)

    for step in range(1, max_len):
        if jnp.all(finished):
            break
        tgt_padding_mask = tgt_inputs != pad_id
        logits = model_decode(params, tgt_inputs, encoder_output,
                              src_padding_mask=src_padding_mask,
                              tgt_padding_mask=tgt_padding_mask)
        next_tokens = jnp.argmax(logits[:, step - 1, :], axis=-1)
        next_tokens = jnp.where(finished, pad_id, next_tokens)
        tgt_inputs = tgt_inputs.at[:, step].set(next_tokens)
        finished = finished | (next_tokens == eos_id)

    return tgt_inputs


def translate_examples(params, src_vocab, tgt_vocab,
                       model_encode, model_decode, max_length):
    batch = [
        ("they planted roses along the fence every spring morning", "你好吗"),
        ("he missed the train because his alarm never rang", "你好吗"),
        ("he is also very famous in japan", "你好世界"),
        ("i don't expect anything from you", "早上好"),
        ("i found it easy to speak english", "你好吗"),
        ("she opened the old wooden box and found a letter inside", "你好吗"),
        ("time passes quickly when we're doing something we like .", "你好吗"),
        ("tom needs to study more if he hopes to pass this class .", "你好吗"),
    ]
    # Clean and tokenize
    cleaned = []
    for en_text, cn_text in batch:
        en_tokens = [t for t in clean_english_text(en_text).split() if t.strip()]
        cn_tokens = [t for t in jieba.cut(clean_chinese_text(cn_text)) if t.strip()]
        cleaned.append((en_tokens, cn_tokens))

    src_batch = []
    for en_tokens, _ in cleaned:
        src_batch.append(src_vocab.encode(en_tokens))
    src_mat = np.full((len(src_batch), max_length),
                      src_vocab.word2idx[PAD_TOKEN], dtype=np.int32)
    for i, s in enumerate(src_batch):
        src_mat[i, :min(len(s), max_length)] = s[:max_length]

    translations = translate_batch(params, src_mat, src_vocab, tgt_vocab,
                                   model_encode, model_decode, max_len=max_length)
    logger.info("──── 翻译示例 ────")
    for i in range(translations.shape[0]):
        tokens = tgt_vocab.decode(translations[i], remove_special=True)
        logger.info(f"  EN: {' '.join(cleaned[i][0])}")
        logger.info(f"  ZH: {''.join(tokens)}")


# ═══════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ═══════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            params, opt_state = data
            logger.info(f"加载检查点 (含优化器状态): {checkpoint_path}")
            return params, opt_state
        else:
            params = data
            logger.info(f"加载检查点 (旧格式): {checkpoint_path}")
            return params, None
    except FileNotFoundError:
        logger.warning(f"检查点不存在: {checkpoint_path}")
        return None, None


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Hyperparameters (matching Stage 1 architecture) ──────────
    n_heads = 8
    head_dim = 64
    embed_dim = n_heads * head_dim          # 512
    hidden_dim = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_length = 64
    block_size = 1024
    batch_size = 64

    # Training schedule (lower LR for fine-tuning)
    epochs = 200
    warmup_epochs = 5
    peak_lr = 1e-4               # Lower than pretraining (3e-4)
    weight_decay = 0.0001
    grad_clip = 1.0

    # ── Paths ────────────────────────────────────────────────────
    dataset_path = "./data/ai_challenger_zh_en.tsv"
    pretrained_path = "ckpt/translate_denoise_pretrain.pkl"

    checkpoint_path = os.path.join(FINETUNE_DIR, f"model_{_Path(dataset_path).stem}.pkl")
    best_checkpoint_path = os.path.join(FINETUNE_DIR,
                                        f"model_{_Path(dataset_path).stem}_best.pkl")
    curve_path = os.path.join(FINETUNE_DIR,
                              f"model_{_Path(dataset_path).stem}_curve.png")
    history_path = os.path.join(FINETUNE_DIR,
                                f"model_{_Path(dataset_path).stem}_history.npz")

    # ── Seeds ────────────────────────────────────────────────────
    np.random.seed(42)
    random.seed(42)

    # ── Load parallel data + build vocabs ────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2: Translation Fine-tuning (from Pretrained)")
    logger.info("=" * 60)

    src_vocab, tgt_vocab, train_loader, val_loader = load_dataset_and_vocab(
        dataset_path, FINETUNE_DIR, min_freq=1,
        max_vocab_size_src=55000, max_vocab_size_tgt=55000,
        max_length=max_length, batch_size=batch_size, val_ratio=0.1,
    )

    if src_vocab is None:
        logger.error("数据加载失败，退出。")
        return

    # ── Load pretrained weights ──────────────────────────────────
    logger.info("\n加载预训练权重...")
    pretrained_params, pt_zh_vocab_size = load_pretrained_weights(pretrained_path)

    # ── Initialize fresh translation model ───────────────────────
    logger.info("初始化翻译模型...")
    model_init, model_apply, model_encode, model_decode = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        mlp_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_len=max_length,
        block_size=block_size,
    )

    rng = jax.random.PRNGKey(0)
    src_shape = (batch_size, max_length)
    tgt_shape = (batch_size, max_length)
    output_shape, new_params = model_init(rng, (src_shape, tgt_shape))
    logger.info(f"模型输出形状: {output_shape}")

    # ── Transfer weights ─────────────────────────────────────────
    params = transfer_weights(pretrained_params, new_params, src_vocab, tgt_vocab)

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

    # ── Resume if checkpoint exists ──────────────────────────────
    loaded_params, loaded_opt_state = load_checkpoint(checkpoint_path)
    if loaded_params is not None:
        params = loaded_params
        logger.info("从微调检查点恢复")
        opt_state = loaded_opt_state if loaded_opt_state is not None else optimizer.init(params)
    else:
        opt_state = optimizer.init(params)
        logger.info("优化器状态已初始化（微调）")

    train_step_fn = make_train_step(model_apply, optimizer, src_vocab, tgt_vocab)
    val_step_fn = make_val_step(model_apply, src_vocab, tgt_vocab)

    # ── Training history ─────────────────────────────────────────
    validate_every = 10
    early_stop_patience = 5

    best_val_loss = float('inf')
    nan_recovery_count = 0
    epochs_no_improve = 0
    start_epoch = 0

    train_loss_history, val_loss_history, grad_norm_history = [], [], []
    val_epochs_logged = []

    if os.path.exists(history_path):
        try:
            old = np.load(history_path)
            train_loss_history = list(old['train_loss'])
            val_loss_history = list(old['val_loss'])
            grad_norm_history = list(old['grad_norm'])
            val_epochs_logged = list(old['val_epochs'])
            start_epoch = len(train_loss_history)
            logger.info(f"加载历史，从 epoch {start_epoch + 1} 继续")
        except Exception as e:
            logger.warning(f"加载历史失败: {e}")

    # Evaluate best checkpoint if available (for early stop tracking)
    if os.path.exists(best_checkpoint_path):
        best_params, _ = load_checkpoint(best_checkpoint_path)
        if best_params is not None:
            best_val_loss = validate_epoch(val_step_fn, best_params, val_loader)
            logger.info(f"评估最佳检查点 val_loss={best_val_loss:.4f}")

    logger.info(f"\n开始微调训练: epochs={epochs}, peak_lr={peak_lr}")
    logger.info(f"warmup={warmup_steps} steps, early_stop={early_stop_patience}")
    logger.info("=" * 50)

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
                logger.error(f"Epoch {epoch+1}, batch {batch_idx}: NaN/Inf，恢复检查点")
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
                    logger.error("无检查点，跳过")
                continue

            epoch_loss += loss
            num_batches += 1
            grad_norms.append(float(grad_norm))

            if batch_idx % 500 == 0:
                logger.info(f"  Epoch {epoch+1:4d}/{epochs} | Batch {batch_idx:4d} | "
                            f"loss={loss:.4f} | grad_norm={grad_norm:.4f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

        # ── Save checkpoints (every epoch) ────────────────────────
        with open(checkpoint_path, "wb") as f:
            pickle.dump((params, opt_state), f)
        # Always save latest weights to best path too, so inference
        # can use the most recent model without waiting for val improvement
        with open(best_checkpoint_path, "wb") as f:
            pickle.dump((params, opt_state), f)

        # ── Validation ───────────────────────────────────────────
        val_done = False
        if (epoch + 1) % validate_every == 0:
            val_loss = validate_epoch(val_step_fn, params, val_loader)
            val_done = True
            logger.info(f"Epoch {epoch+1:4d}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Already saved above; just log the milestone
                logger.info(f"  ★ 新最佳模型 (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                logger.info(f"早停: val_loss 连续 {early_stop_patience} 次未改善")
                break

        # ── Translation examples ─────────────────────────────────
        translate_examples(params, src_vocab, tgt_vocab,
                           model_encode, model_decode, max_length)

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
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Fine-tuning — Translation Loss')
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

    # ── Final ────────────────────────────────────────────────────
    logger.info(f"\n微调完成。最佳 val_loss: {best_val_loss:.4f}, "
                f"NaN 恢复: {nan_recovery_count} 次")
    logger.info(f"最佳模型已保存到: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
