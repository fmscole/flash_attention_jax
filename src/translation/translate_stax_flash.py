# 1. 导入必要的库
import os,sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# 必须在 import jax 之前设置：禁用 CUDA 命令缓冲区
if '--xla_gpu_enable_command_buffer' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_command_buffer='

import time
import jax
import jax.numpy as jnp
import optax
from lib import stax_plus as stax
from transformers.transformer_flash_v1 import Transformer
import numpy as np
import random
import re
import jieba
from collections import Counter
import logging
import json
import pickle
from torch.utils.data import Dataset, DataLoader
import opencc
from functools import partial

# opencc 繁体→简体转换器（模块级单例）
_OPENCC_T2S = opencc.OpenCC('t2s')

# 创建保存目录
TRANSLATE_DIR = "./translate_jax"
os.makedirs(TRANSLATE_DIR, exist_ok=True)

# 日志
logging.basicConfig(
    filename=os.path.join(TRANSLATE_DIR, 'translation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# 特殊标记
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


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


def clean_english_text(text):
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'([.,,!?()])', r' \1 ', text)
    text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1 '\2", text)
    text = text.replace('...', ' ... ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def clean_chinese_text(text):
    text = _OPENCC_T2S.convert(text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)
    text = text.replace(' . . .', ' ...')
    text = re.sub(r' ([.,?!])', r'\1', text)
    return text.strip()


def load_tatoeba_corpus(file_path, max_length):
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
            if i == 0:
                logger.info(f"EN: {' '.join(en_tokens)}")
                logger.info(f"CN: {' '.join(cn_tokens)}")
    logger.info(f"加载: {len(en_sentences)} 句对, 跳过 {skipped}")
    return en_sentences, cn_sentences


def load_json_corpus(file_path):
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


# ═══════════════════════════════════════════════════════════════════
# 所有 Transformer 组件由 transformers.transformer_flash.Transformer 提供
# ═══════════════════════════════════════════════════════════════════


# 5. 训练步（含梯度 + dropout）
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
            pred = jnp.argmax(logits, axis=-1)
            labels = tgt_inputs[:, 1:]
            logits = logits[:, :-1, :]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            mask = (labels != tgt_vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
            loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(params)
        
        # 梯度全局 L2 范数（监控用）
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred, grad_norm

    return train_step


# 6. 验证步（无 dropout、无梯度）
def make_val_step(model_apply, src_vocab, tgt_vocab):
    """构建纯前向验证步。

    关键正确性保证：
    - is_training=False → Dropout 层直接透传，输出确定性。
    - 与训练时使用相同的 loss 公式 (softmax_cross_entropy + pad mask)，
      确保 train/val loss 可比。
    - 无 grad_fn → 不构建反向图，内存开销小，速度比训练步快约 2×。
    """
    @jax.jit
    def val_step(params, batch):
        src_inputs, tgt_inputs = batch[0], batch[1]
        src_padding_mask = create_padding_mask(src_inputs, src_vocab)
        tgt_padding_mask = create_padding_mask(tgt_inputs, tgt_vocab)

        logits = model_apply(params, (src_inputs, tgt_inputs),
                             src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask,
                             is_training=False)  # 关闭 dropout，确定性推理
        labels = tgt_inputs[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask = (labels != tgt_vocab.word2idx[PAD_TOKEN]).astype(jnp.float32)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    return val_step


def validate_epoch(val_step_fn, params, val_loader):
    """在整个验证集上跑一遍，返回平均 loss。"""
    total_loss, num_batches = 0.0, 0
    for batch in val_loader:
        total_loss += val_step_fn(params, batch)
        num_batches += 1
    return total_loss / max(num_batches, 1)


def create_padding_mask(sequences, vocab, max_len=None):
    pad_id = vocab.word2idx[PAD_TOKEN]
    mask = sequences != pad_id
    if max_len is not None:
        mask = mask & (jnp.arange(sequences.shape[1]) < max_len)
    return mask


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
        # 只把已有实际 token 的位置标记为有效（PAD 位置不可见）
        tgt_padding_mask = tgt_inputs != pad_id

        logits = model_decode(params, tgt_inputs, encoder_output,
                              src_padding_mask=src_padding_mask,
                              tgt_padding_mask=tgt_padding_mask)
        next_tokens = jnp.argmax(logits[:, step - 1, :], axis=-1)
        # 已完成序列写入 PAD，避免 EOS 后继续生成垃圾 token
        next_tokens = jnp.where(finished, pad_id, next_tokens)
        tgt_inputs = tgt_inputs.at[:, step].set(next_tokens)
        finished = finished | (next_tokens == eos_id)

    return tgt_inputs


def clean_and_tokenize_batch(batch):
    cleaned = []
    for en_text, cn_text in batch:
        en_tokens = [t for t in clean_english_text(en_text).split() if t.strip()]
        cn_tokens = [t for t in jieba.cut(clean_chinese_text(cn_text)) if t.strip()]
        cleaned.append((en_tokens, cn_tokens))
    return cleaned


def translate_examples(params, src_vocab, tgt_vocab,
                       model_encode, model_decode, max_length):
    batch = [
        ("they planted roses along the fence every spring morning", "你好吗"),
        ("he missed the train because his alarm never rang", "你好吗"),
        ("he is also very famous in japan", "你好世界"),
        ("i don 't expect anything from you", "早上好"),
        ("i found it easy to speak english", "你好吗"),
        ("she opened the old wooden box and found a letter inside", "你好吗"),
        ("time passes quickly when we 're doing something we like .", "你好吗"),
        ("tom needs to study more if he hopes to pass this class .", "你好吗"),
    ]
    batch_cleared = clean_and_tokenize_batch(batch)
    src_batch, _ = collate_fn(batch_cleared, src_vocab, tgt_vocab, max_length)
    translations = translate_batch(params, src_batch, src_vocab, tgt_vocab,
                                   model_encode, model_decode, max_len=max_length)
    logger.info("──── 翻译示例 ────")
    for i in range(translations.shape[0]):
        tokens = tgt_vocab.decode(translations[i], remove_special=True)
        logger.info(f"  EN: {' '.join(batch_cleared[i][0])}")
        logger.info(f"  ZH: {''.join(tokens)}")


def load_checkpoint(checkpoint_path):
    """加载检查点，兼容新旧格式。

    Returns:
        params, opt_state — 新格式 (params, opt_state) 元组
                            旧格式仅 params，opt_state 为 None

    加载后将 numpy 数组转回 JAX DeviceArray，避免 pickle 反序列化
    产生的隐式类型转换导致 JIT re-trace 或数值问题。
    """
    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            params, opt_state = data
        else:
            params, opt_state = data, None

        # pickle 将 DeviceArray 转为 numpy，这里显式挂回 GPU
        params = jax.device_put(params)
        if opt_state is not None:
            opt_state = jax.device_put(opt_state)

        fmt = "含优化器状态" if opt_state is not None else "旧格式，仅参数"
        logger.info(f"加载检查点（{fmt}）: {checkpoint_path}")
        return params, opt_state
    except FileNotFoundError:
        logger.warning(f"检查点不存在: {checkpoint_path}")
        return None, None


def _params_valid(params):
    """检查 pytree 中是否有 NaN/Inf。"""
    for leaf in jax.tree_util.tree_leaves(params):
        if jnp.isnan(leaf).any() or jnp.isinf(leaf).any():
            return False
    return True


def init_model_params(model_init, input_shapes):
    """初始化新模型参数。"""
    rng = jax.random.PRNGKey(0)
    _, params = model_init(rng, input_shapes)
    return params


def load_dataset_and_vocab(dataset_path, vocab_dir, min_freq=1,
                           max_vocab_size_src=55000, max_vocab_size_tgt=55000,
                           max_length=50, batch_size=32, val_ratio=0.1,
                           val_dataset_path=None):
    """加载数据、构建/加载词汇表、分割训练/验证集。

    返回 (src_vocab, tgt_vocab, train_loader, val_loader)。

    如果 val_dataset_path 指定了外部验证集，则用外部验证集替代随机切分。
    词汇表从训练集构建（不暴露验证集词汇）。
    """
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

    if val_dataset_path and os.path.exists(val_dataset_path):
        # ── 使用外部验证集（不参与训练、不从训练集切分）──
        val_en_sents, val_cn_sents = load_tatoeba_corpus(val_dataset_path, max_length)
        logger.info(f"外部验证集: {len(val_en_sents)} 句对 (来自 {val_dataset_path})")

        en_sents, cn_sents = np.array(en_sents, dtype=object), np.array(cn_sents, dtype=object)
        val_en_sents, val_cn_sents = np.array(val_en_sents, dtype=object), np.array(val_cn_sents, dtype=object)
    else:
        # ── 从训练集随机切分 10% 做验证（默认方式）──
        n = len(en_sents)
        rng_split = np.random.RandomState(42)
        indices = rng_split.permutation(n)
        split = int(n * (1 - val_ratio))
        train_idx, val_idx = indices[:split], indices[split:]
        logger.info(f"数据集分割: 训练 {len(train_idx)} + 验证 {len(val_idx)} (总计 {n})")

        en_sents, cn_sents = np.array(en_sents, dtype=object), np.array(cn_sents, dtype=object)
        val_en_sents, val_cn_sents = en_sents[val_idx], cn_sents[val_idx]
        en_sents, cn_sents = en_sents[train_idx], cn_sents[train_idx]

    train_dataset = TranslationDataset(
        [en_sents[i] for i in range(len(en_sents))],
        [cn_sents[i] for i in range(len(cn_sents))],
    )
    val_dataset = TranslationDataset(
        [val_en_sents[i] for i in range(len(val_en_sents))],
        [val_cn_sents[i] for i in range(len(val_cn_sents))],
    )

    def collate_wrapper(b):
        return collate_fn(b, src_vocab, tgt_vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_wrapper)
    logger.info(f"DataLoader: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    return src_vocab, tgt_vocab, train_loader, val_loader


def main():
    n_heads = 8
    head_dim = 64
    embed_dim = n_heads * head_dim
    hidden_dim = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_length = 64
    block_size = 1024  # FlashAttention 分块大小（auto-clamp 到可整除值）
    batch_size = 64

    np.random.seed(42)
    random.seed(42)

    dataset_path = "./data/ai_challenger_zh_en.tsv"
    val_dataset_path = "./data/ai_challenger_valid_zh_en.tsv"
    src_vocab, tgt_vocab, train_loader, val_loader = load_dataset_and_vocab(
        dataset_path, TRANSLATE_DIR, min_freq=1,
        max_vocab_size_src=55000, max_vocab_size_tgt=55000,
        max_length=max_length, batch_size=batch_size, val_ratio=0.1,
        val_dataset_path=val_dataset_path,
    )

    # 构建模型 — 使用 transformer_flash.Transformer
    logger.info("构建翻译模型...")
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
    output_shape, params = model_init(rng, (src_shape, tgt_shape))
    print(f"模型输出形状: {output_shape}")

    checkpoint_path = os.path.join(TRANSLATE_DIR, f"model_{_Path(dataset_path).stem}.pkl")
    best_checkpoint_path = os.path.join(TRANSLATE_DIR, f"model_{_Path(dataset_path).stem}_best.pkl")
    safe_checkpoint_path = os.path.join(TRANSLATE_DIR, f"model_{_Path(dataset_path).stem}_safe.pkl")
    curve_path = os.path.join(TRANSLATE_DIR, f"model_{_Path(dataset_path).stem}_curve.png")
    history_path = os.path.join(TRANSLATE_DIR, f"model_{_Path(dataset_path).stem}_history.npz")

    # ── 加载已有检查点（支持续训）──
    loaded_params, loaded_opt_state = load_checkpoint(checkpoint_path)
    if loaded_params is not None and _params_valid(loaded_params):
        params = loaded_params
        logger.info("模型参数已从检查点恢复")
    elif os.path.exists(safe_checkpoint_path):
        logger.warning("常规检查点无效或缺失，回退到安全备份")
        loaded_params, loaded_opt_state = load_checkpoint(safe_checkpoint_path)
        params = loaded_params if loaded_params is not None else params
    else:
        logger.info("初始化新模型参数")

    # 学习率调度: 短 warmup → 峰值后余弦衰减
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=3e-5, warmup_steps=4000,
        decay_steps=400000, end_value=1e-5,
    )
    # 优化器: 梯度裁剪 + AdamW
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.0001),
    )
    if loaded_opt_state is not None:
        opt_state = loaded_opt_state
        logger.info("优化器状态已恢复")
    else:
        opt_state = optimizer.init(params)
        logger.info("优化器状态重新初始化")

    train_step_fn = make_train_step(model_apply, optimizer, src_vocab, tgt_vocab)
    val_step_fn = make_val_step(model_apply, src_vocab, tgt_vocab)

    num_epochs = 10000
    validate_every = 10          # 每 N 个 epoch 跑一次验证
    early_stop_patience = 3      # 验证连续不降即停（×validate_every = 30 epoch）

    best_val_loss = float('inf')
    nan_recovery_count = 0
    epochs_no_improve = 0
    start_epoch = 0

    # ── 训练历史（续训时追加）──
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
            logger.info(f"加载已有训练历史，从 epoch {start_epoch + 1} 继续")
        except Exception as e:
            logger.warning(f"加载训练历史失败，从头开始: {e}")

    # 评估最佳 val_loss（从 best checkpoint 反推，无效时回退到 safe）
    best_val_loss = float('inf')
    ckpt_to_eval = None
    if os.path.exists(best_checkpoint_path):
        ckpt_to_eval = best_checkpoint_path
    elif os.path.exists(safe_checkpoint_path):
        ckpt_to_eval = safe_checkpoint_path

    if ckpt_to_eval:
        best_params, _ = load_checkpoint(ckpt_to_eval)
        if best_params is not None and _params_valid(best_params):
            best_val_loss = validate_epoch(val_step_fn, best_params, val_loader)
            logger.info(f"评估最佳检查点 ({ckpt_to_eval}) val_loss={best_val_loss:.4f}")

    logger.info(f"开始训练，共 {num_epochs} 周期 (早停 patience={early_stop_patience})")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss, num_batches = 0.0, 0
        begin_time = time.time()
        pred = None
        grad_norms = []

        # ---- 训练 ----
        for batch_idx, (src_inputs, tgt_inputs) in enumerate(train_loader):
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss, pred, grad_norm = train_step_fn(
                params, (src_inputs, tgt_inputs), opt_state, step_rng)
            
            # NaN/Inf 检测：loss 异常时从安全检查点恢复
            if jnp.isnan(loss) or jnp.isinf(loss):
                logger.error(f"epoch {epoch+1}, batch {batch_idx}: loss 为 NaN/Inf，恢复检查点")
                nan_recovery_count += 1
                recovered = False
                for ckpt in [safe_checkpoint_path, checkpoint_path]:
                    if not os.path.exists(ckpt):
                        continue
                    try:
                        with open(ckpt, "rb") as f:
                            data = pickle.load(f)
                        if isinstance(data, tuple) and len(data) == 2:
                            params, opt_state = data
                        else:
                            params = data
                            opt_state = optimizer.init(params)
                        params = jax.device_put(params)
                        opt_state = jax.device_put(opt_state) if opt_state is not None else None
                        recovered = True
                        logger.warning(f"已从 {ckpt} 恢复参数和优化器状态")
                        break
                    except Exception:
                        continue
                if not recovered:
                    logger.error("无可用检查点，跳过此 batch")
                continue
            
            epoch_loss += loss
            num_batches += 1
            grad_norms.append(float(grad_norm))
            
            if batch_idx % 500 == 0:
                logger.info(f"epoch {epoch+1}/{num_epochs}, batch {batch_idx}, "
                           f"loss: {loss:.4f}, grad_norm: {grad_norm:.4f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

        # ---- 检查点保存（仅参数有效时保存）----
        if _params_valid(params):
            with open(checkpoint_path, "wb") as f:
                pickle.dump((params, opt_state), f)
        else:
            logger.error(f"epoch {epoch+1}: 参数含 NaN/Inf，跳过保存")

        # ---- 验证 + 最佳/安全模型保存（每 validate_every epoch）----
        if (epoch + 1) % validate_every == 0:
            val_loss = validate_epoch(val_step_fn, params, val_loader)
            logger.info(f"epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                with open(best_checkpoint_path, "wb") as f:
                    pickle.dump((params, opt_state), f)
                with open(safe_checkpoint_path, "wb") as f:
                    pickle.dump((params, opt_state), f)
                logger.info(f"epoch {epoch+1}: ★ 新最佳模型 (val_loss={best_val_loss:.4f})，已保存 + 安全备份")
            else:
                epochs_no_improve += 1

            # 早停
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"早停触发：val_loss 连续 {early_stop_patience} 次未改善，"
                            f"最佳 val_loss={best_val_loss:.4f}")
                break

        # ---- 翻译示例（每个 epoch）----
        translate_examples(params, src_vocab, tgt_vocab,
                           model_encode, model_decode, max_length)
        print("-------------------------------------------------")
        if pred is not None:
            print("pred    ", ''.join(tgt_vocab.decode(pred[0], remove_special=True)))
        print("-------------------------------------------------")

        print(f"Epoch {epoch+1} 完成，耗时 {time.time() - begin_time:.2f}s")
        print("=" * 50)

        # ── 记录训练历史 ──
        train_loss_history.append(float(avg_train_loss))
        grad_norm_history.append(float(avg_grad_norm))
        # val_loss: 验证未执行的 epoch 记 NaN（图上不连线）
        if (epoch + 1) % validate_every == 0:
            val_loss_history.append(float(val_loss))
            val_epochs_logged.append(epoch + 1)

        np.savez(history_path,
                 train_loss=np.array(train_loss_history),
                 val_loss=np.array(val_loss_history),
                 grad_norm=np.array(grad_norm_history),
                 val_epochs=np.array(val_epochs_logged))

        # ── 绘图 ──
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
        axes[0].set_title('Training & Validation Loss')
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

    print(f"\n训练结束。最佳 val_loss: {best_val_loss:.4f}，NaN 恢复次数: {nan_recovery_count}")


if __name__ == "__main__":
    main()
