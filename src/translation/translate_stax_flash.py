# 1. 导入必要的库
import os, sys
from pathlib import Path
# 自动将 src 目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 必须在 import jax 之前设置：禁用 CUDA 命令缓冲区
if '--xla_gpu_enable_command_buffer' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_command_buffer='

import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def load_un_corpus(en_path, zh_path, max_length, max_lines=None):
    """加载 UN 平行语料库（en-zh 独立文件格式）。

    每行一句，en 与 zh 文件行号对应。
    max_lines 限制读取行数（None = 全部），用于词表采样。
    """
    en_sentences, cn_sentences, skipped = [], [], 0

    if not os.path.exists(en_path):
        logger.error(f"EN 文件不存在: {en_path}"); return [], []
    if not os.path.exists(zh_path):
        logger.error(f"ZH 文件不存在: {zh_path}"); return [], []

    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(zh_path, 'r', encoding='utf-8') as f_zh:
        for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
            if max_lines is not None and i >= max_lines:
                break
            en_text = en_line.strip()
            cn_text = zh_line.strip()
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

    logger.info(f"加载 UN 语料: {len(en_sentences)} 句对, 跳过 {skipped}")
    return en_sentences, cn_sentences


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


class UNIterableDataset:
    """分片流式读取 UN 平行语料库。

    全量语料划分为 N 个 shard（默认每片 200K 行），
    ``get_shard(i)`` 读取第 i 片、shuffle 后返回列表。
    一个 epoch = 所有 shard 依次训练完毕。
    """

    def __init__(self, en_path, zh_path, val_offset=0, shard_size=200_000):
        self.en_path = en_path
        self.zh_path = zh_path
        self.val_offset = val_offset
        self.shard_size = shard_size
        # 总行数缓存到 .meta 文件，避免每次启动扫 2.5 GB
        meta_path = en_path + '.meta'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self._total = int(f.read().strip())
        else:
            self._total = sum(1 for _ in open(en_path, 'rb'))
            with open(meta_path, 'w') as f:
                f.write(str(self._total))
        self._train_end = self._total - val_offset
        self._num_shards = (self._train_end + shard_size - 1) // shard_size
        logger.info(f"UN 语料: 总计 {self._total:,} 行, "
                    f"训练 {self._train_end:,} 行, "
                    f"{self._num_shards} shards (每片 {shard_size:,})")

    @property
    def num_shards(self):
        return self._num_shards

    def get_shard(self, shard_idx):
        """读取第 shard_idx 个分片，返回 [(en_tokens, zh_tokens), ...]。"""
        start = shard_idx * self.shard_size
        end = min(start + self.shard_size, self._train_end)

        buffer = []
        with open(self.en_path, 'r', encoding='utf-8') as f_en, \
             open(self.zh_path, 'r', encoding='utf-8') as f_zh:
            for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
                if i < start:
                    continue
                if i >= end:
                    break
                en_text = en_line.strip()
                zh_text = zh_line.strip()
                if not en_text or not zh_text:
                    continue
                en_tokens = en_text.split()
                zh_tokens = [t for t in jieba.cut(zh_text) if t.strip()]
                if not en_tokens or not zh_tokens:
                    continue
                buffer.append((en_tokens, zh_tokens))

        import random
        random.shuffle(buffer)
        return buffer


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


def load_model(checkpoint_path, model_init, input_shapes):
    try:
        with open(checkpoint_path, "rb") as f:
            params = pickle.load(f)
        # 当前代码期望 params 为 4 元组 (enc_params, enc_bn, dec_params, dec_bn)
        if not (isinstance(params, tuple) and len(params) == 4):
            logger.warning(
                f"检查点格式不兼容（期望 4 元组，得到 "
                f"{type(params).__name__} 长度 {len(params) if isinstance(params, tuple) else 'N/A'}），重新初始化模型"
            )
            rng = jax.random.PRNGKey(0)
            _, params = model_init(rng, input_shapes)
        else:
            logger.info(f"加载检查点: {checkpoint_path}")
        return params
    except FileNotFoundError:
        logger.warning(f"检查点不存在，初始化新模型")
        rng = jax.random.PRNGKey(0)
        _, params = model_init(rng, input_shapes)
        return params


def load_dataset_and_vocab(en_path, zh_path, vocab_dir, min_freq=3,
                           max_vocab_size_src=55000, max_vocab_size_tgt=55000,
                           max_length=64, batch_size=64,
                           vocab_sample_lines=200_000, val_lines=5000):
    """加载 UN 平行语料库 — 词表采样 + 流式训练。

    词表从前 vocab_sample_lines 行采样构建（首次运行，之后从缓存加载）。
    训练数据通过 UNIterableDataset 流式读取，不整批加载到内存。
    验证集固定从末尾取 val_lines 行。

    返回 (src_vocab, tgt_vocab, train_dataset, val_loader)。
    """
    os.makedirs(vocab_dir, exist_ok=True)
    src_vocab_path = os.path.join(vocab_dir, "src_vocab.json")
    tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.json")

    # ── 词表：采样构建 / 从缓存加载 ──
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        logger.info(f"采样前 {vocab_sample_lines:,} 行构建词表...")
        en_sents, cn_sents = load_un_corpus(
            en_path, zh_path, max_length,
            max_lines=vocab_sample_lines,
        )
        src_vocab = Vocabulary().build_from_sentences(en_sents, min_freq, max_vocab_size_src)
        tgt_vocab = Vocabulary().build_from_sentences(cn_sents, min_freq, max_vocab_size_tgt)
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)
        logger.info(f"词表已缓存: EN={len(src_vocab):,}, ZH={len(tgt_vocab):,}")

    # ── 验证集：从尾部取固定行数 ──
    logger.info(f"加载验证集（末尾 {val_lines} 行）...")
    # 复用 UNIterableDataset 的缓存行数统计
    meta_path = en_path + '.meta'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            _total = int(f.read().strip())
    else:
        _total = sum(1 for _ in open(en_path, 'rb'))
    skip_to = max(0, _total - val_lines)
    val_en, val_zh = [], []
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(zh_path, 'r', encoding='utf-8') as f_zh:
        for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
            if i < skip_to:
                continue
            en_tokens = en_line.strip().split()
            zh_tokens = [t for t in jieba.cut(zh_line.strip()) if t.strip()]
            if en_tokens and zh_tokens:
                val_en.append(en_tokens)
                val_zh.append(zh_tokens)
    logger.info(f"验证集: {len(val_en)} 句对")

    val_dataset = TranslationDataset(val_en, val_zh)
    def collate_wrapper(b):
        return collate_fn(b, src_vocab, tgt_vocab, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_wrapper)

    # ── 训练：分片流式 Dataset ──
    shard_size = 200000
    train_dataset = UNIterableDataset(
        en_path, zh_path,
        val_offset=val_lines, shard_size=shard_size,
    )

    logger.info(f"验证集: val_batches={len(val_loader)}")
    return src_vocab, tgt_vocab, train_dataset, val_loader


def main():
    n_heads = 8
    head_dim = 64
    embed_dim = n_heads * head_dim
    hidden_dim = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_length = 64
    block_size = 64  # FlashAttention 分块大小，等于 max_length 即单块模式
    batch_size = 64

    # ── 训练配置 ──
    max_epochs = 200
    warmup_steps = 4000
    early_stop_patience = 5      # 验证不降连续 N 次即停

    np.random.seed(42)
    random.seed(42)

    # ── UN 平行语料库 ──
    en_file = "/mnt/e/data/UNv1.0.en-zh/en-zh/UNv1.0.en-zh.en"
    zh_file = "/mnt/e/data/UNv1.0.en-zh/en-zh/UNv1.0.en-zh.zh"
    src_vocab, tgt_vocab, train_dataset, val_loader = load_dataset_and_vocab(
        en_file, zh_file, TRANSLATE_DIR, min_freq=3,
        max_vocab_size_src=55000, max_vocab_size_tgt=55000,
        max_length=max_length, batch_size=batch_size,
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

    src_shape = (batch_size, max_length)
    tgt_shape = (batch_size, max_length)
    output_shape, _ = model_init(jax.random.PRNGKey(0), (src_shape, tgt_shape))
    print(f"模型输出形状: {output_shape}")

    checkpoint_path = os.path.join(TRANSLATE_DIR, "model_un_en_zh.pkl")
    params = load_model(checkpoint_path, model_init, (src_shape, tgt_shape))

    # 学习率调度（按 epoch × 实际每轮步数估算总步数）
    steps_per_epoch = train_dataset.num_shards * (train_dataset.shard_size // batch_size)
    max_steps = max_epochs * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-4,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=1e-5,
    )
    # 优化器: 梯度裁剪 + AdamW
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.0001),
    )
    opt_state = optimizer.init(params)
    train_step_fn = make_train_step(model_apply, optimizer, src_vocab, tgt_vocab)
    val_step_fn = make_val_step(model_apply, src_vocab, tgt_vocab)

    logger.info(f"开始训练：max_epochs={max_epochs}, steps/epoch≈{steps_per_epoch}, "
                f"warmup_steps={warmup_steps}")

    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(TRANSLATE_DIR, "model_un_en_zh_best.pkl")
    nan_recovery_count = 0
    no_improve_count = 0
    global_step = 0
    train_losses = []      # 每个 shard 的 loss（用于绘图）

    rng = jax.random.PRNGKey(0)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for shard_idx in range(train_dataset.num_shards):
            shard_start = time.time()
            shard_data = train_dataset.get_shard(shard_idx)

            shard_loss_sum = 0.0
            shard_batches = 0
            batch = []
            for en_sent, zh_sent in shard_data:
                batch.append((en_sent, zh_sent))
                if len(batch) >= batch_size:
                    src_mat, tgt_mat = collate_fn(batch, src_vocab, tgt_vocab, max_length)
                    rng, step_rng = jax.random.split(rng)
                    params, opt_state, loss, pred, grad_norm = train_step_fn(
                        params, (src_mat, tgt_mat), opt_state, step_rng)
                    global_step += 1
                    batch = []

                    # NaN/Inf 检测（必须在累加 loss 之前）
                    if jnp.isnan(loss) or jnp.isinf(loss):
                        logger.error(f"epoch {epoch}, shard {shard_idx+1}, "
                                     f"step {global_step}: loss 为 NaN/Inf，恢复检查点")
                        nan_recovery_count += 1
                        try:
                            with open(checkpoint_path, "rb") as f:
                                params = pickle.load(f)
                            opt_state = optimizer.init(params)
                            logger.warning(f"已从 {checkpoint_path} 恢复参数")
                        except FileNotFoundError:
                            logger.error("无可用检查点，跳过此 batch")
                        continue

                    shard_loss_sum += loss
                    shard_batches += 1

            shard_avg_loss = float(shard_loss_sum / max(shard_batches, 1))
            epoch_loss_sum += shard_loss_sum
            epoch_batches += shard_batches
            train_losses.append(shard_avg_loss)

            # ── 每片结束：打印 + 保存模型 + 画曲线 + 翻译示例 ──
            print(f"  epoch {epoch:3d}/{max_epochs}, shard {shard_idx+1:3d}/{train_dataset.num_shards} | "
                  f"loss={shard_avg_loss:.4f} | batches={shard_batches} | "
                  f"{time.time()-shard_start:.1f}s")

            with open(checkpoint_path, "wb") as f:
                pickle.dump(params, f)

            translate_examples(params, src_vocab, tgt_vocab,
                               model_encode, model_decode, max_length)

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=0.8)
            plt.xlabel('Shard (across epochs)')
            plt.ylabel('Loss')
            plt.title(f'UN EN→ZH Translation — Loss (epoch {epoch}, shard {shard_idx+1})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(TRANSLATE_DIR, 'training_curve.png'), dpi=150)
            plt.close()

        # ── 每 epoch 结束：验证 + 翻译示例 ──
        epoch_time = time.time() - epoch_start
        avg_loss = float(epoch_loss_sum / max(epoch_batches, 1))
        logger.info(f"epoch {epoch}/{max_epochs} | loss={avg_loss:.4f} | "
                    f"steps={global_step} | {epoch_time:.1f}s")

        val_loss = validate_epoch(val_step_fn, params, val_loader)
        logger.info(f"epoch {epoch}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            with open(best_checkpoint_path, "wb") as f:
                pickle.dump(params, f)
            logger.info(f"epoch {epoch}: ★ 新最佳模型 (val_loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            logger.info(f"epoch {epoch}: 连续 {no_improve_count} 次验证未改善 "
                        f"(patience={early_stop_patience})")

        # 早停
        if no_improve_count >= early_stop_patience:
            logger.info(f"早停触发：验证 loss 连续 {early_stop_patience} 次未改善，"
                        f"最佳 val_loss={best_val_loss:.4f}")
            break

        print(f"Epoch {epoch}/{max_epochs} 完成，耗时 {epoch_time:.2f}s")
        print("=" * 50)

    print(f"\n训练结束。最佳 val_loss={best_val_loss:.4f}，NaN 恢复次数: {nan_recovery_count}")


if __name__ == "__main__":
    main()
