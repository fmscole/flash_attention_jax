#!/usr/bin/env python
"""
AI Challenger 中英翻译数据集 — 清洗 & 转 TSV 脚本

从 AiChallenger 机器翻译数据集（train.en / train.zh）生成
与 translate_stax_flash.py 兼容的 TSV 格式。

数据来源: https://aistudio.baidu.com/datasetdetail/220848

用法:
  # 默认：清洗全部 10M 句对，输出 3M
  python scripts/prepare_ai_challenger.py

  # 指定输入路径（如果数据不在默认位置）
  python scripts/prepare_ai_challenger.py \
      --en /mnt/h/.../train.en \
      --zh /mnt/h/.../train.zh

  # 输出全部 10M（不截断）
  python scripts/prepare_ai_challenger.py --max-pairs 10000000

输出:
  data/ai_challenger_zh_en.tsv  (en\\tzh 格式，与 translate_stax_flash.py 兼容)
"""

import argparse
import os
import re
import sys
import time


# ── 清洗函数（与 translate_stax_flash.py 保持一致） ──────────────

def clean_english(text: str) -> str:
    """清洗英文文本。"""
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'([.,,!?()])', r' \1 ', text)
    text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1 '\2", text)
    text = text.replace('...', ' ... ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def clean_chinese(text: str) -> str:
    """清洗中文文本。"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9.,!?'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)
    text = text.replace(' . . .', ' ...')
    text = re.sub(r' ([.,?!])', r'\1', text)
    return text.strip()


# ── 默认路径（WSL 挂载） ─────────────────────────────────────────

AI_CHALLENGER_DIR = '/mnt/h/data_set/AiChallenger'
DEFAULT_EN_PATH = os.path.join(AI_CHALLENGER_DIR, 'train.en')
DEFAULT_ZH_PATH = os.path.join(AI_CHALLENGER_DIR, 'train.zh')
DEFAULT_OUT_PATH = 'data/ai_challenger_zh_en.tsv'
DEFAULT_VALID_EN_PATH = os.path.join(AI_CHALLENGER_DIR, 'valid.en-zh.en.sgm')
DEFAULT_VALID_ZH_PATH = os.path.join(AI_CHALLENGER_DIR, 'valid.en-zh.zh.sgm')
DEFAULT_VALID_OUT_PATH = 'data/ai_challenger_valid_zh_en.tsv'


# ── SGM 解析 ───────────────────────────────────────────────────

def parse_sgm(file_path: str) -> list[str]:
    """从 SGM 文件中提取 <seg id="N"> 标签内的文本。"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'<seg id="\d+">\s*(.*?)\s*</seg>', line)
            if m:
                texts.append(m.group(1))
    return texts


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Challenger 中英翻译数据集 - 清洗 & 转 TSV",
    )
    parser.add_argument('--en', default=DEFAULT_EN_PATH,
                        help=f'英文文件路径 (默认: {DEFAULT_EN_PATH})')
    parser.add_argument('--zh', default=DEFAULT_ZH_PATH,
                        help=f'中文文件路径 (默认: {DEFAULT_ZH_PATH})')
    parser.add_argument('--out', '-o', default=DEFAULT_OUT_PATH,
                        help=f'输出 TSV 路径 (默认: {DEFAULT_OUT_PATH})')
    parser.add_argument('--max-pairs', type=int, default=3_000_000,
                        help='最大输出句对数 (默认: 3,000,000)')
    parser.add_argument('--min-en-len', type=int, default=3,
                        help='英文最小词数 (默认: 3)')
    parser.add_argument('--max-en-len', type=int, default=80,
                        help='英文最大词数 (默认: 80)')
    parser.add_argument('--min-zh-len', type=int, default=3,
                        help='中文最小字符数 (默认: 3)')
    parser.add_argument('--max-zh-len', type=int, default=80,
                        help='中文最大字符数 (默认: 80)')
    parser.add_argument('--no-dedup', action='store_true',
                        help='跳过去重')
    parser.add_argument('--valid-en', default=DEFAULT_VALID_EN_PATH,
                        help=f'SGM 验证集英文文件 (默认: {DEFAULT_VALID_EN_PATH})')
    parser.add_argument('--valid-zh', default=DEFAULT_VALID_ZH_PATH,
                        help=f'SGM 验证集中文文件 (默认: {DEFAULT_VALID_ZH_PATH})')
    parser.add_argument('--out-valid', default=DEFAULT_VALID_OUT_PATH,
                        help=f'验证集输出 TSV (默认: {DEFAULT_VALID_OUT_PATH})')
    args = parser.parse_args()

    if not os.path.exists(args.en) or not os.path.exists(args.zh):
        print(f"错误: 找不到输入文件")
        print(f"  EN: {args.en}")
        print(f"  ZH: {args.zh}")
        print(f"请检查路径，或通过 --en / --zh 指定正确位置")
        sys.exit(1)

    # ── 统计总行数 ──
    print("统计行数...")
    with open(args.en, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"  共 {total_lines:,} 行")

    # ── 流式处理 ──
    t_start = time.time()
    written = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_long = 0
    seen = set()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    with open(args.en, 'r', encoding='utf-8') as f_en, \
         open(args.zh, 'r', encoding='utf-8') as f_zh, \
         open(args.out, 'w', encoding='utf-8') as f_out:

        for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
            if written >= args.max_pairs:
                break

            en_raw = en_line.strip()
            zh_raw = zh_line.strip()

            if not en_raw or not zh_raw:
                skipped_empty += 1
                continue

            en_clean = clean_english(en_raw)
            zh_clean = clean_chinese(zh_raw)

            if not en_clean or not zh_clean:
                skipped_empty += 1
                continue

            # 长度过滤
            en_words = len(en_clean.split())
            zh_chars = len(zh_clean.replace(' ', ''))

            if en_words < args.min_en_len or zh_chars < args.min_zh_len:
                skipped_short += 1
                continue
            if en_words > args.max_en_len or zh_chars > args.max_zh_len:
                skipped_long += 1
                continue

            # 去重
            if not args.no_dedup:
                key = (en_clean, zh_clean)
                if key in seen:
                    continue
                seen.add(key)

            f_out.write(f"{en_clean}\t{zh_clean}\n")
            written += 1

            if (i + 1) % 500000 == 0:
                elapsed = time.time() - t_start
                print(f"  进度: {i+1:,}/{total_lines:,} | "
                      f"已输出: {written:,} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    out_size = os.path.getsize(args.out)

    print(f"\n完成!")
    print(f"  总处理: {total_lines:,} 句对")
    print(f"  输出:   {written:,} 句对 ({out_size/1024/1024:.1f}MB)")
    print(f"  跳过:   空 {skipped_empty:,} | 过短 {skipped_short:,} | "
          f"过长 {skipped_long:,}")
    print(f"  耗时:   {elapsed:.1f}s ({total_lines/elapsed:,.0f} 行/秒)")
    print(f"  文件:   {args.out}")

    # ── 验证集（SGM → TSV）──
    if os.path.exists(args.valid_en) and os.path.exists(args.valid_zh):
        print(f"\n处理验证集...")
        en_texts = parse_sgm(args.valid_en)
        zh_texts = parse_sgm(args.valid_zh)
        print(f"  有效 EN: {len(en_texts)}, 有效 ZH: {len(zh_texts)}")

        os.makedirs(os.path.dirname(args.out_valid) or '.', exist_ok=True)
        written_valid = 0
        with open(args.out_valid, 'w', encoding='utf-8') as f:
            for en_raw, zh_raw in zip(en_texts, zh_texts):
                en_clean = clean_english(en_raw)
                zh_clean = clean_chinese(zh_raw)
                if not en_clean or not zh_clean:
                    continue
                en_words = len(en_clean.split())
                zh_chars = len(zh_clean.replace(' ', ''))
                if en_words < args.min_en_len or zh_chars < args.min_zh_len:
                    continue
                if en_words > args.max_en_len or zh_chars > args.max_zh_len:
                    continue
                f.write(f"{en_clean}\t{zh_clean}\n")
                written_valid += 1

        valid_size = os.path.getsize(args.out_valid)
        print(f"  验证集: {written_valid} 句对 ({valid_size/1024:.1f}KB) → {args.out_valid}")
    else:
        print(f"\n跳过验证集（文件不存在）")

    print(f"\n下一步：修改 translate_stax_flash.py 中的 dataset_path:")
    print(f'  dataset_path = "./{args.out}"')


if __name__ == '__main__':
    main()
