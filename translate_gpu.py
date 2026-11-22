#!/usr/bin/env python3
"""SentencePiece-safe translation on AMD Polaris (gfx803) via HIP-built CTranslate2.

The wrapper is model-agnostic: give it any CTranslate2 encoder-decoder model and
its matching Hugging Face tokenizer. CTranslate2's own CLI splits text on spaces,
which is wrong for SentencePiece models, and the pip ctranslate2 wheel is a CUDA
build. This wrapper pre-tokenizes with the model's SentencePiece tokenizer and
drives the HIP-built `ct2-translator` binary, then detokenizes the output.

Every row is passed through the tokenizer, which already appends the `</s>`
end-of-sentence token. Keeping that token on blank rows is what makes the CLI
emit one output line per input line instead of collapsing them.

Environment (set by run.sh):  LD_LIBRARY_PATH -> ROCm 5.7 stack + ct2 dir
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CT2_BIN = HERE / "ct2/ct2-translator"

SENTENCE_END = (".", "!", "?", "…", ";", ":")
WORD_START = "▁"


def _split_row(row: str, tokenizer, limit: int) -> list[str]:
    """Cut a row into token chunks of at most `limit`, preferring sentence ends.

    The model has a hard position limit (512 for the example model) and the CT2
    CLI aborts as soon as one row exceeds it. Splitting at sentence ends keeps
    the pieces translatable; a word boundary is the fallback, a hard cut the last
    resort. Only rows over the limit are touched, so short rows are unchanged.
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(row))
    if len(tokens) <= limit:
        return [row]

    # Never start a chunk with the end-of-sentence token.
    if tokens and tokens[-1] in (tokenizer.eos_token, tokenizer.pad_token):
        tokens = tokens[:-1]
    # Re-encoding a chunk appends `</s>` again, so leave room for it: the CLI
    # limit applies to the written line, not to the chunk text.
    limit = max(limit - 1, 8)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + limit, len(tokens))
        if end < len(tokens):
            window = range(end, start + limit // 4, -1)
            sentence = next((i for i in window if tokens[i - 1].endswith(SENTENCE_END)), None)
            if sentence:
                end = sentence
            else:
                word = next((i for i in window if tokens[i].startswith(WORD_START)), None)
                if word:
                    end = word
        text = tokenizer.convert_tokens_to_string(tokens[start:end]).strip()
        if text:
            chunks.append(text)
        start = end
    return chunks or [row]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 encoder-decoder model dir")
    ap.add_argument("--hf-tokenizer", required=True, help="matching HF tokenizer dir (source.spm, vocab.json)")
    ap.add_argument("--src", required=True, help="source text, one row per line")
    ap.add_argument("--out", required=True, help="output text file")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--compute-type", default="int8", choices=["int8", "int8_float16", "float16", "float32", "bfloat16"])
    ap.add_argument(
        "--src-prefix",
        default=None,
        help="source language token, e.g. NLLB 'tur_Latn' (sets tokenizer.src_lang and is prepended once)",
    )
    ap.add_argument(
        "--tgt-prefix",
        default=None,
        help="token forced as target prefix (e.g. NLLB target language 'fin_Latn'); stripped from output",
    )
    ap.add_argument(
        "--no-split",
        action="store_true",
        help="do not split rows longer than the model limit; the CLI then aborts on them",
    )
    ap.add_argument(
        "--max-source-tokens",
        type=int,
        default=512,
        help="split rows longer than this many tokens (default: 512, the model limit)",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    tgt_prefix = args.tgt_prefix.split() if args.tgt_prefix else []
    if args.src_prefix:
        # Multilingual tokenizers (NLLB/M2M100) prepend the source language token
        # themselves when src_lang is set; the fast tokenizer used by transformers 5
        # reads the attribute, not the encode() keyword.
        tokenizer.src_lang = args.src_prefix
    with open(args.src, encoding="utf-8") as f:
        rows = [line.rstrip("\r\n") for line in f]

    # One working unit per source row keeps the row count 1:1. Long rows are cut
    # into chunks at sentence boundaries; each chunk is translated on its own and
    # the pieces are joined back. The CT2 CLI aborts on a row that exceeds the
    # model's position limit, so a single long row would otherwise kill the run.
    units = [(row, index) for index, row in enumerate(rows)]
    if not args.no_split:
        units = []
        limit = args.max_source_tokens - (1 if args.src_prefix else 0)  # prefix takes a slot
        for index, row in enumerate(rows):
            for chunk in _split_row(row, tokenizer, limit):
                units.append((chunk, index))

    with tempfile.TemporaryDirectory() as tmp:
        src_tok = Path(tmp) / "src.tok"
        out_tok = Path(tmp) / "out.tok"
        with src_tok.open("w", encoding="utf-8") as f:
            for row, _ in units:
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(row))
                if args.src_prefix and tokens[:1] != [args.src_prefix]:
                    # unknown prefix / tokenizer that ignores src_lang: prepend manually
                    tokens = [args.src_prefix, *tokens]
                f.write(" ".join(tokens) + "\n")

        cmd = [
            str(CT2_BIN),
            "--model", args.model,
            "--src", str(src_tok),
            "--out", str(out_tok),
            "--device", "cuda",
            "--compute_type", args.compute_type,
            "--beam_size", str(args.beam_size),
            "--batch_size", str(args.batch_size),
        ]
        if tgt_prefix:
            tgt_tok = Path(tmp) / "tgt.tok"
            tgt_tok.write_text("\n".join(" ".join(tgt_prefix) for _ in units) + "\n", encoding="utf-8")
            cmd += ["--tgt", str(tgt_tok)]

        subprocess.run(cmd, check=True)

        with out_tok.open(encoding="utf-8") as f:
            hypotheses = [line.rstrip("\r\n").split(" ") for line in f]

        if len(hypotheses) != len(units):
            raise SystemExit(f"row mismatch: {len(units)} in, {len(hypotheses)} out")

        texts: list[list[str]] = [[] for _ in rows]
        for (_, index), tokens in zip(units, hypotheses):
            if tgt_prefix and tokens[: len(tgt_prefix)] == tgt_prefix:
                tokens = tokens[len(tgt_prefix):]
            ids = tokenizer.convert_tokens_to_ids([t for t in tokens if t])
            texts[index].append(tokenizer.decode(ids, skip_special_tokens=True))

        with open(args.out, "w", encoding="utf-8") as f:
            for parts in texts:
                f.write(" ".join(p for p in parts if p) + "\n")

    print(f"{len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
