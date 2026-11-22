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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 encoder-decoder model dir")
    ap.add_argument("--hf-tokenizer", required=True, help="matching HF tokenizer dir (source.spm, vocab.json)")
    ap.add_argument("--src", required=True, help="source text, one row per line")
    ap.add_argument("--out", required=True, help="output text file")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--compute-type", default="int8", choices=["int8", "float16", "float32", "bfloat16"])
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

    with tempfile.TemporaryDirectory() as tmp:
        src_tok = Path(tmp) / "src.tok"
        out_tok = Path(tmp) / "out.tok"
        with src_tok.open("w", encoding="utf-8") as f:
            for row in rows:
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
            tgt_tok.write_text("\n".join(" ".join(tgt_prefix) for _ in rows) + "\n", encoding="utf-8")
            cmd += ["--tgt", str(tgt_tok)]

        subprocess.run(cmd, check=True)

        with out_tok.open(encoding="utf-8") as f:
            hypotheses = [line.rstrip("\r\n").split(" ") for line in f]

        if len(hypotheses) != len(rows):
            raise SystemExit(f"row mismatch: {len(rows)} in, {len(hypotheses)} out")

        with open(args.out, "w", encoding="utf-8") as f:
            for tokens in hypotheses:
                if tgt_prefix and tokens[: len(tgt_prefix)] == tgt_prefix:
                    tokens = tokens[len(tgt_prefix):]
                ids = tokenizer.convert_tokens_to_ids([t for t in tokens if t])
                f.write(tokenizer.decode(ids, skip_special_tokens=True) + "\n")

    print(f"{len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
