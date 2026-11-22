#!/usr/bin/env python3
"""Score hypothesis files against a reference TSV (BLEU / chrF).

`--ref` defaults to bench/ref.fi.tsv, which is not in git — see bench/README.md
for how to build the benchmark set. Hypothesis files in bench/ are CT2 token
streams while translate_gpu.py output is already detokenized text; both work,
the format is detected from the `▁` marker. Nothing is written.

    python tools/score.py --ref bench/ref.fi.tsv --hf-tokenizer tokenizer /tmp/out.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sacrebleu
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-tokenizer", required=True, help="HF model dir with source/target.spm")
    ap.add_argument("--ref", default=str(HERE.parent / "bench/ref.fi.tsv"),
                    help="reference TSV; not in git, see bench/README.md")
    ap.add_argument("hyp", nargs="+", help="tokenized hypothesis files")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    refs = [line.rstrip("\r\n") for line in open(args.ref, encoding="utf-8")]

    for path in args.hyp:
        with open(path, encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]
        if any("▁" in line for line in lines):
            hyps = [
                tokenizer.decode(
                    tokenizer.convert_tokens_to_ids([t for t in line.split(" ") if t]),
                    skip_special_tokens=True,
                )
                for line in lines
            ]
        else:
            hyps = lines
        if len(hyps) != len(refs):
            print(f"{path}: row mismatch {len(hyps)} vs {len(refs)}")
            continue
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        ident = sum(1 for h in hyps if not h.strip())
        print(f"{Path(path).name}: BLEU {bleu:.2f} chrF {chrf:.2f} tyhjiä {ident}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
