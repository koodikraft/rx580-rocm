#!/usr/bin/env python3
"""Fetch a HF translation model and convert it to CTranslate2 int8.

`models/` is not in git (weights). This script rebuilds it so a fresh clone can
run. Both outputs are needed: the CT2 model dir for the CLI and the tokenizer dir
for SentencePiece pre-tokenization.

Pick the model one of two ways:

1. set `DEFAULT_MODEL` below (commented examples are ready to enable), or
2. leave it `None` and pass `--hf-model` every time.

Any encoder-decoder model works. Output goes to `models/` and `tokenizer/`;
override with `--model-dir` / `--tokenizer-dir` to keep several models side by side.
`--force` overwrites, so give another model its own directories.

    python tools/prepare_model.py --hf-model <org/model>
    python tools/prepare_model.py --hf-model facebook/nllb-200-distilled-600M \
        --model-dir models/nllb --tokenizer-dir tokenizer/nllb
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# Set the model you use most, or leave None and always pass --hf-model.
# DEFAULT_MODEL = "koodikraft/koodikraft_opus_tr_fi"  # TR→FI OPUS-MT (the measured one)
# DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-fi"        # EN→FI OPUS-MT
# DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"  # multilingual NLLB-200
DEFAULT_MODEL: str | None = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf-model",
        help="HF model id or local path (required unless DEFAULT_MODEL is set)",
    )
    ap.add_argument(
        "--model-dir", default=str(HERE / "models"), help="CT2 output dir"
    )
    ap.add_argument(
        "--tokenizer-dir", default=str(HERE / "tokenizer"), help="tokenizer output dir"
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing dirs")
    args = ap.parse_args()

    hf_model = args.hf_model or DEFAULT_MODEL
    if not hf_model:
        ap.error("anna --hf-model tai aseta DEFAULT_MODEL skriptiin")

    model_dir = Path(args.model_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    import ctranslate2
    from transformers import AutoTokenizer

    if model_dir.is_dir() and (model_dir / "model.bin").is_file() and not args.force:
        print(f"{model_dir} on jo olemassa (--force korvaa)")
        return 1

    if args.force:
        for target in (model_dir, tokenizer_dir):
            if target.is_dir():
                shutil.rmtree(target)

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5 renames Marian's embeddings, so a checkpoint without explicit
    # encoder/decoder embed_tokens loads them randomly (output becomes "......").
    # Detect that and tell the user instead of producing a silently broken model.
    _warn_if_embeddings_missing(hf_model)

    converter = ctranslate2.converters.TransformersConverter(hf_model)
    converter.convert(str(model_dir), quantization="int8", force=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.save_pretrained(str(tokenizer_dir))

    print(f"valmis: {model_dir} (int8) ja {tokenizer_dir}")
    return 0


def _warn_if_embeddings_missing(hf_model: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    try:
        path = hf_hub_download(hf_model, "model.safetensors")
    except Exception:
        return  # gated repo, sharded weights or local path: nothing to check

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        keys = set(json.loads(f.read(header_len)))

    if "model.encoder.embed_tokens.weight" in keys:
        return
    print(
        f"VAROITUS: {hf_model} ei sisällä encoder/decoder embed_tokens-painoja.\n"
        "  transformers 5 alustaa ne satunnaisesti ja mallin tuotos on pelkkää\n"
        "  pistejonoa. Käytä repoa, jossa on model.shared.weight sidottuna\n"
        "  (esim. koodikraft/koodikraft_opus_tr_fi).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
