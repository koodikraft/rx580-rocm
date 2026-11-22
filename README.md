# rx580-rocm — CTranslate2 on AMD Polaris (ROCm 5.7)

**English** · [Suomi](#rx580-rocm--ctranslate2-amd-polarisilla-rocm-57)

CTranslate2 translation engine on an RX 580 8GB (`gfx803`, GCN4) under Debian 13.
ROCm has no official Polaris support, so the stack is built by hand against ROCm
5.7 (see [Why 5.7](#why-rocm-57-and-not-64)).

Any CTranslate2 encoder-decoder model with its matching Hugging Face tokenizer
runs. No model is bundled: set `DEFAULT_MODEL` or pass `--hf-model`. Measurements
use TR→FI OPUS-MT: 117 rows/s (CLI), 97–98 rows/s end to end, BLEU 12.39 / chrF
32.48 — the same quality as CPU (12.54 / 32.70). One card only.

## Quick start

Needs the ROCm 5.7 stack at `$ROCM_STACK57` (default `/srv/rocm-root/stack57`, see
[Installing the stack](#installing-the-rocm-stack)) and a Python with
`transformers` + `ctranslate2`, e.g.
`/srv/ai-agent/workspaces/aitranskriptio/backend/venv/bin/python` — used as
`python` below.

```bash
python tools/prepare_model.py --hf-model koodikraft/koodikraft_opus_tr_fi  # → models/, tokenizer/
bash tools/build_ct2.sh                                                   # → ct2/
./run.sh ./tools/hip_smoke                                                # dev=1, err=no error
PY=python tools/selftest.sh                                               # regression cases, exit code = failures
```

`ct2/` is generated. If you copy the binary by hand, keep the symlink
`ct2/libctranslate2.so.4` → `libctranslate2.so.4.9.0` (startup fails without it)
and set the `RUNPATH` on **both** `ct2/ct2-translator` and
`ct2/libctranslate2.so.4.9.0`; then check
`ldd ct2/ct2-translator | grep 'not found'`.

## Use

Simplest:

```bash
printf 'Merhaba.\nGörüşürüz.\n' > /tmp/in.txt
./run.sh python translate_gpu.py --model models --hf-tokenizer tokenizer \
  --src /tmp/in.txt --out /tmp/out.txt
```

Batch (keep it batched — at batch 1 the GPU is 12× slower than CPU):

```bash
./run.sh python translate_gpu.py --model models --hf-tokenizer tokenizer \
  --src rows.txt --out out.txt --batch-size 256
```

Other models: give them their own directories and pass `--src-prefix` /
`--tgt-prefix` for multilingual ones (NLLB, M2M100); without the target prefix a
multilingual model may pick the wrong language. Decoder-only models (Llama, Qwen)
do not work.

```bash
python tools/prepare_model.py --hf-model facebook/nllb-200-distilled-600M \
  --model-dir models/nllb --tokenizer-dir tokenizer/nllb
./run.sh python translate_gpu.py --model models/nllb --hf-tokenizer tokenizer/nllb \
  --src-prefix tur_Latn --tgt-prefix fin_Latn --src rows.txt --out out.txt \
  --compute-type int8_float16 --batch-size 256
```

**NLLB is not reliable with `int8` on the GPU.** With a batch over 1 the decoder
sometimes collapses into repeated tokens (`ga ga ga ga …`, `▁( ▁( ▁(`); roughly
0–1 row in 100 with the settings above. `int8` output is also not fully
repeatable — the same input gives different rows between runs, while OPUS-MT is
byte-identical every time. Use `int8_float16` or `float16` (0/10 failed runs,
repeatable) and keep the batch at 256 or below: 512 dies with
`CUDA failed with error out of memory` (large vocabulary, beam 4). The measured
OPUS-MT numbers are not affected.

Empty input gives empty output; the row count is checked. CRLF is fine.

Rows over the model limit (512 tokens) are split at sentence ends and joined
back (rows stay 1:1). `--no-split` turns it off, `--max-source-tokens` changes the
limit. Short rows are untouched.

Scoring: `python tools/score.py --ref bench/ref.fi.tsv --hf-tokenizer tokenizer out.txt`
(data: [bench/README](bench/README.md)).

## Why ROCm 5.7 and not 6.4

6.4's `rocBLAS` ships no gfx803 kernels (`908, 90a, 942, 1030, 110x, 120x`), so it
fails with `TensileLibrary.dat: Illegal seek for GPU arch : gfx803`; 5.7 has 72
gfx803 files. 6.4 and 5.7 do not mix (`librocblas.so.4` vs `.3`), so the whole
stack is 5.7.

## Limits

- One card: the B550 x4 slot does not forward PCIe atomics, KFD skips it.
- The HIP build has no CPU GEMM backend, so `--device cpu` fails (CPU numbers come
  from the pip wheel, which is a CUDA build).
- ROCm 5.7 is retired; research bench, not a production path.

## Installing the ROCm stack

```bash
# 6.4 runtime + 5.7 stack unpacked into $ROCM_STACK57 (no dpkg: Debian 13 dependency hell)
# deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/5.7 jammy main
export ROCM_STACK57=/path/to/stack57      # default /srv/rocm-root/stack57
export CT2_SRC=/path/to/CTranslate2       # default /srv/rocm-test/ct2
ln -sfn "$(clang++-17 -print-resource-dir)/amdgcn" "$ROCM_STACK57/amdgcn"
bash tools/build_ct2.sh                   # compiler must be clang++-17
```

The patch fixes `__syncwarp`, the 5.7 `hipblasDatatype_t` signature and
`std::nearbyint` deduction. ROCm clang 19 pulls 6.4 libraries and the CMake import
targets pull 6.4 headers — copy those into `$ROCM_STACK57`.

MIT — see [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).

---

# rx580-rocm — CTranslate2 AMD Polarisilla (ROCm 5.7)

[English](#rx580-rocm--ctranslate2-on-amd-polaris-rocm-57) · **Suomi**

CTranslate2-käännösmoottori RX 580 8GB:llä (`gfx803`, GCN4) Debian 13:ssa. ROCm ei
tue Polarisia virallisesti, joten pino rakennetaan käsin ROCm 5.7:tä vasten (ks.
[Miksi 5.7](#miksi-rocm-57-eikä-64)).

Mikä tahansa CTranslate2-encoder-decoder-malli ja sitä vastaava Hugging Face
tokenisaattori käy. Mallia ei ole mukana: aseta `DEFAULT_MODEL` tai anna
`--hf-model`. Mittaukset ovat TR→FI OPUS-MT:llä: 117 riviä/s (CLI), 97–98 riviä/s
päästä päähän, BLEU 12,39 / chrF 32,48 — sama laatu kuin CPU:lla (12,54 / 32,70).
Vain yksi kortti.

## Pika-aloitus

Vaatii ROCm 5.7 -pinon polussa `$ROCM_STACK57` (oletus `/srv/rocm-root/stack57`, ks.
[Pinon asennus](#rocm-pinon-asennus)) ja Pythonin, jossa on `transformers` +
`ctranslate2` (testattu:
`/srv/ai-agent/workspaces/aitranskriptio/backend/venv/bin/python`).

```bash
python tools/prepare_model.py --hf-model koodikraft/koodikraft_opus_tr_fi  # → models/, tokenizer/
bash tools/build_ct2.sh                                                   # → ct2/
./run.sh ./tools/hip_smoke                                                # dev=1, err=no error
PY=python tools/selftest.sh                                               # regressiot, exit = virheiden määrä
```

`ct2/` generoidaan. Jos kopioit binäärin käsin, säilytä symlinkki
`ct2/libctranslate2.so.4` → `libctranslate2.so.4.9.0` (ilman sitä käynnistys
kaatuu) ja aseta `RUNPATH` **molempiin**: `ct2/ct2-translator` ja
`ct2/libctranslate2.so.4.9.0`. Tarkista lopuksi
`ldd ct2/ct2-translator | grep 'not found'`.

## Käyttö

Yksinkertaisin:

```bash
printf 'Merhaba.\nGörüşürüz.\n' > /tmp/in.txt
./run.sh python translate_gpu.py --model models --hf-tokenizer tokenizer \
  --src /tmp/in.txt --out /tmp/out.txt
```

Eränä (pidä eräajona — batch 1:llä GPU on 12× CPU:ta hitaampi):

```bash
./run.sh python translate_gpu.py --model models --hf-tokenizer tokenizer \
  --src rivit.txt --out tulos.txt --batch-size 256
```

Muut mallit: anna niille omat hakemistot ja monikielisille (NLLB, M2M100)
`--src-prefix` / `--tgt-prefix`; ilman kohdeprefixiä malli voi valita väärän
kielen. Dekooderimallit (Llama, Qwen) eivät toimi.

```bash
python tools/prepare_model.py --hf-model facebook/nllb-200-distilled-600M \
  --model-dir models/nllb --tokenizer-dir tokenizer/nllb
./run.sh python translate_gpu.py --model models/nllb --hf-tokenizer tokenizer/nllb \
  --src-prefix tur_Latn --tgt-prefix fin_Latn --src rivit.txt --out tulos.txt \
  --compute-type int8_float16 --batch-size 256
```

**NLLB ei ole luotettava `int8`:lla GPU:lla.** Yli 1:n eräkoolla dekooderi
romahtaa toisinaan toistojonoksi (`ga ga ga ga …`, `▁( ▁( ▁(`); yllä olevilla
asetuksilla noin 0–1 riviä sadasta. `int8`-tulos ei myöskään ole täysin
toistettava — sama syöte antaa eri rivejä ajojen välillä, kun taas OPUS-MT on
aina tavu-identtinen. Käytä `int8_float16`:ää tai `float16`:ta (0/10 epäonnistui,
toistettava) ja pidä eräkoko enintään 256:ssa: 512 kaatuu virheeseen
`CUDA failed with error out of memory` (iso sanasto, beam 4). OPUS-MT:n mitatut
luvut eivät muutu.

Tyhjä syöte → tyhjä tulos; rivimäärä tarkistetaan. CRLF käy.

Mallin positiorajan (512 tokenia) ylittävät rivit pilkotaan lauseenlopuista ja
yhdistetään takaisin (rivimäärä 1:1). `--no-split` poistaa pilkkomisen,
`--max-source-tokens` vaihtaa rajan. Lyhyet rivit säilyvät ennallaan.

Pisteytys: `python tools/score.py --ref bench/ref.fi.tsv --hf-tokenizer tokenizer tulos.txt`
(aineisto: [bench/README](bench/README.md)).

## Miksi ROCm 5.7 eikä 6.4

6.4:n `rocBLAS` ei toimita gfx803-ytimiä (`908, 90a, 942, 1030, 110x, 120x`), joten
ajo kaatuu virheeseen `TensileLibrary.dat: Illegal seek for GPU arch : gfx803`;
5.7:ssä on 72 gfx803-tiedostoa. 6.4 ja 5.7 eivät sekoitu (`librocblas.so.4` vs
`.3`), siksi koko pino on 5.7.

## Rajoitteet

- Vain yksi kortti: B550:n x4-slotti ei välitä PCIe-atomiceja, KFD ohittaa sen.
- HIP-buildissa ei ole CPU-GEMM-backendiä, joten `--device cpu` ei toimi (CPU-luvut
  ovat pip-pyörästä, joka on CUDA-build).
- ROCm 5.7 on eläkkeellä; tutkimuspenkki, ei tuotantopolku.

## ROCm-pinon asennus

```bash
# 6.4 runtime + 5.7 pino purettuina $ROCM_STACK57:ään (ei dpkg: Debian 13:n riippuvuudet)
# deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/5.7 jammy main
export ROCM_STACK57=/polku/stack57        # oletus /srv/rocm-root/stack57
export CT2_SRC=/polku/CTranslate2         # oletus /srv/rocm-test/ct2
ln -sfn "$(clang++-17 -print-resource-dir)/amdgcn" "$ROCM_STACK57/amdgcn"
bash tools/build_ct2.sh                   # kääntäjänä clang++-17
```

Patch korjaa `__syncwarp`in, 5.7:n `hipblasDatatype_t`-allekirjoituksen ja
`std::nearbyint`-deduktion. ROCm clang 19 vetää 6.4-kirjastot ja CMake-targetit
6.4-headerit — kopioi ne `$ROCM_STACK57`:ään.

MIT — ks. [LICENSE](LICENSE) ja [THIRD_PARTY.md](THIRD_PARTY.md).
