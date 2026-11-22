# Kolmannen osapuolen lisenssit

Oma koodi tässä repossa on MIT (ks. [`LICENSE`](LICENSE)). Mukana on kuitenkin
johdannaista ja kolmannen osapuolen aineistoa:

| Kohde | Lisenssi | Huom |
|---|---|---|
| `patches/ct2-hip-gfx803.patch` | MIT | Patch CTranslate2:een (OpenNMT), sama lisenssi; `ct2/`-build on sen johdannainen |
| `tokenizer/` | CC BY 4.0 | Peräisin mallista `koodikraft/koodikraft_opus_tr_fi` |
| `bench/rows.tsv`, `bench/ref.fi.tsv`, `bench/*.tok` | projektin oma | Johdettu TR→FI-muistista (`smart_tm_export.tsv`); sisältää OPUS/OpenSubtitles-aineistoa |

`models/` (ei versioissa) noudetaan `tools/prepare_model.py`llä valitsemastasi
mallista (`--hf-model`). Mukana oleva esimerkkitokenisaattori ja README:n
mittaukset ovat mallista `koodikraft/koodikraft_opus_tr_fi`, joka on **CC BY 4.0**.
Mallin kortissa on sen oma attribuutiovaatimus.

Huom: `koodikraft/opus-mt-tr-fi-2020-04-12` -reposta puuttuvat embedding-painot
(`model.encoder.embed_tokens` / `model.decoder.embed_tokens`), joten transformers 5
alustaa ne satunnaisesti ja tuotos on pelkkää pistejonoa. Siksi esimerkissä
käytetään `koodikraft_opus_tr_fi`-mallia.

CC BY 4.0 vaatii tekijän nimeämisen ja lisenssiviittauksen, jos aineistoa jaetaan
edelleen. MIT-koodin osalta riittää `LICENSE`-tiedoston säilyttäminen.
