# bench/ — miten data kerätään

Benchmark-aineisto ei ole gitissä (johdettu TR→FI-muistista, sisältää
OPUS/OpenSubtitles-rivejä). Tarvittavat tiedostot:

| Tiedosto | Sisältö |
|---|---|
| `rows.tsv` | 1500 turkkilaista lähderivä, yksi per rivi |
| `ref.fi.tsv` | samat 1500 riviä viitekääntäjänä |
| `rows.tok` | `rows.tsv` esitokenisoituna (vapaaehtoinen, nopeuttaa) |

Lähde on projektin `data/smart_tm_export.tsv` (TR- ja FI-sarakkeet). Ota 1500
peräkkäistä riviä ja kirjoita sarakkeet erikseen yllä oleviin nimiin. Rivien on
oltava 1:1-kohdakkain — `translate_gpu.py` pitää rivimäärän ja `score.py`
vertaa sitä viitteeseen.

Sama setillä mitatut luvut (README) olivat: GPU int8 batch 512 **117,3 r/s**,
BLEU 12,39 / chrF 32,48; CPU int8 16 s 59,6 r/s. Eri otos antaa hieman eri
BLEU:n, mutta nopeusluokka pysyy samana. Muilla malleilla absoluuttiset luvut
muuttuvat; setin rakenne (1:1 rivit) pysyy samana.

Ajaminen:

```bash
./run.sh <python> translate_gpu.py --model models --hf-tokenizer tokenizer \
  --src bench/rows.tsv --out bench/out.txt --batch-size 512
python tools/score.py --ref bench/ref.fi.tsv --hf-tokenizer tokenizer bench/out.txt
```

Tokenisoitu syöte kannattaa tehdä kerran, jos ajaa useita vertailuja:
`--src bench/rows.tok` ohittaa Python-tokenisoinnin ja antaa raa'an CLI:n
nopeuden (117 r/s), mutta silloin tulos on tokeneina ja `score.py` dekoodaa sen.
