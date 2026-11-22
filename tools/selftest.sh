#!/usr/bin/env bash
# Manual smoke test for the runner. Needs the model, the tokenizer and ct2/.
#
#   PY=/path/to/python ./tools/selftest.sh
#
# Covers the cases that broke before: a long row over the position limit, mixed
# long/short/empty rows, CRLF and an empty file. Exit code is the failure count.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$HERE"

fails=0
pass() { printf 'ok   %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }

translate() {  # translate <src> <out> [extra args...]
  local src="$1" out="$2"; shift 2
  ./run.sh "$PY" translate_gpu.py --model models --hf-tokenizer tokenizer \
    --src "$src" --out "$out" --batch-size 64 "$@" >"$WORK/log" 2>&1
}

# Two long rows far over the 512-token position limit, plus short, empty and CRLF rows.
"$PY" - "$WORK/mixed.txt" <<'PY'
import sys
cat = "bu kedi bahçede uyuyor ve " * 120 + "SONKELIME"
dog = "bir köpek bahçede koşuyor ve " * 120 + "SONKELIME"
rows = ["Merhaba.", cat, "", "Görüşürüz.", dog]
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(rows) + "\n")
PY
printf 'Merhaba.\r\nGörüşürüz.\r\n' >"$WORK/crlf.txt"
: >"$WORK/empty.txt"

./run.sh ./tools/hip_smoke >"$WORK/smoke" 2>&1 && grep -q 'err=no error' "$WORK/smoke" \
  && pass 'device' || fail 'device'

if translate "$WORK/mixed.txt" "$WORK/mixed.out"; then
  in=$(wc -l <"$WORK/mixed.txt")
  out=$(wc -l <"$WORK/mixed.out")
  empty=$(sed -n '3p' "$WORK/mixed.out" | wc -c)
  [[ $in -eq $out ]] && pass 'row count' || fail "row count: $in in, $out out"
  [[ $empty -eq 1 ]] && pass 'empty row kept' || fail 'empty row not kept'
  [[ $(sed -n '2p' "$WORK/mixed.out" | wc -w) -gt 20 ]] \
    && pass 'long row translated' || fail 'long row too short'
  [[ $(sed -n '5p' "$WORK/mixed.out") != "$(sed -n '2p' "$WORK/mixed.out")" ]] \
    && pass 'rows aligned' || fail 'rows shifted'
else
  fail 'long row crashed'
fi

translate "$WORK/crlf.txt" "$WORK/crlf.out" && [[ $(wc -l <"$WORK/crlf.out") -eq 2 ]] \
  && pass 'crlf' || fail 'crlf'

translate "$WORK/empty.txt" "$WORK/empty.out" && [[ ! -s "$WORK/empty.out" ]] \
  && pass 'empty file' || fail 'empty file'

# Short rows must be byte-identical with and without splitting.
head -50 bench/rows.tsv >"$WORK/short.txt" 2>/dev/null \
  && translate "$WORK/short.txt" "$WORK/a.out" \
  && translate "$WORK/short.txt" "$WORK/b.out" --no-split \
  && cmp -s "$WORK/a.out" "$WORK/b.out" \
  && pass 'split does not touch short rows' || fail 'split changes short rows'

# Multilingual path: language prefixes take a token slot each, so the split
# budget must shrink rather than overflow.
if [[ -d models/nllb && -d tokenizer/nllb ]]; then
  # int8 collapses into repeats on NLLB now and then; int8_float16 is stable.
  if ./run.sh "$PY" translate_gpu.py --model models/nllb --hf-tokenizer tokenizer/nllb \
    --src-prefix tur_Latn --tgt-prefix fin_Latn --src "$WORK/mixed.txt" \
    --out "$WORK/nllb.out" --batch-size 64 --compute-type int8_float16 >"$WORK/log" 2>&1; then
    [[ $(wc -l <"$WORK/nllb.out") -eq $(wc -l <"$WORK/mixed.txt") ]] \
      && pass 'nllb prefixes + split' || fail 'nllb row count'
    grep -q '_Latn' "$WORK/nllb.out" && fail 'nllb prefix leaked' \
      || pass 'nllb prefix stripped'
  else
    fail 'nllb crashed'
  fi
else
  printf 'skip nllb (models/nllb or tokenizer/nllb missing)\n'
fi

printf '%d failure(s)\n' "$fails"
exit "$fails"
