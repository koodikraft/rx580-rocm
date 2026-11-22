#!/usr/bin/env bash
# Rebuild the HIP CTranslate2 CLI for gfx803 into ct2/.
#
# Needs the ROCm 5.7 stack (see README: "ROCm-pinon asennus") and clang-17.
# The build tree stays in a temp dir; only the two artifacts land in ct2/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK="${ROCM_STACK57:-/srv/rocm-root/stack57}"
CT2_SRC="${CT2_SRC:-/srv/rocm-test/ct2}"
PATCH="$HERE/patches/ct2-hip-gfx803.patch"

[[ -d "$STACK/lib" ]] || { echo "puuttuu: $STACK (aja ROCm-pinon asennus)" >&2; exit 1; }
[[ -d "$CT2_SRC" ]] || { echo "puuttuu: $CT2_SRC (kloonaa CTranslate2)" >&2; exit 1; }

cd "$CT2_SRC"
if ! git apply --check -R "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
fi

cmake -B build -GNinja -DWITH_HIP=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx803 \
  -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 \
  -DCMAKE_HIP_COMPILER=clang++-17 \
  -DHIP_ROOT_DIR="$STACK" -DROCM_PATH="$STACK" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$STACK --rocm-device-lib-path=/usr/lib/llvm-17/lib/clang/17/amdgcn/bitcode" \
  -DCMAKE_CXX_FLAGS="-I$STACK/include"
cmake --build build -j"$(nproc)" --target ct2-translator

mkdir -p "$HERE/ct2"
cp -L build/cli/ct2-translator "$HERE/ct2/ct2-translator"
cp -L build/libctranslate2.so.4.9.0 "$HERE/ct2/"
ln -sf libctranslate2.so.4.9.0 "$HERE/ct2/libctranslate2.so.4"
ln -sf libctranslate2.so.4.9.0 "$HERE/ct2/libctranslate2.so"

# RUNPATH in BOTH files: the library resolves its own deps via its own RUNPATH.
command -v patchelf >/dev/null || { echo "puuttuu: patchelf" >&2; exit 1; }
patchelf --set-rpath "\$ORIGIN:$STACK/lib" "$HERE/ct2/ct2-translator"
patchelf --set-rpath "\$ORIGIN:$STACK/lib" "$HERE/ct2/libctranslate2.so.4.9.0"

echo "valmis: $HERE/ct2"
ldd "$HERE/ct2/ct2-translator" | grep 'not found' && exit 1 || true
