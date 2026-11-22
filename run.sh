#!/usr/bin/env bash
# Activate the ROCm 5.7 userspace stack for gfx803 (Polaris) and run a command.
#
#   ./run.sh ./ct2/ct2-translator --model models ...
#   ./run.sh python3 translate_gpu.py --model models ...
#
# The stack lives in $ROCM_STACK57 because / has no room and ROCm 6.4 dropped
# gfx803 kernels from rocBLAS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK="${ROCM_STACK57:-/srv/rocm-root/stack57}"

export LD_LIBRARY_PATH="$STACK/lib:$HERE/ct2:${LD_LIBRARY_PATH:-}"
# The build targets gfx803 natively; the override is belt-and-braces for drivers
# that report a different gfx name.
export HSA_OVERRIDE_GFX_VERSION=8.0.3

exec "$@"
