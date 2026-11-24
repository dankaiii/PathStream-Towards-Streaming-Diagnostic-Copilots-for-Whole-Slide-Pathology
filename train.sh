#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_THREADING_LAYER=GNU TOKENIZERS_PARALLELISM=false
exec python -m train.run "$@"
