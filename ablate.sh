#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
preset="${1:?Usage: bash ablate.sh PRESET|all [training options]}"
shift
exec bash train.sh --ablation "$preset" "$@"
