#!/usr/bin/env bash
# Launch a training run on ms, memory-safe (this host serves the household;
# ZFS ARC also competes for RAM — hard caps are mandatory).
#
# Usage: bash run-train.sh <config.yml> [extra train.py args]
set -euo pipefail

BASE=$(cd "$(dirname "$0")" && pwd)
CFG=$(realpath "$1"); shift || true

FREE_GB=$(free -g | awk 'NR==2{print $7}')
if (( FREE_GB < 8 )); then
  echo "ABORT: only ${FREE_GB}G available on ms (need >=8 headroom)" >&2
  exit 1
fi

exec docker run --rm --name greyduck-train \
  --memory=12g --memory-swap=12g --shm-size=4g \
  --gpus all \
  -v "$BASE/traiNNer-redux:/workspace/traiNNer-redux" \
  -v "$BASE/datasets:/workspace/datasets" \
  -v "$BASE/pretrained:/workspace/pretrained:ro" \
  -v "$BASE/config:/workspace/config:ro" \
  -v "$BASE/experiments:/workspace/traiNNer-redux/experiments" \
  greyduck-train:dev \
  python train.py -opt "/workspace/config/$(basename "$CFG")" "$@"
