#!/usr/bin/env bash
# Launch a training run in the RTUS container image.
#
#   bash scripts/run-train.sh <config.yml> [extra train.py args]
#
# The memory cap is not optional. Dataloader workers dominate host RAM at
# roughly 1.2 GB each, so an uncapped run will OOM any machine that is doing
# anything else. See num_worker_per_gpu in the configs.
#
# Paths are taken from the environment so this works on any host:
#   RTUS_DATA_ROOT  datasets, pretrained weights, experiments  (default ./)
#   RTUS_IMAGE      container image to run        (default rtus-train:dev)
#   RTUS_MEMORY     hard memory cap               (default 12g)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT="${RTUS_DATA_ROOT:-$REPO}"
IMAGE="${RTUS_IMAGE:-rtus-train:dev}"
MEMORY="${RTUS_MEMORY:-12g}"

if [ $# -lt 1 ]; then
  echo "usage: $0 <config.yml> [extra train.py args]" >&2
  exit 2
fi
CFG=$(realpath -- "$1"); shift
CFG_DIR=$(dirname -- "$CFG")
CFG_NAME=$(basename -- "$CFG")

FREE_GB=$(free -g | awk 'NR==2{print $7}')
if (( FREE_GB < 8 )); then
  echo "ABORT: only ${FREE_GB}G free (need >=8 GB headroom)" >&2
  exit 1
fi

for d in datasets pretrained experiments; do
  mkdir -p -- "$DATA_ROOT/$d"
done

# The config's own directory is mounted, rather than assuming it lives in the
# repo: passing a config from anywhere else used to fail silently, because only
# its basename survived into the container.
exec docker run --rm --name rtus-train \
  --memory="$MEMORY" --memory-swap="$MEMORY" --shm-size=4g \
  --gpus all \
  -v "$DATA_ROOT/datasets:/workspace/datasets" \
  -v "$DATA_ROOT/pretrained:/workspace/pretrained:ro" \
  -v "$DATA_ROOT/experiments:/workspace/traiNNer-redux/experiments" \
  -v "$CFG_DIR:/workspace/config:ro" \
  "$IMAGE" \
  python train.py -opt "/workspace/config/$CFG_NAME" "$@"
