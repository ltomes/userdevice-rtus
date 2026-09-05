#!/usr/bin/env bash
# Generate 2x paired training data with REAL video-codec degradations.
#
# For each HR png:  HR -> (downscale 1/2, random filter) -> encode 1-frame
# H.264 (random CRF 20-32, yuv420p — the library's SD/web tier) -> decode
# back to PNG => LR. HR is copied cropped to a 4-multiple so 2x aligns.
# Every 20th image goes to val. Idempotent: skips existing outputs.
#
# The whole loop runs INSIDE one ffmpeg container (host has no ffmpeg;
# per-image containers would cost hours of startup overhead). Memory-capped
# and single-threaded encodes so it can share a machine with other work.
#
# Usage: bash gen_pairs_h264.sh <hr_dir> <out_root> [limit]
set -euo pipefail

HR_DIR=$(realpath "$1")
OUT=$(realpath "$2")
LIMIT="${3:-0}"

mkdir -p "$OUT"/{train,val}/{hr,lr}

docker run --rm --name gen-pairs --memory=4g --memory-swap=4g \
  -v "$HR_DIR":/in:ro -v "$OUT":/out \
  --entrypoint bash linuxserver/ffmpeg:latest -c '
set -euo pipefail
FILTERS=(bicubic bilinear lanczos area)
RANDOM=42
i=0
for hr in /in/*.png; do
  name=$(basename "$hr" .png)
  if (( i % 20 == 0 )); then split=val; else split=train; fi
  i=$((i+1))
  if [[ '"$LIMIT"' != 0 && $i -gt '"$LIMIT"' ]]; then break; fi
  lr_out="/out/$split/lr/$name.png"
  hr_out="/out/$split/hr/$name.png"
  flt=${FILTERS[$((RANDOM % 4))]}
  crf=$((20 + RANDOM % 13))
  [[ -f "$lr_out" && -f "$hr_out" ]] && continue
  ffmpeg -hide_banner -loglevel error -threads 1 -y -i "$hr" \
    -vf "crop=floor(iw/4)*4:floor(ih/4)*4,scale=iw/2:ih/2:flags=$flt,format=yuv420p" \
    -c:v libx264 -crf "$crf" -preset medium -threads 1 -f h264 "/tmp/$name.264"
  ffmpeg -hide_banner -loglevel error -threads 1 -y -f h264 -i "/tmp/$name.264" \
    -frames:v 1 "$lr_out"
  rm -f "/tmp/$name.264"
  ffmpeg -hide_banner -loglevel error -threads 1 -y -i "$hr" \
    -vf "crop=floor(iw/4)*4:floor(ih/4)*4" "$hr_out"
  (( i % 250 == 0 )) && echo "[gen_pairs] $i done" || true
done
echo "[gen_pairs] complete: $(ls /out/train/lr | wc -l) train, $(ls /out/val/lr | wc -l) val"
'
