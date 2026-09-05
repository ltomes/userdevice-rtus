#!/usr/bin/env bash
# Fetch open benchmark sets into ./testsets (gitignored). Never committed.
#
# STATUS: skeleton. The URLs below have NOT been executed or verified --
# fill in and pin a sha256 for each set on the first successful fetch,
# then delete this notice. Do not trust an unpinned fetch.
#
# Licence posture differs per set. Read docs/BENCHMARKS.md before
# redistributing ANY of these or quoting numbers from them.
set -euo pipefail
OUT="${1:-./testsets}"
mkdir -p "$OUT"

echo "Sets to acquire:"
echo "  Set5, Set14, BSD100, Urban100  -- research use; standard mirrors"
echo "  DIV2K validation               -- research use; official NTIRE source"
echo "  Manga109                       -- GATED: requires application + academic"
echo "                                    agreement. NOT scriptable, NOT"
echo "                                    redistributable. Acquire manually."
echo
echo "  Blender open movies (CC-BY)    -- Sintel, Tears of Steel, Cosmos"
echo "                                    Laundromat, Spring, Big Buck Bunny."
echo "                                    Freely redistributable. Use these to"
echo "                                    build the in-distribution, temporal"
echo "                                    eval set via gen_pairs_h264.sh."
echo
echo "NOT IMPLEMENTED -- see docs/BENCHMARKS.md."
exit 1
