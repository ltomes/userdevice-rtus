#!/usr/bin/env bash
# Fetch the two model assets the pipeline needs but does not ship.
#
#   bash scripts/fetch_assets.sh [dest]
#
# dest defaults to $RTUS_DATA_ROOT/pretrained, or ./pretrained.
#
# Neither file is redistributed by this repository: both are third-party
# weights with their own terms. See docs/PROVENANCE.md.
set -euo pipefail

DEST="${1:-${RTUS_DATA_ROOT:-.}/pretrained}"
mkdir -p -- "$DEST"

echo "Fetching into $DEST"

# --- Teacher -----------------------------------------------------------------
# 4xNomosWebPhoto_RealPLKSR by Philip Hofmann (Phhofm), CC-BY-4.0.
# The tools expect this exact filename, so the rename is part of the contract,
# not a preference.
TEACHER="$DEST/teacher_4xNomosWebPhoto_RealPLKSR.pth"
if [ -f "$TEACHER" ]; then
  echo "  teacher: already present"
else
  echo "  teacher: downloading from Hugging Face (Phips/4xNomosWebPhoto_RealPLKSR)"
  python - "$TEACHER" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download

# The release ships more than one file; the PLKSR .pth is the teacher.
path = hf_hub_download(
    repo_id="Phips/4xNomosWebPhoto_RealPLKSR",
    filename="4xNomosWebPhoto_RealPLKSR.pth",
)
shutil.copyfile(path, sys.argv[1])
print(f"    -> {sys.argv[1]}")
PY
fi

# --- Face-gate detector ------------------------------------------------------
# SCRFD 2.5g (bnkps) from InsightFace, used by the BLOCKING face gate.
#
# NOT SCRIPTED, deliberately: InsightFace distributes its model zoo through
# links that have moved more than once, and a hardcoded URL that silently
# fetches the wrong file would corrupt a gate whose entire job is to be
# trustworthy. Obtain scrfd_2.5g_bnkps.onnx from the InsightFace model zoo
# and place it at the path below.
SCRFD="$DEST/scrfd_2.5g_bnkps.onnx"
if [ -f "$SCRFD" ]; then
  echo "  scrfd:   already present"
else
  echo "  scrfd:   MISSING — the face gate cannot run without it."
  echo "           Place scrfd_2.5g_bnkps.onnx (InsightFace model zoo) at:"
  echo "             $SCRFD"
fi

echo
echo "Verify with: rtus-info"
