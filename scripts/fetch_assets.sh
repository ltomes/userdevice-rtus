#!/usr/bin/env bash
# Fetch the third-party model assets this repo does not redistribute.
#
#   bash scripts/fetch_assets.sh [dest]
#
# dest defaults to $RTUS_DATA_ROOT/pretrained, or ./pretrained.
#
# NEITHER IS NEEDED TO RUN THE MODEL. Running RTUS means inference on the
# released weights (ONNX or TensorRT). These two are for *developing* it:
# the teacher is needed to regenerate distillation targets, and the SCRFD
# detector is needed to run the face gate.
#
# Both are third-party weights with their own terms; see docs/PROVENANCE.md.
set -euo pipefail

DEST="${1:-${RTUS_DATA_ROOT:-.}/pretrained}"
mkdir -p -- "$DEST"
echo "Fetching into $DEST"

# Every asset is pinned by repo AND revision, and verified by digest. A mirror
# that changes under us then fails loudly instead of silently substituting a
# different model -- which for the face gate would quietly corrupt the one
# check whose whole value is being trustworthy.
fetch() {
  local dest="$1" repo="$2" file="$3" rev="$4" want="$5"
  if [ -f "$dest" ]; then
    echo "  $(basename -- "$dest"): already present"
    return
  fi
  echo "  $(basename -- "$dest"): downloading from $repo@${rev:0:12}"
  python - "$dest" "$repo" "$file" "$rev" "$want" <<'PY'
import hashlib
import shutil
import sys

from huggingface_hub import hf_hub_download

dest, repo, filename, revision, want = sys.argv[1:6]
path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
if got != want:
    raise SystemExit(
        f"DIGEST MISMATCH for {filename}\n  expected {want}\n  got      {got}\n"
        "Refusing to install it. The pinned revision should be immutable, so "
        "this means the pin is wrong or the source changed."
    )
shutil.copyfile(path, dest)
print(f"    verified -> {dest}")
PY
}

# Teacher: 4xNomosWebPhoto_RealPLKSR, Philip Hofmann (Phhofm), CC-BY-4.0.
# The tools expect this exact filename, so the rename is part of the contract.
fetch "$DEST/teacher_4xNomosWebPhoto_RealPLKSR.safetensors" \
      "Phips/4xNomosWebPhoto_RealPLKSR" \
      "4xNomosWebPhoto_RealPLKSR.safetensors" \
      "49d5da19489e645e870eb076ea84815471f27ef4" \
      "9be0228f98156a100d6636d99b373ed2785b999723f9adc4cca504329ab157f2"

# Face-gate detector: SCRFD 2.5g (bnkps), InsightFace.
# InsightFace publishes no stable direct URL for the ONNX export, so this
# pulls from a pinned mirror revision and verifies the digest measured
# 2026-09-05 (3,290,207 bytes).
fetch "$DEST/scrfd_2.5g_bnkps.onnx" \
      "OwlMaster/AllFilesRope" \
      "scrfd_2.5g_bnkps.onnx" \
      "d783e61585b3d83a85c91ca8a3b299e8ade94d72" \
      "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce"

echo
echo "Verify with: rtus-info"
