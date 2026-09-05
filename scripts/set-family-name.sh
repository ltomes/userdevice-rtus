#!/usr/bin/env bash
# Stamp the model family name across the repo. Run once, when it is chosen.
#
#   bash scripts/set-family-name.sh <name>
#
# The family is the middle slot of  userdevice-<family>-<version>-<shape>.
# Until this runs, the repo says "family name pending" everywhere rather
# than carrying a guess. See docs/NAMING.md.
set -euo pipefail
NAME="${1:?usage: set-family-name.sh <name>}"
LOWER=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')

grep -rl "FAMILY_NAME_PENDING" . --exclude-dir=.git | while read -r f; do
  sed -i "s/FAMILY_NAME_PENDING/${LOWER}/g" "$f"
  echo "  stamped $f"
done
echo
echo "Family set to '${LOWER}'. Artifact stem is now userdevice-${LOWER}-<version>-<shape>."
echo "Remaining manual steps: rename the repo directory and set the git remote."
