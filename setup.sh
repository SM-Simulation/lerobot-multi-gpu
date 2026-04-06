#!/bin/bash
# Apply fast multi-GPU training patches to lerobot 0.4.4
#
# Usage:
#   bash setup.sh              # Apply patches
#   bash setup.sh --revert     # Remove patches (restore originals)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
LEROBOT_DIR="$SITE_PACKAGES/lerobot"

if [ ! -d "$LEROBOT_DIR" ]; then
    echo "ERROR: lerobot not found at $LEROBOT_DIR"
    echo "Install it first: pip install lerobot==0.4.4"
    exit 1
fi

# Check version
VERSION=$(python3 -c "from importlib.metadata import version; print(version('lerobot'))" 2>/dev/null || echo "unknown")
if [ "$VERSION" != "0.4.4" ]; then
    echo "WARNING: Expected lerobot 0.4.4, found $VERSION. Patches may not apply cleanly."
fi

if [ "${1:-}" = "--revert" ]; then
    echo "Reverting patches..."
    for f in "$LEROBOT_DIR/datasets/lerobot_dataset.py" \
             "$LEROBOT_DIR/scripts/lerobot_train.py" \
             "$LEROBOT_DIR/policies/act/modeling_act.py" \
             "$LEROBOT_DIR/policies/act/configuration_act.py"; do
        if [ -f "${f}.orig" ]; then
            cp "${f}.orig" "$f"
            rm "${f}.orig"
            echo "  Restored: $f"
        fi
    done
    echo "Done. Original lerobot restored."
    exit 0
fi

echo "Applying fast multi-GPU training patches to lerobot $VERSION..."
echo "  Site packages: $SITE_PACKAGES"

# Backup originals
for f in "$LEROBOT_DIR/datasets/lerobot_dataset.py" \
         "$LEROBOT_DIR/scripts/lerobot_train.py" \
         "$LEROBOT_DIR/policies/act/modeling_act.py" \
         "$LEROBOT_DIR/policies/act/configuration_act.py"; do
    if [ ! -f "${f}.orig" ]; then
        cp "$f" "${f}.orig"
    fi
done

# Apply patches
cd "$SITE_PACKAGES"
for patch in "$SCRIPT_DIR/patches/"*.patch; do
    echo "  Applying: $(basename $patch)"
    patch -p0 --forward < "$patch" || echo "    (already applied or conflict)"
done

echo ""
echo "Patches applied successfully!"
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset:"
echo "     python $SCRIPT_DIR/prepare_dataset.py --dataset-root /path/to/dataset --output-dir /dev/shm"
echo ""
echo "  2. Train:"
echo "     bash $SCRIPT_DIR/train.sh --dataset-root /dev/shm/sim_pick_vase"
