#!/usr/bin/env bash
# Character 2 pipeline: PSD import -> rig/animation/export -> runtime render -> verification.
#   NODE_DIR=/path/with/node_modules CHROME=/path/to/chrome bash spine_pipeline/run_char2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE_DIR="${NODE_DIR:?set NODE_DIR to a folder containing node_modules}"
export SPINE_OUT="$ROOT/output/char2" SPINE_REF="$ROOT/assets/char2/reference/original.png"
python3 "$ROOT/spine_pipeline/psd_import.py" "$ROOT/assets/char2/seethrough.psd" \
  --depth "$ROOT/assets/char2/seethrough_depth.psd" --ref "$SPINE_REF"
python3 "$ROOT/spine_pipeline/rig_char2.py"
(cd "$NODE_DIR" && node "$ROOT/spine_pipeline/verify/render_frames.mjs" \
   "$SPINE_OUT/spine" "$SPINE_OUT/verification/frames" ${CHROME:-})
python3 "$ROOT/spine_pipeline/verify.py" --player-dir "$NODE_DIR/node_modules/@esotericsoftware/spine-player/dist"
