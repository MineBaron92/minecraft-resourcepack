#!/usr/bin/env bash
# Full pipeline: assembly -> rig/animation/export -> runtime render -> verification.
# Needs: python3 (numpy, pillow, opencv-python-headless), node 18+, and a
# node_modules dir with @esotericsoftware/spine-webgl@4.2.x, spine-player@4.2.x, playwright-core.
#   NODE_DIR=/path/with/node_modules CHROME=/path/to/chrome bash spine_pipeline/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE_DIR="${NODE_DIR:?set NODE_DIR to a folder containing node_modules}"
python3 "$ROOT/spine_pipeline/assemble.py"
python3 "$ROOT/spine_pipeline/rig.py"
(cd "$NODE_DIR" && node "$ROOT/spine_pipeline/verify/render_frames.mjs" \
   "$ROOT/output/spine_character/spine" "$ROOT/output/spine_character/verification/frames" ${CHROME:-})
python3 "$ROOT/spine_pipeline/verify.py" --player-dir "$NODE_DIR/node_modules/@esotericsoftware/spine-player/dist"
