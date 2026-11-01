#!/bin/bash
set -euo pipefail
cd /workspace/microbubble_cutie_initial_20261026-172733
exec .venv/bin/python -u -m customer_workspace.cutie_initial_tracking inputs video.mp4 models/cutie-base-mega.pth pilot --limit 65
