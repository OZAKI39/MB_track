#!/bin/bash
set -euo pipefail
cd /workspace/microbubble_sam21_initial_20261026-170243
.venv/bin/python -u sam21_initial_tracking.py run inputs video.mp4 models/sam2.1_hiera_large.pt predictions --memory-keep 32 2>&1 | tee full.log
