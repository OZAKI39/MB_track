set -euo pipefail
cd /workspace/microbubble_joint_sam21_20261026-190838
export PYTHONUNBUFFERED=1
/workspace/microbubble_joint_sam21_20261026-190838/.venv/bin/python -u -m customer_workspace.joint_sam21_runner /workspace/microbubble_joint_sam21_20261026-190838/inputs /workspace/microbubble_joint_sam21_20261026-190838/video.mp4 /workspace/microbubble_sam21_initial_20261026-170243/models/sam2.1_hiera_large.pt /workspace/microbubble_joint_sam21_20261026-190838/pilot_full --limit 64 --memory-keep 0
