#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
  cat <<'EOF'
Usage:
  bash run_texture_syn_global_refine.sh <TASK_NAME> <OBJ> [MODE] [NUM_RENDERS] [PROFILE] [FREEZE_POSE_OPT] [SYN_EDGE_SHRINK_PX]

Args:
  TASK_NAME    Dataset/task folder name under data/ and outputs/ (e.g. real_0212_trash)
  OBJ          Object folder name (e.g. object_1)
  MODE         main_color | project (default: main_color)
  NUM_RENDERS  Number of synthetic views (default: 64)
  PROFILE      protect_seen | completion_boost (default: protect_seen)
  FREEZE_POSE_OPT   1 to freeze pose updates safely (optimize_poses:1 + lrate_pose:0) in outputs/<task>/<obj>/config_nerf.yml (default: 1)
  SYN_EDGE_SHRINK_PX  Synthetic depth support shrink in pixels (default: 2)

Examples:
  bash run_texture_syn_global_refine.sh real_0212_trash object_1
  bash run_texture_syn_global_refine.sh real_0212_trash object_1 project 64 completion_boost 1 1
EOF
  exit 1
fi
 
TASK_NAME="$1"
OBJ="$2"
MODE="${3:-main_color}"
NUM_RENDERS="${4:-64}"
PROFILE="${5:-protect_seen}"
FREEZE_POSE_OPT="${6:-1}"
SYN_EDGE_SHRINK_PX="${7:-2}"

if [[ "$MODE" != "main_color" && "$MODE" != "project" ]]; then
  echo "Error: MODE must be 'main_color' or 'project', got: $MODE" >&2
  exit 1
fi
if [[ "$PROFILE" != "protect_seen" && "$PROFILE" != "completion_boost" ]]; then
  echo "Error: PROFILE must be 'protect_seen' or 'completion_boost', got: $PROFILE" >&2
  exit 1
fi
if [[ "$FREEZE_POSE_OPT" != "0" && "$FREEZE_POSE_OPT" != "1" ]]; then
  echo "Error: FREEZE_POSE_OPT must be 0 or 1, got: $FREEZE_POSE_OPT" >&2
  exit 1
fi
if ! [[ "$NUM_RENDERS" =~ ^[0-9]+$ ]] || [[ "$NUM_RENDERS" -le 0 ]]; then
  echo "Error: NUM_RENDERS must be a positive integer, got: $NUM_RENDERS" >&2
  exit 1
fi
if ! [[ "$SYN_EDGE_SHRINK_PX" =~ ^[0-9]+$ ]]; then
  echo "Error: SYN_EDGE_SHRINK_PX must be a non-negative integer, got: $SYN_EDGE_SHRINK_PX" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_OBJ="$ROOT/data/$TASK_NAME/$OBJ"
OUT_OBJ="$ROOT/outputs/$TASK_NAME/$OBJ"
SAM3D_DIR="$DATA_OBJ/sam3d"

TARGET_MESH="$SAM3D_DIR/mesh_affine_aligned_partialaware.obj"
REAL_MESH="$OUT_OBJ/textured_mesh.obj"
UV_REF_MESH="$SAM3D_DIR/mesh_aligned.obj"
CAM_K="$DATA_OBJ/cam_K.txt"

OUT_PROJECT_OBJ="$SAM3D_DIR/mesh_affine_aligned_partialaware_projected_color_textured_v2.obj"
OUT_MAIN_OBJ="$SAM3D_DIR/mesh_affine_aligned_partialaware_main_color_textured_v2.obj"

if [[ ! -f "$TARGET_MESH" ]]; then
  echo "Error: missing target mesh: $TARGET_MESH" >&2
  exit 1
fi
if [[ ! -f "$REAL_MESH" ]]; then
  echo "Error: missing real mesh: $REAL_MESH" >&2
  exit 1
fi
if [[ ! -f "$CAM_K" ]]; then
  echo "Error: missing camera intrinsics: $CAM_K" >&2
  exit 1
fi

REAL_DEPTH_DIR="$OUT_OBJ/depth_filtered"
if [[ ! -d "$REAL_DEPTH_DIR" ]]; then
  REAL_DEPTH_DIR="$DATA_OBJ/depth"
fi
REAL_MASK_DIR="$OUT_OBJ/mask"
REAL_POSE_DIR="$OUT_OBJ/ob_in_cam"

if [[ ! -d "$REAL_DEPTH_DIR" ]]; then
  echo "Error: missing real depth dir: $REAL_DEPTH_DIR" >&2
  exit 1
fi
if [[ ! -d "$REAL_MASK_DIR" ]]; then
  echo "Error: missing real mask dir: $REAL_MASK_DIR" >&2
  exit 1
fi
if [[ ! -d "$REAL_POSE_DIR" ]]; then
  echo "Error: missing real pose dir: $REAL_POSE_DIR" >&2
  exit 1
fi

if [[ "$MODE" == "main_color" ]]; then
  MESH_FOR_RENDER="$OUT_MAIN_OBJ"
else
  MESH_FOR_RENDER="$OUT_PROJECT_OBJ"
fi

if [[ "$PROFILE" == "protect_seen" ]]; then
  RENDER_SEEN_PENALTY="3.0"
  RENDER_MIN_UNSEEN_RATIO="0.78"
  RENDER_SEEN_THRESHOLD="0.48"
  RENDER_BOUNDARY_RATIO="0.08"
  RENDER_MIN_VIEW_ANGLE_DEG="8.0"
  RENDER_DISTANCE_SCALE="1.15"
  RENDER_FRAME_STRIDE="6"
  RENDER_MAX_REAL_FRAMES="180"
  RENDER_ESTIMATION_DOWNSCALE="2"
else
  RENDER_SEEN_PENALTY="2.0"
  RENDER_MIN_UNSEEN_RATIO="0.60"
  RENDER_SEEN_THRESHOLD="0.55"
  RENDER_BOUNDARY_RATIO="0.12"
  RENDER_MIN_VIEW_ANGLE_DEG="6.0"
  RENDER_DISTANCE_SCALE="1.10"
  RENDER_FRAME_STRIDE="6"
  RENDER_MAX_REAL_FRAMES="180"
  RENDER_ESTIMATION_DOWNSCALE="2"
fi

SYN_BASE="$SAM3D_DIR/syn_${MODE}_${PROFILE}_v2"
SYN_RGB_DIR="$SYN_BASE/rgb"
SYN_DEPTH_DIR="$SYN_BASE/depth"
SYN_POSE_DIR="$SYN_BASE/ob_in_cam"
SYN_PREFIX="syn_affv2_${MODE}_${PROFILE}_"

mkdir -p "$SYN_RGB_DIR" "$SYN_DEPTH_DIR" "$SYN_POSE_DIR"

echo "[config] TASK_NAME=$TASK_NAME OBJ=$OBJ MODE=$MODE NUM_RENDERS=$NUM_RENDERS PROFILE=$PROFILE FREEZE_POSE_OPT=$FREEZE_POSE_OPT SYN_EDGE_SHRINK_PX=$SYN_EDGE_SHRINK_PX"

echo "[1/5] texture transfer (project + main_color) ..."
python "$ROOT/texture_transfer_simple.py" \
  --target_mesh "$TARGET_MESH" \
  --real_mesh "$REAL_MESH" \
  --mode both \
  --uv_ref_mesh "$UV_REF_MESH" \
  --out_project_obj "$OUT_PROJECT_OBJ" \
  --out_main_obj "$OUT_MAIN_OBJ"

if [[ ! -f "$MESH_FOR_RENDER" ]]; then
  echo "Error: textured mesh not generated: $MESH_FOR_RENDER" >&2
  exit 1
fi

echo "[2/5] render synthetic RGBD using mode=$MODE profile=$PROFILE ..."
rm -f "$SYN_RGB_DIR"/*.png "$SYN_DEPTH_DIR"/*.png "$SYN_POSE_DIR"/*.txt
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
python "$ROOT/render_synthetic_rgbd.py" \
  --mesh "$MESH_FOR_RENDER" \
  --cam_k "$CAM_K" \
  --rgb_dir "$SYN_RGB_DIR" \
  --depth_dir "$SYN_DEPTH_DIR" \
  --pose_dir "$SYN_POSE_DIR" \
  --real_depth_dir "$REAL_DEPTH_DIR" \
  --real_mask_dir "$REAL_MASK_DIR" \
  --real_pose_dir "$REAL_POSE_DIR" \
  --coverage_report "$SYN_BASE/coverage_report.json" \
  --num_renders "$NUM_RENDERS" \
  --seen_penalty "$RENDER_SEEN_PENALTY" \
  --min_unseen_ratio "$RENDER_MIN_UNSEEN_RATIO" \
  --seen_threshold "$RENDER_SEEN_THRESHOLD" \
  --boundary_ratio "$RENDER_BOUNDARY_RATIO" \
  --min_view_angle_deg "$RENDER_MIN_VIEW_ANGLE_DEG" \
  --distance_scale "$RENDER_DISTANCE_SCALE" \
  --frame_stride "$RENDER_FRAME_STRIDE" \
  --max_real_frames "$RENDER_MAX_REAL_FRAMES" \
  --estimation_downscale "$RENDER_ESTIMATION_DOWNSCALE"

echo "[3/5] shrink synthetic depth support near silhouettes (px=$SYN_EDGE_SHRINK_PX) ..."
if [[ "$SYN_EDGE_SHRINK_PX" -gt 0 ]]; then
  python - "$SYN_DEPTH_DIR" "$SYN_EDGE_SHRINK_PX" <<'PY'
import glob
import os
import sys

import cv2
import numpy as np

depth_dir = sys.argv[1]
shrink_px = int(sys.argv[2])
kernel_size = int(2 * shrink_px + 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
if not files:
    raise RuntimeError(f"No depth png found in {depth_dir}")

changed = 0
removed_px = 0
for p in files:
    depth = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth: {p}")
    if depth.dtype != np.uint16:
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
    mask = (depth > 0).astype(np.uint8)
    if mask.max() == 0:
        cv2.imwrite(p, depth)
        continue
    eroded = cv2.erode(mask, kernel, iterations=1)
    removed_px += int(((mask > 0) & (eroded == 0)).sum())
    depth[eroded == 0] = 0
    cv2.imwrite(p, depth)
    changed += 1

print(
    f"[edge_shrink] files={len(files)} changed={changed} shrink_px={shrink_px} removed_pixels={removed_px}"
)
PY
else
  echo "[edge_shrink] skip (SYN_EDGE_SHRINK_PX=0)"
fi

echo "[4/5] optional freeze pose optimization in global refine config ..."
if [[ "$FREEZE_POSE_OPT" == "1" ]]; then
  CFG_NERF="$OUT_OBJ/config_nerf.yml"
  if [[ ! -f "$CFG_NERF" ]]; then
    echo "Error: $CFG_NERF not found. Run tracking first to generate config_nerf.yml." >&2
    exit 1
  fi
  python - "$CFG_NERF" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
if cfg is None:
    cfg = {}
# Keep pose model instantiated (required by current global refine post-processing),
# but freeze updates by using zero pose learning rate.
cfg["optimize_poses"] = 1
cfg["lrate_pose"] = 0.0
with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(
    f"[freeze_pose_opt] set optimize_poses=1 and lrate_pose=0.0 in {cfg_path}"
)
PY
else
  echo "[freeze_pose_opt] keep original optimize_poses setting"
fi

echo "[5/5] global refine with synthetic RGBD ..."
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
python "$ROOT/run_custom.py" \
  --mode global_refine \
  --video_dir "$DATA_OBJ" \
  --out_folder "$OUT_OBJ" \
  --use_syn_rgbd 1 \
  --syn_rgb_dir "$SYN_RGB_DIR" \
  --syn_depth_dir "$SYN_DEPTH_DIR" \
  --syn_pose_dir "$SYN_POSE_DIR" \
  --syn_prefix "$SYN_PREFIX" \
  --interpolate_missing_vertices 1

echo "Done."
echo "Task: $TASK_NAME  Object: $OBJ  Mode: $MODE  NumRenders: $NUM_RENDERS  Profile: $PROFILE"
echo "Synthetic RGBD: $SYN_BASE"
echo "Refined mesh: $OUT_OBJ/textured_mesh.obj"
