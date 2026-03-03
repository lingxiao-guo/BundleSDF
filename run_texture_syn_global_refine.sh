#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  cat <<'EOF'
Usage:
  bash run_texture_syn_global_refine.sh <TASK_NAME> <OBJ> [MODE] [NUM_RENDERS]

Args:
  TASK_NAME    Dataset/task folder name under data/ and outputs/ (e.g. real_0212_trash)
  OBJ          Object folder name (e.g. object_1)
  MODE         main_color | project (default: main_color)
  NUM_RENDERS  Number of synthetic views (default: 64)
EOF
  exit 1
fi

TASK_NAME="$1"
OBJ="$2"
MODE="${3:-main_color}"
NUM_RENDERS="${4:-64}"

if [[ "$MODE" != "main_color" && "$MODE" != "project" ]]; then
  echo "Error: MODE must be 'main_color' or 'project', got: $MODE" >&2
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

SYN_BASE="$SAM3D_DIR/syn_${MODE}_v2"
SYN_RGB_DIR="$SYN_BASE/rgb"
SYN_DEPTH_DIR="$SYN_BASE/depth"
SYN_POSE_DIR="$SYN_BASE/ob_in_cam"
SYN_PREFIX="syn_affv2_${MODE}_"

mkdir -p "$SYN_RGB_DIR" "$SYN_DEPTH_DIR" "$SYN_POSE_DIR"

echo "[1/3] texture transfer (project + main_color) ..."
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

echo "[2/3] render synthetic RGBD using mode=$MODE ..."
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
  --num_renders "$NUM_RENDERS"

echo "[3/3] global refine with synthetic RGBD ..."
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
echo "Task: $TASK_NAME  Object: $OBJ  Mode: $MODE  NumRenders: $NUM_RENDERS"
echo "Synthetic RGBD: $SYN_BASE"
echo "Refined mesh: $OUT_OBJ/textured_mesh.obj"
