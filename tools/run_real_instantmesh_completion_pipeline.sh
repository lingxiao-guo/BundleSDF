#!/usr/bin/env bash
set -euo pipefail

# 3-stage pipeline:
# 1) Real RGB-D tracking + reconstruction only (BundleSDF).
# 2) Similarity registration (R, t, s) of instantmesh to stage-1 reconstructed mesh.
# 3) Offline co-reconstruction from real + synthetic RGB-D (fixed tracked real poses).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/data/real_0212_trash/object_2}"
STAGE1_OUT="${STAGE1_OUT:-$ROOT_DIR/outputs/real_0212_trash/object_2_realonly}"
PARTIAL_MESH="${PARTIAL_MESH:-$STAGE1_OUT/textured_mesh.obj}"
INSTANT_MESH="${INSTANT_MESH:-$ROOT_DIR/data/real_0212_trash/object_2/instantmesh/mesh.obj}"
REGISTERED_MESH="${REGISTERED_MESH:-$INSTANT_MESH}"
FINAL_OUT="${FINAL_OUT:-$ROOT_DIR/outputs/real_0212_trash/object_2_completion_registered_offline}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"

USE_GUI="${USE_GUI:-0}"
USE_SEGMENTER="${USE_SEGMENTER:-0}"
DEBUG_LEVEL="${DEBUG_LEVEL:-2}"

SYNTHETIC_VIEWS="${SYNTHETIC_VIEWS:-48}"
SYNTHETIC_CANDIDATES="${SYNTHETIC_CANDIDATES:-384}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_REAL_FRAMES="${MAX_REAL_FRAMES:--1}"
TEX_RES="${TEX_RES:-2048}"
FREEZE_REAL_POSES="${FREEZE_REAL_POSES:-1}"
SYNTHETIC_FRAME_WEIGHT="${SYNTHETIC_FRAME_WEIGHT:-0.35}"
SYNTHETIC_RGB_WEIGHT="${SYNTHETIC_RGB_WEIGHT:-0.05}"
SYNTHETIC_ONLY_UNCOVERED_TEXELS="${SYNTHETIC_ONLY_UNCOVERED_TEXELS:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --python_bin PATH
  --video_dir PATH
  --stage1_out PATH
  --partial_mesh PATH
  --instant_mesh PATH
  --registered_mesh PATH
  --final_out PATH
  --run_stage1 {0|1}
  --run_stage2 {0|1}
  --run_stage3 {0|1}
  --use_gui {0|1}
  --use_segmenter {0|1}
  --debug_level INT
  --synthetic_views INT
  --synthetic_candidates INT
  --frame_stride INT
  --max_real_frames INT
  --tex_res INT
  --freeze_real_poses {0|1}
  --synthetic_frame_weight FLOAT
  --synthetic_rgb_weight FLOAT
  --synthetic_only_uncovered_texels {0|1}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --video_dir) VIDEO_DIR="$2"; shift 2 ;;
    --stage1_out) STAGE1_OUT="$2"; shift 2 ;;
    --partial_mesh) PARTIAL_MESH="$2"; shift 2 ;;
    --instant_mesh) INSTANT_MESH="$2"; shift 2 ;;
    --registered_mesh) REGISTERED_MESH="$2"; shift 2 ;;
    --final_out) FINAL_OUT="$2"; shift 2 ;;
    --run_stage1) RUN_STAGE1="$2"; shift 2 ;;
    --run_stage2) RUN_STAGE2="$2"; shift 2 ;;
    --run_stage3) RUN_STAGE3="$2"; shift 2 ;;
    --use_gui) USE_GUI="$2"; shift 2 ;;
    --use_segmenter) USE_SEGMENTER="$2"; shift 2 ;;
    --debug_level) DEBUG_LEVEL="$2"; shift 2 ;;
    --synthetic_views) SYNTHETIC_VIEWS="$2"; shift 2 ;;
    --synthetic_candidates) SYNTHETIC_CANDIDATES="$2"; shift 2 ;;
    --frame_stride) FRAME_STRIDE="$2"; shift 2 ;;
    --max_real_frames) MAX_REAL_FRAMES="$2"; shift 2 ;;
    --tex_res) TEX_RES="$2"; shift 2 ;;
    --freeze_real_poses) FREEZE_REAL_POSES="$2"; shift 2 ;;
    --synthetic_frame_weight) SYNTHETIC_FRAME_WEIGHT="$2"; shift 2 ;;
    --synthetic_rgb_weight) SYNTHETIC_RGB_WEIGHT="$2"; shift 2 ;;
    --synthetic_only_uncovered_texels) SYNTHETIC_ONLY_UNCOVERED_TEXELS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$STAGE1_OUT" "$FINAL_OUT"
REG_TFORM_TXT="$FINAL_OUT/instantmesh_to_partial_transform.txt"
REG_METRICS_JSON="$FINAL_OUT/instantmesh_registration_metrics.json"

echo "[Pipeline] ROOT_DIR: $ROOT_DIR"
echo "[Pipeline] PYTHON_BIN: $PYTHON_BIN"

if [[ "$RUN_STAGE1" == "1" ]]; then
  echo "[Stage 1/3] Real RGB-D tracking + reconstruction (no completion prior)"
  "$PYTHON_BIN" "$ROOT_DIR/run_custom.py" \
    --mode run_video \
    --video_dir "$VIDEO_DIR" \
    --out_folder "$STAGE1_OUT" \
    --use_gui "$USE_GUI" \
    --use_segmenter "$USE_SEGMENTER" \
    --debug_level "$DEBUG_LEVEL" \
    --enable_completion 0
fi

if [[ ! -f "$PARTIAL_MESH" ]]; then
  echo "Expected reconstructed mesh not found: $PARTIAL_MESH"
  echo "Either run stage-1 or pass --partial_mesh to an existing real-only reconstructed mesh."
  exit 1
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  echo "[Stage 2/3] Similarity registration (instantmesh -> real reconstructed mesh)"
  "$PYTHON_BIN" "$ROOT_DIR/tools/register_similarity_obj.py" \
    --src "$INSTANT_MESH" \
    --dst "$PARTIAL_MESH" \
    --out "$REGISTERED_MESH" \
    --transform_txt "$REG_TFORM_TXT" \
    --metrics_json "$REG_METRICS_JSON"
fi

if [[ "$RUN_STAGE3" == "1" ]]; then
  echo "[Stage 3/3] Offline co-reconstruction from real + synthetic RGB-D"
  "$PYTHON_BIN" "$ROOT_DIR/tools/offline_complete_with_registered_prior.py" \
    --stage1_out "$STAGE1_OUT" \
    --prior_mesh "$REGISTERED_MESH" \
    --out_dir "$FINAL_OUT" \
    --synthetic_views "$SYNTHETIC_VIEWS" \
    --synthetic_candidates "$SYNTHETIC_CANDIDATES" \
    --frame_stride "$FRAME_STRIDE" \
    --max_real_frames "$MAX_REAL_FRAMES" \
    --tex_res "$TEX_RES" \
    --freeze_real_poses "$FREEZE_REAL_POSES" \
    --synthetic_frame_weight "$SYNTHETIC_FRAME_WEIGHT" \
    --synthetic_rgb_weight "$SYNTHETIC_RGB_WEIGHT" \
    --synthetic_only_uncovered_texels "$SYNTHETIC_ONLY_UNCOVERED_TEXELS"
fi

echo "[Done] Final output folder: $FINAL_OUT"
echo "[Done] Registered mesh path: $REGISTERED_MESH"

