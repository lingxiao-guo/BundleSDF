#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  cat <<'EOF'
Usage:
  bash run_oneclick_reconstruct_align_fuse.sh <TASK_NAME> <OBJECT_NAME>

Args:
  TASK_NAME     Dataset/task folder name under data/ and outputs/ (e.g. real_0212_trash)
  OBJECT_NAME   Object folder name (e.g. object_1)

Notes:
  - This script requires data/<TASK_NAME>/<OBJECT_NAME>/sam3d/mesh.glb.
  - Affine axis default is z via AFFINE_AXIS env var.
    You can adjust it to control scale dimensions for SAM3D affine refinement, e.g.:
      AFFINE_AXIS=xyz bash run_oneclick_reconstruct_align_fuse.sh real_0212_trash object_1
EOF
  exit 1
fi

TASK_NAME="$1"
OBJ="$2"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_OBJ="$ROOT/data/$TASK_NAME/$OBJ"
OUT_OBJ="$ROOT/outputs/$TASK_NAME/$OBJ"
SAM3D_DIR="$DATA_OBJ/sam3d"

SAM3D_MESH_GLB="$SAM3D_DIR/mesh.glb"
REAL_MESH_OBJ="$OUT_OBJ/textured_mesh.obj"
ALIGNED_MESH_OBJ="$SAM3D_DIR/mesh_aligned.obj"
AFFINE_MESH_OBJ="$SAM3D_DIR/mesh_affine_aligned_partialaware.obj"

# AFFINE_AXIS controls how axis-scale refinement is constrained:
# z -> only z is independent; x/z or y/z etc can be configured in align_affine script semantics.
AFFINE_AXIS="${AFFINE_AXIS:-z}"

if [[ ! -d "$DATA_OBJ" ]]; then
  echo "Error: missing data directory: $DATA_OBJ" >&2
  exit 1
fi
if [[ ! -f "$SAM3D_MESH_GLB" ]]; then
  echo "Error: required SAM3D mesh not found: $SAM3D_MESH_GLB" >&2
  echo "This pipeline requires mesh.glb under sam3d/ by default." >&2
  exit 1
fi

mkdir -p "$OUT_OBJ"

echo "[1/3] Reconstruct from real RGBD only ..."
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
python "$ROOT/run_custom.py" \
  --mode run_video \
  --video_dir "$DATA_OBJ" \
  --out_folder "$OUT_OBJ" \
  --use_segmenter 0 \
  --use_gui 0 \
  --debug_level 2 \
  --mesh_out_name textured_mesh.obj \
  --interpolate_missing_vertices 1

if [[ ! -f "$REAL_MESH_OBJ" ]]; then
  echo "Error: real reconstruction failed (missing $REAL_MESH_OBJ)" >&2
  exit 1
fi

echo "[2/3] First registration + affine refinement ..."
python "$ROOT/align_instantmesh_and_real.py" \
  --source_mesh "$SAM3D_MESH_GLB" \
  --real_mesh "$REAL_MESH_OBJ" \
  --out "$ALIGNED_MESH_OBJ"

if [[ ! -f "$ALIGNED_MESH_OBJ" ]]; then
  echo "Error: registration failed (missing $ALIGNED_MESH_OBJ)" >&2
  exit 1
fi

python "$ROOT/align_affine_axes_partialaware.py" \
  --source_mesh_aligned "$ALIGNED_MESH_OBJ" \
  --real_mesh "$REAL_MESH_OBJ" \
  --affine_axis "$AFFINE_AXIS"

if [[ ! -f "$AFFINE_MESH_OBJ" ]]; then
  echo "Error: affine refinement failed (missing $AFFINE_MESH_OBJ)" >&2
  exit 1
fi

echo "[3/3] Fuse SAM3D with real mesh completion pipeline ..."
bash "$ROOT/run_texture_syn_global_refine.sh" "$TASK_NAME" "$OBJ" main_color 64 protect_seen 1 2 textured_mesh_fuse.obj

echo "Done."
echo "Real-only mesh kept: $OUT_OBJ/textured_mesh.obj"
echo "Fused mesh exported: $OUT_OBJ/textured_mesh_fuse.obj"
