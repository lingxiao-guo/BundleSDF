#!/usr/bin/env python3
"""
Align InstantMesh to a real reconstructed (incomplete) mesh with similarity transform.

This is a step-1 helper script:
- input moving mesh: InstantMesh OBJ (or InstantMesh directory containing mesh.obj)
- input fixed mesh: real reconstructed mesh OBJ
- output aligned OBJ + transform + optional metrics json
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_INSTANTMESH = (
    "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/instantmesh"
)
DEFAULT_REAL_MESH = (
    "/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/textured_mesh.obj"
)


def resolve_instantmesh_obj(path_or_dir: str) -> str:
    p = Path(path_or_dir).expanduser().resolve()
    if p.is_dir():
        obj = p / "mesh.obj"
        if not obj.exists():
            raise FileNotFoundError(f"InstantMesh directory given but no mesh.obj found: {obj}")
        return str(obj)
    if p.suffix.lower() != ".obj":
        raise ValueError(f"InstantMesh input must be a directory or OBJ file: {p}")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Align InstantMesh to real reconstructed mesh (R + t + uniform scale)."
    )
    ap.add_argument(
        "--instantmesh",
        type=str,
        default=DEFAULT_INSTANTMESH,
        help="InstantMesh directory (contains mesh.obj) or InstantMesh OBJ path.",
    )
    ap.add_argument(
        "--real_mesh",
        type=str,
        default=DEFAULT_REAL_MESH,
        help="Real reconstructed incomplete mesh path (OBJ).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output aligned OBJ path. Default: <instantmesh_dir>/mesh_aligned.obj",
    )
    ap.add_argument(
        "--transform_txt",
        type=str,
        default="",
        help="Output 4x4 transform txt. Default: <out_dir>/instantmesh_to_real_transform.txt",
    )
    ap.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Optional metrics json. Default: <out_dir>/instantmesh_to_real_metrics.json",
    )
    ap.add_argument("--sample_points", type=int, default=12000)
    ap.add_argument("--max_iter", type=int, default=65)
    ap.add_argument("--trim_quantile", type=float, default=0.90)
    ap.add_argument("--random_rot_seeds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--overwrite_instantmesh",
        type=int,
        default=0,
        help="If 1, write aligned mesh directly back to InstantMesh mesh.obj.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    src_obj = resolve_instantmesh_obj(args.instantmesh)
    real_mesh = str(Path(args.real_mesh).expanduser().resolve())
    if not os.path.exists(real_mesh):
        raise FileNotFoundError(real_mesh)

    instantmesh_dir = os.path.dirname(src_obj)
    out_obj = (
        str(Path(args.out).expanduser().resolve())
        if args.out
        else os.path.join(instantmesh_dir, "mesh_aligned.obj")
    )
    out_dir = os.path.dirname(out_obj)
    os.makedirs(out_dir, exist_ok=True)

    transform_txt = (
        str(Path(args.transform_txt).expanduser().resolve())
        if args.transform_txt
        else os.path.join(out_dir, "instantmesh_to_real_transform.txt")
    )
    metrics_json = (
        str(Path(args.metrics_json).expanduser().resolve())
        if args.metrics_json
        else os.path.join(out_dir, "instantmesh_to_real_metrics.json")
    )

    register_script = Path(__file__).resolve().parent / "tools" / "register_similarity_obj.py"
    if not register_script.exists():
        raise FileNotFoundError(f"Missing registration script: {register_script}")

    cmd = [
        sys.executable,
        str(register_script),
        "--src",
        src_obj,
        "--dst",
        real_mesh,
        "--out",
        out_obj,
        "--transform_txt",
        transform_txt,
        "--metrics_json",
        metrics_json,
        "--sample_points",
        str(args.sample_points),
        "--max_iter",
        str(args.max_iter),
        "--trim_quantile",
        str(args.trim_quantile),
        "--random_rot_seeds",
        str(args.random_rot_seeds),
        "--seed",
        str(args.seed),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if int(args.overwrite_instantmesh) == 1:
        if os.path.abspath(out_obj) != os.path.abspath(src_obj):
            Path(src_obj).write_bytes(Path(out_obj).read_bytes())
        print(f"Overwrote InstantMesh mesh with aligned mesh: {src_obj}")

    print(f"Aligned mesh: {out_obj}")
    print(f"Transform: {transform_txt}")
    print(f"Metrics: {metrics_json}")


if __name__ == "__main__":
    main()

