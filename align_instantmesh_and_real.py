#!/usr/bin/env python3
"""
Align a completion mesh (e.g. SAM3D/InstantMesh) to a real reconstructed
(incomplete) mesh with similarity transform.

This is a step-1 helper script:
- input moving mesh: completion mesh file (supports common mesh formats) or directory
- input fixed mesh: real reconstructed mesh OBJ
- output aligned OBJ + transform + optional metrics json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_SOURCE_MESH = (
    "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/sam3d"
)
DEFAULT_REAL_MESH = (
    "/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/textured_mesh.obj"
)


def resolve_source_mesh(path_or_dir: str) -> str:
    p = Path(path_or_dir).expanduser().resolve()
    if p.is_dir():
        candidates = [
            p / "mesh.glb",
            p / "mesh.obj",
            p / "mesh.ply",
            p / "mesh.stl",
            p / "mesh.off",
        ]
        for mesh_path in candidates:
            if mesh_path.exists():
                return str(mesh_path)
        raise FileNotFoundError(
            f"Directory given but no mesh found. Tried: {[str(x) for x in candidates]}"
        )
    if not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Align completion mesh to real reconstructed mesh (R + t + uniform scale)."
    )
    ap.add_argument(
        "--instantmesh",
        "--source_mesh",
        dest="source_mesh",
        type=str,
        default=DEFAULT_SOURCE_MESH,
        help="Completion mesh directory or mesh file path. For SAM3D directory, mesh.glb is auto-picked.",
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
        help="Output aligned mesh path. Default: <source_mesh_dir>/mesh_aligned.obj",
    )
    ap.add_argument(
        "--transform_txt",
        type=str,
        default="",
        help="Output 4x4 transform txt. Default: <out_dir>/sam3d_to_real_transform.txt",
    )
    ap.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Optional metrics json. Default: <out_dir>/sam3d_to_real_metrics.json",
    )
    ap.add_argument(
        "--axis_json",
        type=str,
        default="",
        help=(
            "Output source-axis json transformed to real frame. "
            "Default: <out_dir>/sam3d_aligned_axes.json"
        ),
    )
    ap.add_argument("--sample_points", type=int, default=12000)
    ap.add_argument("--max_iter", type=int, default=65)
    ap.add_argument("--trim_quantile", type=float, default=0.90)
    ap.add_argument("--random_rot_seeds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--overwrite_instantmesh",
        "--overwrite_source_mesh",
        dest="overwrite_source_mesh",
        type=int,
        default=0,
        help="If 1, write aligned mesh directly back to the source mesh path.",
    )
    return ap.parse_args()


def _load_single_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return mesh


def _write_axis_json(src_mesh_path: str, transform_txt: str, axis_json_path: str):
    src_mesh = _load_single_mesh(src_mesh_path)
    t = np.loadtxt(transform_txt, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform at {transform_txt}, got {t.shape}")

    lin = t[:3, :3]
    trans = t[:3, 3]

    center_local = np.asarray(src_mesh.vertices, dtype=np.float64).mean(axis=0)
    center_world = lin @ center_local + trans

    # Source mesh canonical axes are local XYZ. Transform and normalize into real frame.
    axis_dirs_world = []
    for i in range(3):
        d = lin @ np.eye(3, dtype=np.float64)[:, i]
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            raise RuntimeError("Degenerate axis under similarity transform.")
        axis_dirs_world.append((d / n).tolist())

    verts = np.asarray(src_mesh.vertices, dtype=np.float64)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extents_local = (maxs - mins).tolist()
    long_axis_index = int(np.argmax(maxs - mins))

    axis_data = {
        "source_mesh": str(Path(src_mesh_path).resolve()),
        "transform_txt": str(Path(transform_txt).resolve()),
        "axis_definition": "transformed_source_xyz",
        "center_local": center_local.tolist(),
        "center_world": center_world.tolist(),
        "axis_dirs_world": axis_dirs_world,  # [x_dir, y_dir, z_dir]
        "axis_extents_local": extents_local,
        "long_axis_index": long_axis_index,
    }
    os.makedirs(os.path.dirname(os.path.abspath(axis_json_path)), exist_ok=True)
    with open(axis_json_path, "w") as f:
        json.dump(axis_data, f, indent=2)


def main():
    args = parse_args()

    src_mesh = resolve_source_mesh(args.source_mesh)
    real_mesh = str(Path(args.real_mesh).expanduser().resolve())
    if not os.path.exists(real_mesh):
        raise FileNotFoundError(real_mesh)

    source_mesh_dir = os.path.dirname(src_mesh)
    out_obj = (
        str(Path(args.out).expanduser().resolve())
        if args.out
        else os.path.join(source_mesh_dir, "mesh_aligned.obj")
    )
    out_dir = os.path.dirname(out_obj)
    os.makedirs(out_dir, exist_ok=True)

    transform_txt = (
        str(Path(args.transform_txt).expanduser().resolve())
        if args.transform_txt
        else os.path.join(out_dir, "sam3d_to_real_transform.txt")
    )
    metrics_json = (
        str(Path(args.metrics_json).expanduser().resolve())
        if args.metrics_json
        else os.path.join(out_dir, "sam3d_to_real_metrics.json")
    )
    axis_json = (
        str(Path(args.axis_json).expanduser().resolve())
        if args.axis_json
        else os.path.join(out_dir, "sam3d_aligned_axes.json")
    )

    register_script = Path(__file__).resolve().parent / "tools" / "register_similarity_obj.py"
    if not register_script.exists():
        raise FileNotFoundError(f"Missing registration script: {register_script}")

    cmd = [
        sys.executable,
        str(register_script),
        "--src",
        src_mesh,
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
    _write_axis_json(src_mesh_path=src_mesh, transform_txt=transform_txt, axis_json_path=axis_json)

    if int(args.overwrite_source_mesh) == 1:
        if os.path.abspath(out_obj) != os.path.abspath(src_mesh):
            src_ext = Path(src_mesh).suffix.lower()
            out_ext = Path(out_obj).suffix.lower()
            if src_ext == out_ext:
                Path(src_mesh).write_bytes(Path(out_obj).read_bytes())
            else:
                # Re-export with source extension (e.g. overwrite mesh.glb from aligned .obj).
                import trimesh

                trimesh.load(out_obj, force="mesh").export(src_mesh)
        print(f"Overwrote source mesh with aligned mesh: {src_mesh}")

    print(f"Aligned mesh: {out_obj}")
    print(f"Transform: {transform_txt}")
    print(f"Axis JSON: {axis_json}")
    print(f"Metrics: {metrics_json}")


if __name__ == "__main__":
    main()
