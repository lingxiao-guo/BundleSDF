#!/usr/bin/env python3
"""
Similarity registration (R, t, uniform s) for mesh-to-mesh OBJ alignment.

Notes:
- Uses sampled surface points + multistart rigid ICP.
- Uniform scale is estimated once from source/target spread to avoid scale-collapse.
- Writes aligned mesh and transform matrix.
"""

import argparse
import itertools
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)


def pca_basis(points: np.ndarray) -> np.ndarray:
    x = points - points.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(points) - 1, 1)
    w, v = np.linalg.eigh(cov)
    return v[:, np.argsort(w)[::-1]]


def candidate_axis_rotations(src_pts: np.ndarray, dst_pts: np.ndarray):
    bs = pca_basis(src_pts)
    bd = pca_basis(dst_pts)
    out = []
    for perm in itertools.permutations([0, 1, 2]):
        pm = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            sm = np.diag(signs)
            r = bd @ pm @ sm @ bs.T
            if np.linalg.det(r) > 0.0:
                out.append(r)
    uniq = []
    for r in out:
        if not any(np.allclose(r, q, atol=1e-6) for q in uniq):
            uniq.append(r)
    return uniq


def random_rotations(n: int, rng: np.random.Generator):
    rots = []
    for _ in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q) + 1e-12
        w, x, y, z = q
        r = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )
        if np.linalg.det(r) > 0.0:
            rots.append(r)
    return rots


def kabsch(a: np.ndarray, b: np.ndarray):
    # b ~= R a + t
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    aa = a - ca
    bb = b - cb
    h = aa.T @ bb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cb - (r @ ca)
    return r, t


def transform_pts(src: np.ndarray, s: float, r: np.ndarray, t: np.ndarray):
    return s * (src @ r.T) + t[None, :]


def rigid_icp_fixed_scale(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    scale: float,
    r0: np.ndarray,
    t0: np.ndarray,
    max_iter: int,
    trim_quantile: float,
):
    tree = cKDTree(dst_pts)
    r = r0.copy()
    t = t0.copy()
    prev = np.inf
    eps = 1e-12

    for _ in range(max_iter):
        cur = transform_pts(src_pts, scale, r, t)
        d, idx = tree.query(cur, k=1)
        cutoff = np.quantile(d, trim_quantile)
        keep = d <= cutoff
        if int(keep.sum()) < 16:
            break
        a = cur[keep]
        b = dst_pts[idx[keep]]
        r_inc, t_inc = kabsch(a, b)
        r = r_inc @ r
        t = r_inc @ t + t_inc

        cur2 = transform_pts(src_pts, scale, r, t)
        d2, _ = tree.query(cur2, k=1)
        d2_keep = d2[d2 <= np.quantile(d2, trim_quantile)]
        cost = float(np.mean(d2_keep**2)) if len(d2_keep) > 0 else np.inf
        if abs(prev - cost) < eps:
            prev = cost
            break
        prev = cost
    return r, t, float(prev)


def make_transform(scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = scale * rot
    t[:3, 3] = trans
    return t


def parse_args():
    ap = argparse.ArgumentParser(description="Register moving mesh to fixed mesh with similarity transform.")
    ap.add_argument("--src", required=True, help="Moving mesh path (e.g., instantmesh obj).")
    ap.add_argument("--dst", required=True, help="Fixed target mesh path (e.g., BundleSDF mesh).")
    ap.add_argument("--out", required=True, help="Output aligned mesh path.")
    ap.add_argument("--transform_txt", required=True, help="Output 4x4 transform txt.")
    ap.add_argument("--metrics_json", default="", help="Optional output metrics json.")
    ap.add_argument("--sample_points", type=int, default=12000)
    ap.add_argument("--max_iter", type=int, default=65)
    ap.add_argument("--trim_quantile", type=float, default=0.90)
    ap.add_argument("--random_rot_seeds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    src_mesh = load_mesh(args.src)
    dst_mesh = load_mesh(args.dst)
    src_pts = sample_surface(src_mesh, int(args.sample_points))
    dst_pts = sample_surface(dst_mesh, int(args.sample_points))

    c_src = src_pts.mean(axis=0)
    c_dst = dst_pts.mean(axis=0)
    s_src = np.sqrt(np.mean(np.sum((src_pts - c_src) ** 2, axis=1)))
    s_dst = np.sqrt(np.mean(np.sum((dst_pts - c_dst) ** 2, axis=1)))
    scale = float(s_dst / max(s_src, 1e-12))

    rotations = candidate_axis_rotations(src_pts, dst_pts) + random_rotations(
        int(args.random_rot_seeds), rng
    )

    best_cost = np.inf
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    for i, r0 in enumerate(rotations):
        t0 = c_dst - scale * (r0 @ c_src)
        r, t, cost = rigid_icp_fixed_scale(
            src_pts=src_pts,
            dst_pts=dst_pts,
            scale=scale,
            r0=r0,
            t0=t0,
            max_iter=int(args.max_iter),
            trim_quantile=float(args.trim_quantile),
        )
        if cost < best_cost:
            best_cost = cost
            best_r = r
            best_t = t
        if (i + 1) % 8 == 0 or (i + 1) == len(rotations):
            print(f"seed {i + 1:03d}/{len(rotations)} current_best_cost={best_cost:.10f}")

    transform = make_transform(scale, best_r, best_t)

    aligned = src_mesh.copy()
    aligned.apply_transform(transform)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    aligned.export(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.transform_txt)), exist_ok=True)
    np.savetxt(args.transform_txt, transform, fmt="%.10f")

    tree = cKDTree(np.asarray(dst_mesh.vertices, dtype=np.float64))
    d, _ = tree.query(np.asarray(aligned.vertices, dtype=np.float64), k=1)
    metrics = {
        "src": args.src,
        "dst": args.dst,
        "out": args.out,
        "transform_txt": args.transform_txt,
        "uniform_scale": float(scale),
        "trimmed_icp_cost": float(best_cost),
        "nn_mean": float(d.mean()),
        "nn_median": float(np.median(d)),
        "nn_p95": float(np.percentile(d, 95)),
        "nn_max": float(d.max()),
        "rotation": best_r.tolist(),
        "translation": best_t.tolist(),
        "transform_4x4": transform.tolist(),
    }
    if args.metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)), exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Aligned mesh written: {args.out}")
    print(f"Transform written: {args.transform_txt}")
    print(
        "Fit stats: "
        f"mean={metrics['nn_mean']:.6f}, "
        f"median={metrics['nn_median']:.6f}, "
        f"p95={metrics['nn_p95']:.6f}"
    )


if __name__ == "__main__":
    main()
