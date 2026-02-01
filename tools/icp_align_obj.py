#!/usr/bin/env python3
"""
Simple ICP alignment for OBJ meshes (vertex-only alignment).
Uses numpy; optionally uses scipy for faster nearest neighbors if available.
Writes a transformed OBJ and a 3x4 transform matrix (R|t).
"""

import argparse
import os
import sys
import numpy as np


def load_vertices(path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.array(verts, dtype=np.float64)


def kabsch(A, B):
    # A, B: Nx3
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def nearest_neighbor(src, dst):
    # src: Nx3, dst: Mx3
    # Try scipy KDTree for speed, fallback to brute force.
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(dst)
        _, idx = tree.query(src, k=1)
        return dst[idx]
    except Exception:
        matched = np.zeros_like(src)
        for i in range(src.shape[0]):
            diff = dst - src[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            matched[i] = dst[np.argmin(d2)]
        return matched


def icp(src, dst, iters=25, sample_n=2000):
    T_R = np.eye(3)
    T_t = np.zeros(3)

    if src.shape[0] > sample_n:
        src_idx = np.random.choice(src.shape[0], sample_n, replace=False)
    else:
        src_idx = np.arange(src.shape[0])
    if dst.shape[0] > sample_n:
        dst_idx = np.random.choice(dst.shape[0], sample_n, replace=False)
    else:
        dst_idx = np.arange(dst.shape[0])

    src_s = src[src_idx]
    dst_s = dst[dst_idx]

    for _ in range(iters):
        src_trans = (T_R @ src_s.T).T + T_t
        matched = nearest_neighbor(src_trans, dst_s)
        R, t = kabsch(src_trans, matched)
        T_t = R @ T_t + t
        T_R = R @ T_R

    return T_R, T_t


def apply_transform_to_obj(src_path, out_path, R, t):
    with open(src_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    v2 = R @ v + t
                    f_out.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                else:
                    f_out.write(line)
            else:
                f_out.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="OBJ to transform (moving mesh)")
    ap.add_argument("--dst", required=True, help="OBJ to align to (fixed mesh)")
    ap.add_argument("--out", required=True, help="Output OBJ path")
    ap.add_argument("--mat", required=True, help="Output transform txt (3x4)")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    src = load_vertices(args.src)
    dst = load_vertices(args.dst)

    R, t = icp(src, dst, iters=args.iters, sample_n=args.sample)
    apply_transform_to_obj(args.src, args.out, R, t)

    np.savetxt(args.mat, np.hstack([R, t.reshape(3, 1)]), fmt="%.8f")
    print("Aligned mesh written to", args.out)
    print("Transform written to", args.mat)


if __name__ == "__main__":
    main()
