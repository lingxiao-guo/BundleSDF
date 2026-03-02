#!/usr/bin/env python3
"""
Refine mesh_aligned.obj to a real reconstructed mesh with a partial-aware affine model.

Model:
- Camera-derived seen area only on the real mesh.
- Axis-constrained affine in the frame from sam3d_aligned_axes.json:
    v ~= diag(sx, sy, sz) * u + t_axis
  where u, v are source/destination points in axis coordinates.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


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


def load_transform_4x4(path: Path) -> np.ndarray:
    t = np.loadtxt(str(path), dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform at {path}, got {t.shape}")
    return t


def orthonormalize_basis(a: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def parse_axis_json(axis_json_path: Path) -> Tuple[np.ndarray, np.ndarray, Path, int]:
    with open(axis_json_path, "r") as f:
        data = json.load(f)

    if "axis_dirs_world" not in data:
        raise KeyError(f"'axis_dirs_world' missing in {axis_json_path}")
    if "center_world" not in data:
        raise KeyError(f"'center_world' missing in {axis_json_path}")
    if "transform_txt" not in data:
        raise KeyError(f"'transform_txt' missing in {axis_json_path}")

    axis_dirs_world = np.asarray(data["axis_dirs_world"], dtype=np.float64)
    if axis_dirs_world.shape != (3, 3):
        raise ValueError(
            f"axis_dirs_world must be 3x3 in {axis_json_path}, got {axis_dirs_world.shape}"
        )
    center_world = np.asarray(data["center_world"], dtype=np.float64).reshape(3)
    # axis_dirs_world stores [x_dir, y_dir, z_dir] in world coords.
    # Use columns as world basis vectors.
    a_raw = axis_dirs_world.T
    a = orthonormalize_basis(a_raw)

    transform_txt_raw = str(data["transform_txt"]).strip()
    if not transform_txt_raw:
        raise ValueError(f"Empty transform_txt in {axis_json_path}")
    transform_txt = resolve_path(transform_txt_raw, axis_json_path.parent)
    if "long_axis_index" in data:
        long_axis_index = int(data["long_axis_index"])
    else:
        ext = np.asarray(data.get("axis_extents_local", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
        long_axis_index = int(np.argmax(ext))
    long_axis_index = int(np.clip(long_axis_index, 0, 2))
    return a, center_world, transform_txt, long_axis_index


def make_raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def read_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    d = cv2.imread(str(path), -1)
    if d is None:
        raise FileNotFoundError(path)
    d = d.astype(np.float32)
    if depth_scale <= 0.0:
        raise ValueError("real_depth_scale must be > 0")
    d = d / float(depth_scale)
    return d


def read_mask(path: Path, threshold: int) -> np.ndarray:
    m = cv2.imread(str(path), -1)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > int(threshold)).astype(np.uint8)


def collect_common_frame_ids(depth_dir: Path, mask_dir: Path, pose_dir: Path) -> List[str]:
    def stems(path: Path, suffix: str) -> set:
        out = set()
        for p in path.glob(f"*{suffix}"):
            out.add(p.stem)
        return out

    ids = stems(depth_dir, ".png") & stems(mask_dir, ".png") & stems(pose_dir, ".txt")
    return sorted(ids)


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = k.copy().astype(np.float64)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def build_pixel_grid_and_dirs(
    k: np.ndarray, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.arange(0, w, dtype=np.float32)
    v = np.arange(0, h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    u_flat = uu.reshape(-1).astype(np.int32)
    v_flat = vv.reshape(-1).astype(np.int32)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    dirs = np.stack(
        [(u_flat - cx) / fx, (v_flat - cy) / fy, np.ones_like(u_flat)], axis=1
    ).astype(np.float32)
    n = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs = dirs / n
    return u_flat, v_flat, dirs


def transform_rays_to_object(
    dirs_cam: np.ndarray, ob_in_cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cam_in_ob = np.linalg.inv(ob_in_cam).astype(np.float64)
    r = cam_in_ob[:3, :3]
    t = cam_in_ob[:3, 3]
    dirs_obj = (r @ dirs_cam.T).T
    dirs_obj /= np.linalg.norm(dirs_obj, axis=1, keepdims=True) + 1e-12
    origins_obj = np.repeat(t[None, :], len(dirs_obj), axis=0)
    return origins_obj, dirs_obj


def cast_rays(
    scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, dirs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    prim_ids = ans["primitive_ids"].numpy().reshape(-1).astype(np.int64, copy=False)
    t_hit = ans["t_hit"].numpy().reshape(-1).astype(np.float64)
    invalid_u32 = np.iinfo(np.uint32).max
    valid = np.isfinite(t_hit) & (prim_ids >= 0) & (prim_ids != invalid_u32)
    hit_obj = origins + dirs * t_hit[:, None]
    return prim_ids, t_hit, hit_obj, valid


def estimate_seen_faces(
    real_mesh: trimesh.Trimesh,
    cam_k: np.ndarray,
    depth_dir: Path,
    mask_dir: Path,
    pose_dir: Path,
    frame_stride: int,
    max_real_frames: int,
    estimation_downscale: int,
    mask_threshold: int,
    depth_abs_tol_m: float,
    depth_rel_tol: float,
    seen_threshold: float,
    real_depth_scale: float,
) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    frame_ids = collect_common_frame_ids(depth_dir, mask_dir, pose_dir)
    if frame_stride > 1:
        frame_ids = frame_ids[::frame_stride]
    if max_real_frames > 0:
        frame_ids = frame_ids[:max_real_frames]
    if len(frame_ids) == 0:
        raise RuntimeError(
            f"No overlapping depth/mask/pose frames under {depth_dir}, {mask_dir}, {pose_dir}"
        )

    first_depth = read_depth_m(
        depth_dir / f"{frame_ids[0]}.png", depth_scale=float(real_depth_scale)
    )
    h, w = first_depth.shape[:2]
    down = max(int(estimation_downscale), 1)
    h_est = max(1, h // down)
    w_est = max(1, w // down)
    sx = float(w_est) / float(w)
    sy = float(h_est) / float(h)
    cam_k_est = scale_intrinsics(cam_k, sx=sx, sy=sy)
    u_idx, v_idx, dirs_cam = build_pixel_grid_and_dirs(cam_k_est, h_est, w_est)

    scene = make_raycast_scene(real_mesh)
    n_faces = len(real_mesh.faces)
    visible_votes = np.zeros((n_faces,), dtype=np.float64)
    seen_votes = np.zeros((n_faces,), dtype=np.float64)

    for i, fid in enumerate(frame_ids):
        ob_in_cam = np.loadtxt(str(pose_dir / f"{fid}.txt"), dtype=np.float64).reshape(4, 4)
        depth_m = read_depth_m(
            depth_dir / f"{fid}.png", depth_scale=float(real_depth_scale)
        )
        mask = read_mask(mask_dir / f"{fid}.png", threshold=mask_threshold)
        if depth_m.shape[:2] != (h_est, w_est):
            depth_m = cv2.resize(depth_m, (w_est, h_est), interpolation=cv2.INTER_NEAREST)
        if mask.shape[:2] != (h_est, w_est):
            mask = cv2.resize(mask, (w_est, h_est), interpolation=cv2.INTER_NEAREST)

        origins_obj, dirs_obj = transform_rays_to_object(dirs_cam, ob_in_cam)
        prim_ids, _t_hit, hit_obj, valid = cast_rays(scene, origins_obj, dirs_obj)

        sampled_depth = depth_m[v_idx, u_idx]
        sampled_mask = mask[v_idx, u_idx] > 0
        real_valid = sampled_depth > 1e-6
        considered = valid & sampled_mask & real_valid
        if not np.any(considered):
            continue

        visible_faces = prim_ids[considered]
        np.add.at(visible_votes, visible_faces, 1.0)

        pred_depth = np.full(sampled_depth.shape, np.inf, dtype=np.float64)
        valid_idx = np.where(valid)[0]
        hit_cam = (ob_in_cam[:3, :3] @ hit_obj[valid_idx].T).T + ob_in_cam[:3, 3]
        pred_depth[valid_idx] = hit_cam[:, 2]
        tol = np.maximum(depth_abs_tol_m, depth_rel_tol * np.maximum(pred_depth, 1e-6))

        cidx = np.where(considered)[0]
        good = np.abs(sampled_depth[cidx] - pred_depth[cidx]) <= tol[cidx]
        if np.any(good):
            seen_faces = prim_ids[cidx[good]]
            np.add.at(seen_votes, seen_faces, 1.0)

        if (i + 1) % 20 == 0 or (i + 1) == len(frame_ids):
            print(f"[seen] processed {i + 1}/{len(frame_ids)} frames")

    seen_conf = np.zeros((n_faces,), dtype=np.float64)
    nz = visible_votes > 0.0
    seen_conf[nz] = seen_votes[nz] / np.maximum(visible_votes[nz], 1e-12)
    seen_face_mask = seen_conf >= float(seen_threshold)

    summary = {
        "frames_used": int(len(frame_ids)),
        "total_faces": int(n_faces),
        "visible_faces": int(np.count_nonzero(visible_votes > 0.0)),
        "seen_faces": int(np.count_nonzero(seen_face_mask)),
        "seen_face_ratio": float(np.count_nonzero(seen_face_mask)) / float(max(n_faces, 1)),
        "median_seen_conf": float(np.median(seen_conf)),
        "frame_stride": int(frame_stride),
        "max_real_frames": int(max_real_frames),
        "estimation_downscale": int(estimation_downscale),
        "mask_threshold": int(mask_threshold),
        "depth_abs_tol_m": float(depth_abs_tol_m),
        "depth_rel_tol": float(depth_rel_tol),
        "real_depth_scale": float(real_depth_scale),
        "seen_threshold": float(seen_threshold),
    }
    return seen_face_mask, summary, frame_ids


def sample_points_on_faces(
    mesh: trimesh.Trimesh, face_indices: np.ndarray, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    if len(face_indices) == 0:
        raise RuntimeError("No faces available for sampling.")
    areas = mesh.area_faces[face_indices].astype(np.float64)
    area_sum = float(np.sum(areas))
    if area_sum <= 1e-16:
        raise RuntimeError("Selected faces have near-zero total area.")
    prob = areas / area_sum
    chosen = rng.choice(face_indices, size=int(n_points), replace=True, p=prob)

    tri = np.asarray(mesh.faces, dtype=np.int64)[chosen]
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v0 = v[tri[:, 0]]
    v1 = v[tri[:, 1]]
    v2 = v[tri[:, 2]]

    r1 = rng.random(int(n_points))
    r2 = rng.random(int(n_points))
    sr1 = np.sqrt(r1)
    b0 = 1.0 - sr1
    b1 = sr1 * (1.0 - r2)
    b2 = sr1 * r2
    pts = b0[:, None] * v0 + b1[:, None] * v1 + b2[:, None] * v2
    return pts


def sample_points_on_mesh(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    all_faces = np.arange(len(mesh.faces), dtype=np.int64)
    return sample_points_on_faces(mesh, all_faces, n_points, rng)


def sample_points_normals_on_mesh(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    face_indices = np.arange(len(mesh.faces), dtype=np.int64)
    areas = mesh.area_faces[face_indices].astype(np.float64)
    area_sum = float(np.sum(areas))
    if area_sum <= 1e-16:
        raise RuntimeError("Mesh has near-zero total area.")
    prob = areas / area_sum
    chosen = rng.choice(face_indices, size=int(n_points), replace=True, p=prob)

    tri = np.asarray(mesh.faces, dtype=np.int64)[chosen]
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v0 = v[tri[:, 0]]
    v1 = v[tri[:, 1]]
    v2 = v[tri[:, 2]]

    r1 = rng.random(int(n_points))
    r2 = rng.random(int(n_points))
    sr1 = np.sqrt(r1)
    b0 = 1.0 - sr1
    b1 = sr1 * (1.0 - r2)
    b2 = sr1 * r2
    pts = b0[:, None] * v0 + b1[:, None] * v1 + b2[:, None] * v2

    fn = np.asarray(mesh.face_normals, dtype=np.float64)[chosen]
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn = fn / fn_norm
    return pts, fn


def transform_with_axis_model(
    points_world: np.ndarray, a: np.ndarray, c: np.ndarray, scale: np.ndarray, t_axis: np.ndarray
) -> np.ndarray:
    u = (a.T @ (points_world - c[None, :]).T).T
    u2 = u * scale[None, :] + t_axis[None, :]
    return c[None, :] + (u2 @ a.T)


def huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    w = np.ones_like(norms, dtype=np.float64)
    mask = norms > delta
    w[mask] = delta / (norms[mask] + 1e-12)
    return w


def huber_loss(norms: np.ndarray, delta: float) -> np.ndarray:
    out = np.empty_like(norms, dtype=np.float64)
    mask = norms <= delta
    out[mask] = 0.5 * norms[mask] ** 2
    out[~mask] = delta * (norms[~mask] - 0.5 * delta)
    return out


def dist_stats(d: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(np.max(d)),
    }


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(w)
    k = w / theta
    kx = skew(k)
    s = np.sin(theta)
    c = np.cos(theta)
    return np.eye(3, dtype=np.float64) + s * kx + (1.0 - c) * (kx @ kx)


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / n
    kx = skew(a)
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    return np.eye(3, dtype=np.float64) + s * kx + (1.0 - c) * (kx @ kx)


def rigid_to_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def apply_rigid(
    pts: np.ndarray, nrm: np.ndarray, r: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pts2 = (r @ pts.T).T + t[None, :]
    nrm2 = (r @ nrm.T).T
    nrm2 = nrm2 / (np.linalg.norm(nrm2, axis=1, keepdims=True) + 1e-12)
    return pts2, nrm2


def trimmed_mean(d: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.5, 1.0))
    c = np.quantile(d, q)
    keep = d <= c
    if int(np.count_nonzero(keep)) == 0:
        return float(np.mean(d))
    return float(np.mean(d[keep]))


def build_global_silhouette_context(
    frame_ids: List[str],
    mask_dir: Path,
    pose_dir: Path,
    cam_k: np.ndarray,
    downscale: int,
    max_frames: int,
) -> Dict[str, object]:
    if len(frame_ids) == 0 or max_frames <= 0:
        return {"enabled": False}

    if len(frame_ids) <= max_frames:
        selected = frame_ids
    else:
        idx = np.linspace(0, len(frame_ids) - 1, max_frames).round().astype(np.int64)
        selected = [frame_ids[int(i)] for i in idx]

    first = read_mask(mask_dir / f"{selected[0]}.png", threshold=0)
    h0, w0 = first.shape[:2]
    down = max(int(downscale), 1)
    h = max(1, h0 // down)
    w = max(1, w0 // down)
    k_est = scale_intrinsics(cam_k, sx=float(w) / float(w0), sy=float(h) / float(h0))
    fx, fy = float(k_est[0, 0]), float(k_est[1, 1])
    cx, cy = float(k_est[0, 2]), float(k_est[1, 2])

    masks = []
    poses = []
    for fid in selected:
        m = read_mask(mask_dir / f"{fid}.png", threshold=0).astype(bool)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(m.reshape(-1))
        ob_in_cam = np.loadtxt(str(pose_dir / f"{fid}.txt"), dtype=np.float64).reshape(4, 4)
        poses.append(ob_in_cam)

    return {
        "enabled": True,
        "w": int(w),
        "h": int(h),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "masks_flat": masks,
        "poses": poses,
        "num_frames": int(len(selected)),
        "downscale": int(down),
    }


def candidate_silhouette_iou(
    src_pts_rot: np.ndarray,
    sil_ctx: Dict[str, object],
) -> float:
    if not bool(sil_ctx.get("enabled", False)):
        return 0.0
    w = int(sil_ctx["w"])
    h = int(sil_ctx["h"])
    fx = float(sil_ctx["fx"])
    fy = float(sil_ctx["fy"])
    cx = float(sil_ctx["cx"])
    cy = float(sil_ctx["cy"])
    masks_flat = sil_ctx["masks_flat"]
    poses = sil_ctx["poses"]

    ious = []
    hw = int(w * h)
    for ob_in_cam, mask_flat in zip(poses, masks_flat):
        pts_cam = (ob_in_cam[:3, :3] @ src_pts_rot.T).T + ob_in_cam[:3, 3]
        z = pts_cam[:, 2]
        valid = z > 1e-6
        if int(np.count_nonzero(valid)) < 16:
            continue
        uv = np.zeros((int(np.count_nonzero(valid)), 2), dtype=np.int64)
        p = pts_cam[valid]
        uv[:, 0] = np.round(fx * (p[:, 0] / p[:, 2]) + cx).astype(np.int64)
        uv[:, 1] = np.round(fy * (p[:, 1] / p[:, 2]) + cy).astype(np.int64)
        in_img = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        if int(np.count_nonzero(in_img)) < 16:
            continue
        pix = uv[in_img, 1] * w + uv[in_img, 0]
        z_vis = p[in_img, 2]
        zbuf = np.full((hw,), np.inf, dtype=np.float64)
        np.minimum.at(zbuf, pix, z_vis)
        occ = np.isfinite(zbuf)
        inter = int(np.count_nonzero(occ & mask_flat))
        union = int(np.count_nonzero(occ | mask_flat))
        if union == 0:
            continue
        ious.append(float(inter) / float(union))

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))


def find_best_global_init_rotation(
    src_pts: np.ndarray,
    dst_seen_pts: np.ndarray,
    a: np.ndarray,
    long_axis_index: int,
    yaw_step_deg: float,
    tilt_deg: float,
    eval_trim_quantile: float,
    silhouette_weight: float = 0.0,
    silhouette_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    long_axis = a[:, int(np.clip(long_axis_index, 0, 2))]
    other_idx = [i for i in [0, 1, 2] if i != int(np.clip(long_axis_index, 0, 2))]

    step = max(float(yaw_step_deg), 1e-3)
    yaw_angles = np.arange(-180.0, 180.0 + 1e-6, step, dtype=np.float64)
    tilt = float(max(0.0, tilt_deg))

    candidates: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for yaw in yaw_angles:
        r_yaw = rotation_about_axis(long_axis, np.deg2rad(float(yaw)))
        candidates.append(r_yaw)
        if tilt > 1e-6:
            for oi in other_idx:
                axis_tilt = r_yaw @ a[:, oi]
                candidates.append(rotation_about_axis(axis_tilt, np.deg2rad(+tilt)) @ r_yaw)
                candidates.append(rotation_about_axis(axis_tilt, np.deg2rad(-tilt)) @ r_yaw)

    best_cost = np.inf
    best_nn_cost = np.inf
    best_iou = 0.0
    best_r = np.eye(3, dtype=np.float64)
    sil_ctx = silhouette_context if silhouette_context is not None else {"enabled": False}
    for i, r0 in enumerate(candidates):
        src_rot = (r0 @ src_pts.T).T
        tree = cKDTree(src_rot)
        d, _ = tree.query(dst_seen_pts, k=1)
        nn_cost = trimmed_mean(d, q=float(eval_trim_quantile))
        sil_iou = candidate_silhouette_iou(src_rot, sil_ctx=sil_ctx)
        cost = float(nn_cost + float(silhouette_weight) * (1.0 - sil_iou))
        if cost < best_cost:
            best_cost = float(cost)
            best_nn_cost = float(nn_cost)
            best_iou = float(sil_iou)
            best_r = r0.copy()
        if (i + 1) % 16 == 0 or (i + 1) == len(candidates):
            print(
                f"[global] candidate {i + 1:03d}/{len(candidates)} "
                f"best_cost={best_cost:.8f} best_nn={best_nn_cost:.8f} best_iou={best_iou:.4f}"
            )

    return {
        "rotation": best_r,
        "cost_trimmed_mean": float(best_cost),
        "cost_nn_trimmed_mean": float(best_nn_cost),
        "silhouette_iou": float(best_iou),
        "num_candidates": int(len(candidates)),
        "yaw_step_deg": float(yaw_step_deg),
        "tilt_deg": float(tilt_deg),
        "eval_trim_quantile": float(eval_trim_quantile),
        "silhouette_weight": float(silhouette_weight),
    }


def huber_weights_scalar(abs_residual: np.ndarray, delta: float) -> np.ndarray:
    w = np.ones_like(abs_residual, dtype=np.float64)
    mask = abs_residual > float(delta)
    w[mask] = float(delta) / (abs_residual[mask] + 1e-12)
    return w


def rigid_prealign_seen_point_to_plane(
    src_pts: np.ndarray,
    src_normals: np.ndarray,
    dst_seen_pts: np.ndarray,
    max_iter: int,
    trim_quantile: float,
    huber_delta: float,
    max_step_deg: float,
    max_total_deg: float,
    reg: float,
) -> Dict[str, object]:
    r = np.eye(3, dtype=np.float64)
    t = np.zeros((3,), dtype=np.float64)

    best = {
        "objective": np.inf,
        "iter": -1,
        "rotation": r.copy(),
        "translation": t.copy(),
        "inliers": 0,
        "objective_p2l": np.inf,
    }

    max_step_rad = np.deg2rad(float(max_step_deg))
    max_total_rad = np.deg2rad(float(max_total_deg))

    for it in range(int(max_iter)):
        src_cur, nrm_cur = apply_rigid(src_pts, src_normals, r=r, t=t)
        tree = cKDTree(src_cur)
        _d, idx = tree.query(dst_seen_pts, k=1)

        p = src_cur[idx]
        n = nrm_cur[idx]
        y = dst_seen_pts
        residual = np.sum(n * (p - y), axis=1)
        abs_r = np.abs(residual)

        cutoff = np.quantile(abs_r, float(trim_quantile))
        keep = abs_r <= cutoff
        if int(np.count_nonzero(keep)) < 64:
            raise RuntimeError("Too few inliers during rigid pre-alignment.")

        p_k = p[keep]
        n_k = n[keep]
        r_k = residual[keep]
        w = huber_weights_scalar(np.abs(r_k), delta=float(huber_delta))

        j_rot = np.cross(p_k, n_k)
        j = np.concatenate([j_rot, n_k], axis=1)

        jw = j * w[:, None]
        a_mat = j.T @ jw
        b_vec = -j.T @ (w * r_k)
        a_mat += float(reg) * np.eye(6, dtype=np.float64)

        try:
            delta = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        dw = delta[:3]
        dt = delta[3:]

        step_rot = float(np.linalg.norm(dw))
        if step_rot > max_step_rad:
            dw *= max_step_rad / max(step_rot, 1e-12)
        r_inc = exp_so3(dw)
        r_new = r_inc @ r
        t_new = r_inc @ t + dt

        # Clamp total rigid correction to remain small and stable.
        rotvec_total, _ = cv2.Rodrigues(r_new)
        rotvec_total = rotvec_total.reshape(3).astype(np.float64)
        angle_total = float(np.linalg.norm(rotvec_total))
        if angle_total > max_total_rad:
            rotvec_total *= max_total_rad / max(angle_total, 1e-12)
            r_new = exp_so3(rotvec_total)

        src_new, nrm_new = apply_rigid(src_pts, src_normals, r=r_new, t=t_new)
        tree_new = cKDTree(src_new)
        _d2, idx2 = tree_new.query(dst_seen_pts, k=1)
        p2 = src_new[idx2]
        n2 = nrm_new[idx2]
        residual2 = np.sum(n2 * (p2 - dst_seen_pts), axis=1)
        abs_r2 = np.abs(residual2)
        cutoff2 = np.quantile(abs_r2, float(trim_quantile))
        keep2 = abs_r2 <= cutoff2
        obj_p2l = float(np.mean(huber_loss(abs_r2[keep2], delta=float(huber_delta))))
        obj = obj_p2l + float(reg) * (np.sum(rotvec_total**2) + np.sum(t_new**2))

        if obj < best["objective"]:
            best = {
                "objective": obj,
                "iter": int(it),
                "rotation": r_new.copy(),
                "translation": t_new.copy(),
                "inliers": int(np.count_nonzero(keep2)),
                "objective_p2l": obj_p2l,
            }

        step = max(float(np.linalg.norm(dw)), float(np.linalg.norm(dt)))
        r = r_new
        t = t_new
        print(
            f"[rigid] iter {it + 1:03d}/{max_iter} obj={obj:.8e} "
            f"p2l={obj_p2l:.8e} inliers={int(np.count_nonzero(keep2))} step={step:.3e}"
        )
        if step < 1e-7:
            break

    return best


def optimize_axis_affine_partial_aware(
    src_pts: np.ndarray,
    dst_seen_pts: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    max_iter: int,
    trim_quantile: float,
    huber_delta: float,
    scale_reg: float,
    scale_clamp_min: float,
    scale_clamp_max: float,
    secondary_weight: float,
    secondary_shell_quantile: float,
    affine_axis_mode: str,
    single_axis_index: int,
) -> Dict[str, object]:
    src_axis = (a.T @ (src_pts - c[None, :]).T).T
    dst_axis = (a.T @ (dst_seen_pts - c[None, :]).T).T

    scale = np.ones((3,), dtype=np.float64)
    t_axis = np.zeros((3,), dtype=np.float64)
    tree_dst = cKDTree(dst_seen_pts)

    mode = str(affine_axis_mode).strip().lower()
    if mode not in ("all_free", "single_axis_with_shared_other"):
        raise ValueError(
            "affine_axis_mode must be one of: all_free, single_axis_with_shared_other"
        )
    single_axis_index = int(np.clip(int(single_axis_index), 0, 2))

    best = {
        "objective": np.inf,
        "iter": -1,
        "scale": scale.copy(),
        "t_axis": t_axis.copy(),
        "inliers": 0,
        "objective_primary": np.inf,
        "objective_secondary": np.inf,
        "secondary_pairs": 0,
    }
    best_primary = np.inf

    for it in range(int(max_iter)):
        src_trans_axis = src_axis * scale[None, :] + t_axis[None, :]
        src_trans_world = c[None, :] + (src_trans_axis @ a.T)
        tree = cKDTree(src_trans_world)
        _dists, idx = tree.query(dst_seen_pts, k=1)
        matched_src_axis = src_axis[idx]

        residual = matched_src_axis * scale[None, :] + t_axis[None, :] - dst_axis
        r_norm = np.linalg.norm(residual, axis=1)
        cutoff = np.quantile(r_norm, float(trim_quantile))
        keep = r_norm <= cutoff
        if int(np.count_nonzero(keep)) < 32:
            raise RuntimeError("Too few inliers during optimization.")

        w_p = huber_weights(r_norm[keep], delta=float(huber_delta))
        ms_p = matched_src_axis[keep]
        ds_p = dst_axis[keep]

        # Weak secondary term: source_seen_shell -> real_seen.
        d_s2d, idx_s2d = tree_dst.query(src_trans_world, k=1)
        shell_q = float(np.clip(secondary_shell_quantile, 0.05, 1.0))
        shell_cut = np.quantile(d_s2d, shell_q)
        shell = d_s2d <= shell_cut
        ms_s = src_axis[shell]
        ds_s = dst_axis[idx_s2d[shell]]
        if len(ms_s) > 0:
            residual_s = ms_s * scale[None, :] + t_axis[None, :] - ds_s
            r_norm_s = np.linalg.norm(residual_s, axis=1)
            w_s = huber_weights(r_norm_s, delta=float(huber_delta))
        else:
            residual_s = np.zeros((0, 3), dtype=np.float64)
            r_norm_s = np.zeros((0,), dtype=np.float64)
            w_s = np.zeros((0,), dtype=np.float64)

        def collect_axis_observations(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            u = ms_p[:, k]
            v = ds_p[:, k]
            ww = w_p.copy()
            if len(ms_s) > 0 and secondary_weight > 0.0:
                u = np.concatenate([u, ms_s[:, k]], axis=0)
                v = np.concatenate([v, ds_s[:, k]], axis=0)
                ww = np.concatenate([ww, float(secondary_weight) * w_s], axis=0)
            return u, v, ww

        scale_new = scale.copy()
        t_axis_new = t_axis.copy()
        if mode == "all_free":
            for k in range(3):
                u, v, ww = collect_axis_observations(k)
                su2 = np.sum(ww * u * u)
                su = np.sum(ww * u)
                sw = np.sum(ww)
                suv = np.sum(ww * u * v)
                sv = np.sum(ww * v)
                m = np.array([[su2 + scale_reg, su], [su, sw + 1e-12]], dtype=np.float64)
                b = np.array([suv + scale_reg * 1.0, sv], dtype=np.float64)
                try:
                    x = np.linalg.solve(m, b)
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(m, b, rcond=None)[0]
                scale_new[k] = x[0]
                t_axis_new[k] = x[1]
            scale_new = np.clip(scale_new, float(scale_clamp_min), float(scale_clamp_max))
        else:
            # Single-axis mode:
            # - selected axis has independent scale
            # - remaining two axes share one common scale
            k_main = int(single_axis_index)
            k_other = [i for i in [0, 1, 2] if i != k_main]
            k0, k1 = int(k_other[0]), int(k_other[1])

            # Solve main axis independently.
            u_m, v_m, w_m = collect_axis_observations(k_main)
            su2 = np.sum(w_m * u_m * u_m)
            su = np.sum(w_m * u_m)
            sw = np.sum(w_m)
            suv = np.sum(w_m * u_m * v_m)
            sv = np.sum(w_m * v_m)
            m_main = np.array([[su2 + scale_reg, su], [su, sw + 1e-12]], dtype=np.float64)
            b_main = np.array([suv + scale_reg * 1.0, sv], dtype=np.float64)
            try:
                x_main = np.linalg.solve(m_main, b_main)
            except np.linalg.LinAlgError:
                x_main = np.linalg.lstsq(m_main, b_main, rcond=None)[0]
            s_main = float(np.clip(x_main[0], float(scale_clamp_min), float(scale_clamp_max)))
            t_main = float(np.sum(w_m * (v_m - s_main * u_m)) / (np.sum(w_m) + 1e-12))

            # Joint solve for shared scale on the other two axes.
            u0, v0, w0 = collect_axis_observations(k0)
            u1, v1, w1 = collect_axis_observations(k1)

            h00 = (
                np.sum(w0 * u0 * u0)
                + np.sum(w1 * u1 * u1)
                + float(scale_reg)
            )
            h01 = np.sum(w0 * u0)
            h02 = np.sum(w1 * u1)
            h11 = np.sum(w0) + 1e-12
            h22 = np.sum(w1) + 1e-12
            h = np.array(
                [[h00, h01, h02], [h01, h11, 0.0], [h02, 0.0, h22]],
                dtype=np.float64,
            )
            g0 = (
                np.sum(w0 * u0 * v0)
                + np.sum(w1 * u1 * v1)
                + float(scale_reg) * 1.0
            )
            g1 = np.sum(w0 * v0)
            g2 = np.sum(w1 * v1)
            g = np.array([g0, g1, g2], dtype=np.float64)
            try:
                x_shared = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                x_shared = np.linalg.lstsq(h, g, rcond=None)[0]
            s_shared = float(
                np.clip(x_shared[0], float(scale_clamp_min), float(scale_clamp_max))
            )
            t0 = float(np.sum(w0 * (v0 - s_shared * u0)) / (np.sum(w0) + 1e-12))
            t1 = float(np.sum(w1 * (v1 - s_shared * u1)) / (np.sum(w1) + 1e-12))

            scale_new[k_main] = s_main
            t_axis_new[k_main] = t_main
            scale_new[k0] = s_shared
            scale_new[k1] = s_shared
            t_axis_new[k0] = t0
            t_axis_new[k1] = t1

        residual_new = matched_src_axis * scale_new[None, :] + t_axis_new[None, :] - dst_axis
        r_norm_new = np.linalg.norm(residual_new, axis=1)
        cutoff_new = np.quantile(r_norm_new, float(trim_quantile))
        keep_new = r_norm_new <= cutoff_new
        robust_primary = np.mean(huber_loss(r_norm_new[keep_new], delta=float(huber_delta)))

        if len(ms_s) > 0 and secondary_weight > 0.0:
            residual_s_new = ms_s * scale_new[None, :] + t_axis_new[None, :] - ds_s
            r_norm_s_new = np.linalg.norm(residual_s_new, axis=1)
            robust_secondary = np.mean(
                huber_loss(r_norm_s_new, delta=float(huber_delta))
            )
        else:
            robust_secondary = 0.0

        reg = float(scale_reg) * float(np.sum((scale_new - 1.0) ** 2))
        obj = float(robust_primary + float(secondary_weight) * robust_secondary + reg)

        if (robust_primary < best_primary - 1e-12) or (
            abs(robust_primary - best_primary) <= 1e-12 and obj < best["objective"]
        ):
            best_primary = float(robust_primary)
            best = {
                "objective": obj,
                "iter": int(it),
                "scale": scale_new.copy(),
                "t_axis": t_axis_new.copy(),
                "inliers": int(np.count_nonzero(keep_new)),
                "objective_primary": float(robust_primary),
                "objective_secondary": float(robust_secondary),
                "secondary_pairs": int(len(ms_s)),
            }

        step = np.max(np.abs(np.concatenate([scale_new - scale, t_axis_new - t_axis])))
        scale = scale_new
        t_axis = t_axis_new
        print(
            f"[opt] iter {it + 1:03d}/{max_iter} obj={obj:.8e} "
            f"primary={float(robust_primary):.8e} "
            f"secondary={float(robust_secondary):.8e} "
            f"inliers={int(np.count_nonzero(keep_new))} "
            f"secondary_pairs={int(len(ms_s))} step={step:.3e}"
        )
        if step < 1e-7:
            break

    return best


def build_delta_transform(
    a: np.ndarray, c: np.ndarray, scale: np.ndarray, t_axis: np.ndarray
) -> np.ndarray:
    lin = a @ np.diag(scale) @ a.T
    t_world = c + a @ t_axis - lin @ c
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = lin
    t[:3, 3] = t_world
    return t


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Axis-constrained partial-aware affine refinement for mesh_aligned.obj."
    )
    ap.add_argument("--source_mesh_aligned", required=True, type=str)
    ap.add_argument("--real_mesh", required=True, type=str)
    ap.add_argument("--axis_json", type=str, default="")
    ap.add_argument("--cam_k", type=str, default="")
    ap.add_argument("--real_depth_dir", type=str, default="")
    ap.add_argument("--real_mask_dir", type=str, default="")
    ap.add_argument("--real_pose_dir", type=str, default="")
    ap.add_argument("--out_mesh", type=str, default="")
    ap.add_argument("--delta_transform_txt", type=str, default="")
    ap.add_argument("--full_transform_txt", type=str, default="")
    ap.add_argument("--metrics_json", type=str, default="")

    ap.add_argument("--sample_points", type=int, default=12000)
    ap.add_argument("--max_iter", type=int, default=60)
    ap.add_argument("--trim_quantile", type=float, default=0.97)
    ap.add_argument("--huber_delta", type=float, default=0.003)
    ap.add_argument(
        "--affine_axis",
        type=str,
        default="xyz",
        help=(
            "Axis scale control in aligned-axis frame. "
            "Single axis (x/y/z): that axis independent, other two share one scale. "
            "Multi-axis (xy/xz/yz/xyz): all three axes are independently scaled."
        ),
    )
    ap.add_argument("--scale_clamp_min", type=float, default=0.6)
    ap.add_argument("--scale_clamp_max", type=float, default=1.6)
    ap.add_argument("--scale_reg", type=float, default=0.03)
    ap.add_argument("--secondary_weight", type=float, default=0.15)
    ap.add_argument("--secondary_shell_quantile", type=float, default=0.55)
    ap.add_argument("--enable_global_init_search", type=int, default=1)
    ap.add_argument("--global_yaw_step_deg", type=float, default=30.0)
    ap.add_argument("--global_tilt_deg", type=float, default=18.0)
    ap.add_argument("--global_eval_trim_quantile", type=float, default=0.90)
    ap.add_argument("--global_silhouette_weight", type=float, default=0.8)
    ap.add_argument("--global_silhouette_downscale", type=int, default=4)
    ap.add_argument("--global_silhouette_frames", type=int, default=24)
    ap.add_argument("--enable_long_axis_twist_search", type=int, default=1)
    ap.add_argument("--twist_deg_max", type=float, default=50.0)
    ap.add_argument("--twist_deg_step", type=float, default=10.0)
    ap.add_argument("--twist_probe_iters", type=int, default=22)
    ap.add_argument("--rigid_max_iter", type=int, default=20)
    ap.add_argument("--rigid_trim_quantile", type=float, default=0.95)
    ap.add_argument("--rigid_huber_delta", type=float, default=0.002)
    ap.add_argument("--rigid_max_step_deg", type=float, default=1.0)
    ap.add_argument("--rigid_max_total_deg", type=float, default=15.0)
    ap.add_argument("--rigid_reg", type=float, default=1e-4)
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("--max_real_frames", type=int, default=180)
    ap.add_argument("--estimation_downscale", type=int, default=2)
    ap.add_argument("--depth_abs_tol_m", type=float, default=0.02)
    ap.add_argument("--depth_rel_tol", type=float, default=0.03)
    ap.add_argument("--seen_threshold", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_threshold", type=int, default=0)
    ap.add_argument("--real_depth_scale", type=float, default=1000.0)
    return ap.parse_args()


def parse_affine_axis_spec(spec: str) -> Dict[str, object]:
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    cleaned = str(spec).strip().lower().replace(",", "").replace(" ", "")
    if cleaned == "":
        raise ValueError("--affine_axis cannot be empty; use a subset of x,y,z (e.g., z or xyz).")

    free = np.zeros((3,), dtype=bool)
    for ch in cleaned:
        if ch not in axis_to_idx:
            raise ValueError(
                f"Invalid --affine_axis='{spec}'. Only characters x,y,z are allowed."
            )
        free[axis_to_idx[ch]] = True

    if not np.any(free):
        raise ValueError("--affine_axis selected no valid axes.")
    canonical = "".join([a for a in "xyz" if free[axis_to_idx[a]]])
    selected = [axis_to_idx[a] for a in "xyz" if free[axis_to_idx[a]]]
    if len(selected) == 1:
        k = int(selected[0])
        shared = [i for i in [0, 1, 2] if i != k]
        return {
            "canonical": canonical,
            "mode": "single_axis_with_shared_other",
            "single_axis_index": int(k),
            "single_axis_name": ["x", "y", "z"][k],
            "shared_axes": [(["x", "y", "z"][shared[0]]), (["x", "y", "z"][shared[1]])],
        }

    # Any multi-axis selection enables full independent xyz scaling.
    return {
        "canonical": canonical,
        "mode": "all_free",
        "single_axis_index": -1,
        "single_axis_name": "",
        "shared_axes": [],
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    affine_axis_policy = parse_affine_axis_spec(args.affine_axis)
    affine_axis_canonical = str(affine_axis_policy["canonical"])
    affine_axis_mode = str(affine_axis_policy["mode"])
    single_axis_index = int(affine_axis_policy["single_axis_index"])

    source_mesh_path = Path(args.source_mesh_aligned).expanduser().resolve()
    real_mesh_path = Path(args.real_mesh).expanduser().resolve()
    if not source_mesh_path.exists():
        raise FileNotFoundError(source_mesh_path)
    if not real_mesh_path.exists():
        raise FileNotFoundError(real_mesh_path)

    source_dir = source_mesh_path.parent
    real_dir = real_mesh_path.parent

    axis_json_path = (
        Path(args.axis_json).expanduser().resolve()
        if args.axis_json
        else (source_dir / "sam3d_aligned_axes.json").resolve()
    )
    cam_k_path = (
        Path(args.cam_k).expanduser().resolve()
        if args.cam_k
        else (real_dir / "cam_K.txt").resolve()
    )
    depth_dir = (
        Path(args.real_depth_dir).expanduser().resolve()
        if args.real_depth_dir
        else (real_dir / "depth").resolve()
    )
    mask_dir = (
        Path(args.real_mask_dir).expanduser().resolve()
        if args.real_mask_dir
        else (real_dir / "mask").resolve()
    )
    pose_dir = (
        Path(args.real_pose_dir).expanduser().resolve()
        if args.real_pose_dir
        else (real_dir / "ob_in_cam").resolve()
    )
    out_mesh_path = (
        Path(args.out_mesh).expanduser().resolve()
        if args.out_mesh
        else (source_dir / "mesh_affine_aligned_partialaware.obj").resolve()
    )
    delta_transform_path = (
        Path(args.delta_transform_txt).expanduser().resolve()
        if args.delta_transform_txt
        else (source_dir / "sam3d_affine_partialaware_delta_transform.txt").resolve()
    )
    full_transform_path = (
        Path(args.full_transform_txt).expanduser().resolve()
        if args.full_transform_txt
        else (source_dir / "sam3d_affine_partialaware_to_real_transform.txt").resolve()
    )
    metrics_path = (
        Path(args.metrics_json).expanduser().resolve()
        if args.metrics_json
        else (source_dir / "sam3d_affine_partialaware_to_real_metrics.json").resolve()
    )

    for p in [axis_json_path, cam_k_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    for p in [depth_dir, mask_dir, pose_dir]:
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Required directory not found: {p}")

    src_mesh = load_mesh(str(source_mesh_path))
    real_mesh = load_mesh(str(real_mesh_path))
    a, c, base_transform_txt, long_axis_index = parse_axis_json(axis_json_path)
    if not base_transform_txt.exists():
        raise FileNotFoundError(
            f"axis_json transform_txt does not exist: {base_transform_txt}"
        )
    base_transform = load_transform_4x4(base_transform_txt)

    cam_k = np.loadtxt(str(cam_k_path), dtype=np.float64).reshape(3, 3)
    seen_face_mask, seen_summary, _frame_ids = estimate_seen_faces(
        real_mesh=real_mesh,
        cam_k=cam_k,
        depth_dir=depth_dir,
        mask_dir=mask_dir,
        pose_dir=pose_dir,
        frame_stride=int(args.frame_stride),
        max_real_frames=int(args.max_real_frames),
        estimation_downscale=int(args.estimation_downscale),
        mask_threshold=int(args.mask_threshold),
        depth_abs_tol_m=float(args.depth_abs_tol_m),
        depth_rel_tol=float(args.depth_rel_tol),
        seen_threshold=float(args.seen_threshold),
        real_depth_scale=float(args.real_depth_scale),
    )
    if seen_summary["seen_face_ratio"] < 0.005:
        raise RuntimeError(
            "Seen-face ratio is too low for reliable refinement "
            f"({seen_summary['seen_face_ratio']:.6f} < 0.005). "
            "Check cam_K/depth/mask/ob_in_cam consistency."
        )

    seen_face_idx = np.where(seen_face_mask)[0].astype(np.int64)
    src_pts, src_normals = sample_points_normals_on_mesh(
        src_mesh, int(args.sample_points), rng=rng
    )
    dst_seen_pts = sample_points_on_faces(
        real_mesh, seen_face_idx, int(args.sample_points), rng=rng
    )

    # Baseline stats (before refinement): one-sided seen(real)->source.
    baseline_tree = cKDTree(src_pts)
    baseline_d, _ = baseline_tree.query(dst_seen_pts, k=1)
    baseline_stats = dist_stats(baseline_d)

    global_init = {
        "enabled": int(args.enable_global_init_search),
        "rotation": np.eye(3, dtype=np.float64),
        "cost_trimmed_mean": float(baseline_stats["mean"]),
        "num_candidates": 1,
    }
    if int(args.enable_global_init_search) == 1:
        global_sil_ctx = build_global_silhouette_context(
            frame_ids=_frame_ids,
            mask_dir=mask_dir,
            pose_dir=pose_dir,
            cam_k=cam_k,
            downscale=int(args.global_silhouette_downscale),
            max_frames=int(args.global_silhouette_frames),
        )
        global_init = find_best_global_init_rotation(
            src_pts=src_pts,
            dst_seen_pts=dst_seen_pts,
            a=a,
            long_axis_index=int(long_axis_index),
            yaw_step_deg=float(args.global_yaw_step_deg),
            tilt_deg=float(args.global_tilt_deg),
            eval_trim_quantile=float(args.global_eval_trim_quantile),
            silhouette_weight=float(args.global_silhouette_weight),
            silhouette_context=global_sil_ctx,
        )

    r_global = np.asarray(global_init["rotation"], dtype=np.float64)
    src_pts_global, src_normals_global = apply_rigid(
        src_pts, src_normals, r=r_global, t=np.zeros((3,), dtype=np.float64)
    )

    rigid_best = rigid_prealign_seen_point_to_plane(
        src_pts=src_pts_global,
        src_normals=src_normals_global,
        dst_seen_pts=dst_seen_pts,
        max_iter=int(args.rigid_max_iter),
        trim_quantile=float(args.rigid_trim_quantile),
        huber_delta=float(args.rigid_huber_delta),
        max_step_deg=float(args.rigid_max_step_deg),
        max_total_deg=float(args.rigid_max_total_deg),
        reg=float(args.rigid_reg),
    )
    r_rigid = np.asarray(rigid_best["rotation"], dtype=np.float64)
    t_rigid = np.asarray(rigid_best["translation"], dtype=np.float64)
    t_rigid_4x4 = rigid_to_transform(r_rigid, t_rigid)
    t_global_4x4 = rigid_to_transform(r_global, np.zeros((3,), dtype=np.float64))
    t_pre_4x4 = t_rigid_4x4 @ t_global_4x4
    src_pts_rigid, _src_normals_rigid = apply_rigid(
        src_pts_global, src_normals_global, r=r_rigid, t=t_rigid
    )
    # Axis frame is attached to source shape, so update it after rigid pre-stage.
    a_global = r_global @ a
    c_global = r_global @ c
    a_rigid = r_rigid @ a_global
    c_rigid = r_rigid @ c_global + t_rigid

    twist_search = {
        "enabled": int(args.enable_long_axis_twist_search),
        "deg_max": float(args.twist_deg_max),
        "deg_step": float(args.twist_deg_step),
        "probe_iters": int(args.twist_probe_iters),
        "tested": [],
        "best_theta_deg": 0.0,
    }

    long_axis_world = a_rigid[:, int(np.clip(long_axis_index, 0, 2))]
    theta_list = [0.0]
    if int(args.enable_long_axis_twist_search) == 1:
        tmax = max(float(args.twist_deg_max), 0.0)
        tstep = max(float(args.twist_deg_step), 1e-3)
        theta_arr = np.arange(-tmax, tmax + 1e-6, tstep, dtype=np.float64)
        theta_list = [float(x) for x in theta_arr]
        if 0.0 not in theta_list:
            theta_list.append(0.0)
        theta_list = sorted(set(theta_list))

    best = None
    a_best = a_rigid
    best_probe_primary = np.inf
    for theta_deg in theta_list:
        r_tw = rotation_about_axis(long_axis_world, np.deg2rad(theta_deg))
        a_tw = r_tw @ a_rigid
        c_tw = c_rigid.copy()
        probe_iters = int(args.max_iter)
        if int(args.enable_long_axis_twist_search) == 1 and len(theta_list) > 1:
            probe_iters = int(args.twist_probe_iters)
        cur = optimize_axis_affine_partial_aware(
            src_pts=src_pts_rigid,
            dst_seen_pts=dst_seen_pts,
            a=a_tw,
            c=c_tw,
            max_iter=probe_iters,
            trim_quantile=float(args.trim_quantile),
            huber_delta=float(args.huber_delta),
            scale_reg=float(args.scale_reg),
            scale_clamp_min=float(args.scale_clamp_min),
            scale_clamp_max=float(args.scale_clamp_max),
            secondary_weight=float(args.secondary_weight),
            secondary_shell_quantile=float(args.secondary_shell_quantile),
            affine_axis_mode=affine_axis_mode,
            single_axis_index=single_axis_index,
        )
        twist_search["tested"].append(
            {
                "theta_deg": float(theta_deg),
                "objective_total": float(cur["objective"]),
                "objective_primary": float(cur["objective_primary"]),
                "objective_secondary": float(cur["objective_secondary"]),
                "inliers": int(cur["inliers"]),
            }
        )
        if (best is None) or (float(cur["objective_primary"]) < float(best_probe_primary)):
            best = cur
            a_best = a_tw
            twist_search["best_theta_deg"] = float(theta_deg)
            best_probe_primary = float(cur["objective_primary"])

    assert best is not None
    if int(args.enable_long_axis_twist_search) == 1 and len(theta_list) > 1:
        # Final full-iteration refinement from best twist orientation.
        best_final = optimize_axis_affine_partial_aware(
            src_pts=src_pts_rigid,
            dst_seen_pts=dst_seen_pts,
            a=a_best,
            c=c_rigid,
            max_iter=int(args.max_iter),
            trim_quantile=float(args.trim_quantile),
            huber_delta=float(args.huber_delta),
            scale_reg=float(args.scale_reg),
            scale_clamp_min=float(args.scale_clamp_min),
            scale_clamp_max=float(args.scale_clamp_max),
            secondary_weight=float(args.secondary_weight),
            secondary_shell_quantile=float(args.secondary_shell_quantile),
            affine_axis_mode=affine_axis_mode,
            single_axis_index=single_axis_index,
        )
        if float(best_final["objective_primary"]) <= float(best_probe_primary):
            best = best_final
    scale_best = np.asarray(best["scale"], dtype=np.float64)
    t_axis_best = np.asarray(best["t_axis"], dtype=np.float64)

    axis_delta_transform = build_delta_transform(
        a=a_best, c=c_rigid, scale=scale_best, t_axis=t_axis_best
    )
    delta_transform = axis_delta_transform @ t_pre_4x4
    full_transform = delta_transform @ base_transform

    # Refined stats.
    src_pts_refined = transform_with_axis_model(
        src_pts_rigid, a=a_best, c=c_rigid, scale=scale_best, t_axis=t_axis_best
    )
    refined_tree = cKDTree(src_pts_refined)
    d_dst_to_src, _ = refined_tree.query(dst_seen_pts, k=1)
    refined_dst_to_src = dist_stats(d_dst_to_src)

    dst_tree = cKDTree(dst_seen_pts)
    d_src_to_dst, _ = dst_tree.query(src_pts_refined, k=1)
    refined_src_to_dst = dist_stats(d_src_to_dst)
    shell_cut = np.quantile(d_src_to_dst, float(np.clip(args.secondary_shell_quantile, 0.05, 1.0)))
    shell_mask = d_src_to_dst <= shell_cut
    refined_src_shell_to_dst = (
        dist_stats(d_src_to_dst[shell_mask])
        if int(np.count_nonzero(shell_mask)) > 0
        else dist_stats(d_src_to_dst)
    )

    # Apply final delta transform to full source mesh.
    refined_mesh = src_mesh.copy()
    v = np.asarray(refined_mesh.vertices, dtype=np.float64)
    v2 = (delta_transform[:3, :3] @ v.T).T + delta_transform[:3, 3]
    refined_mesh.vertices = v2.astype(np.float32)
    # Avoid exporting texture sidecar files into potentially protected locations.
    refined_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=refined_mesh,
        vertex_colors=np.tile(
            np.array([[200, 200, 200, 255]], dtype=np.uint8), (len(refined_mesh.vertices), 1)
        ),
    )

    os.makedirs(out_mesh_path.parent, exist_ok=True)
    os.makedirs(delta_transform_path.parent, exist_ok=True)
    os.makedirs(full_transform_path.parent, exist_ok=True)
    os.makedirs(metrics_path.parent, exist_ok=True)
    refined_mesh.export(str(out_mesh_path))
    np.savetxt(str(delta_transform_path), delta_transform, fmt="%.10f")
    np.savetxt(str(full_transform_path), full_transform, fmt="%.10f")

    metrics = {
        "mode": "camera_seen_area_global_init_then_rigid_p2l_then_axis_scale_with_secondary_shell",
        "src": str(source_mesh_path),
        "dst": str(real_mesh_path),
        "out": str(out_mesh_path),
        "axis_json": str(axis_json_path),
        "base_transform_txt": str(base_transform_txt),
        "delta_transform_txt": str(delta_transform_path),
        "full_transform_txt": str(full_transform_path),
        "sample_points": int(args.sample_points),
        "max_iter": int(args.max_iter),
        "trim_quantile": float(args.trim_quantile),
        "huber_delta": float(args.huber_delta),
        "affine_axis": affine_axis_canonical,
        "affine_axis_mode": affine_axis_mode,
        "single_axis_name": str(affine_axis_policy.get("single_axis_name", "")),
        "shared_axes": list(affine_axis_policy.get("shared_axes", [])),
        "scale_clamp": [float(args.scale_clamp_min), float(args.scale_clamp_max)],
        "scale_reg": float(args.scale_reg),
        "secondary_weight": float(args.secondary_weight),
        "secondary_shell_quantile": float(args.secondary_shell_quantile),
        "long_axis_twist_search": twist_search,
        "global_init_search": {
            "enabled": int(args.enable_global_init_search),
            "yaw_step_deg": float(args.global_yaw_step_deg),
            "tilt_deg": float(args.global_tilt_deg),
            "eval_trim_quantile": float(args.global_eval_trim_quantile),
            "silhouette_weight": float(args.global_silhouette_weight),
            "silhouette_downscale": int(args.global_silhouette_downscale),
            "silhouette_frames": int(args.global_silhouette_frames),
            "num_candidates": int(global_init["num_candidates"]),
            "cost_trimmed_mean": float(global_init["cost_trimmed_mean"]),
            "cost_nn_trimmed_mean": float(global_init.get("cost_nn_trimmed_mean", global_init["cost_trimmed_mean"])),
            "silhouette_iou": float(global_init.get("silhouette_iou", 0.0)),
            "rotation": np.asarray(global_init["rotation"], dtype=np.float64).tolist(),
            "transform_4x4": t_global_4x4.tolist(),
        },
        "rigid_stage": {
            "max_iter": int(args.rigid_max_iter),
            "trim_quantile": float(args.rigid_trim_quantile),
            "huber_delta": float(args.rigid_huber_delta),
            "max_step_deg": float(args.rigid_max_step_deg),
            "max_total_deg": float(args.rigid_max_total_deg),
            "reg": float(args.rigid_reg),
            "best_iter": int(rigid_best["iter"]),
            "inliers": int(rigid_best["inliers"]),
            "objective_total": float(rigid_best["objective"]),
            "objective_p2l": float(rigid_best["objective_p2l"]),
            "rotation": np.asarray(rigid_best["rotation"], dtype=np.float64).tolist(),
            "translation": np.asarray(rigid_best["translation"], dtype=np.float64).tolist(),
            "rigid_transform_4x4": t_rigid_4x4.tolist(),
            "pre_transform_4x4": t_pre_4x4.tolist(),
        },
        "seen_estimation": seen_summary,
        "best_iter": int(best["iter"]),
        "inliers": int(best["inliers"]),
        "objective_total": float(best["objective"]),
        "objective_primary": float(best["objective_primary"]),
        "objective_secondary": float(best["objective_secondary"]),
        "secondary_pairs": int(best["secondary_pairs"]),
        "axis_scales": scale_best.tolist(),
        "translation_in_axis_frame": t_axis_best.tolist(),
        "axis_frame_final_for_scale": {
            "center_world": c_rigid.tolist(),
            "axis_dirs_world": a_best.T.tolist(),
        },
        "axis_delta_transform_4x4": axis_delta_transform.tolist(),
        "delta_affine_3x3": delta_transform[:3, :3].tolist(),
        "delta_translation": delta_transform[:3, 3].tolist(),
        "delta_transform_4x4": delta_transform.tolist(),
        "full_transform_4x4": full_transform.tolist(),
        "baseline_dst_seen_to_src": baseline_stats,
        "refined_dst_seen_to_src": refined_dst_to_src,
        "refined_src_to_dst_seen": refined_src_to_dst,
        "refined_src_shell_to_dst_seen": refined_src_shell_to_dst,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Refined mesh written: {out_mesh_path}")
    print(f"Delta transform written: {delta_transform_path}")
    print(f"Full transform written: {full_transform_path}")
    print(
        "Seen-area fit stats: "
        f"baseline_mean={baseline_stats['mean']:.6f}, "
        f"refined_mean={refined_dst_to_src['mean']:.6f}, "
        f"refined_p95={refined_dst_to_src['p95']:.6f}"
    )
    print(f"Metrics written: {metrics_path}")


if __name__ == "__main__":
    main()
