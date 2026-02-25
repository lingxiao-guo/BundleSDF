#!/usr/bin/env python3
"""
Complete a partial BundleSDF mesh using a full SAM3D prior mesh.

Pipeline:
1) Similarity-align prior mesh to the partial reconstructed mesh.
2) Use aligned prior geometry as completed shape.
3) Project real RGB-D observations into prior UV atlas.
4) Fill unseen UV texels from synthetic views rendered from the prior mesh.
5) Export completed textured mesh (OBJ/GLB) and debug artifacts.

This script is intended to run in an environment with:
  - trimesh
  - open3d
  - opencv-python
  - scipy
"""

import argparse
import glob
import itertools
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
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


def ensure_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise RuntimeError("Prior mesh has no UV coordinates.")
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise RuntimeError("Prior mesh UV format is invalid.")
    return uv


def make_prior_texture_sampler(mesh: trimesh.Trimesh):
    uv = ensure_uv(mesh)
    _ = uv  # keep validation side effect

    tex_img = None
    base_color = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    material_info: Dict[str, object] = {"has_texture_image": False}

    visual = mesh.visual
    material = getattr(visual, "material", None)
    if material is not None:
        base = getattr(material, "baseColorFactor", None)
        if base is not None:
            base = np.asarray(base).astype(np.float32).reshape(-1)
            if base.size >= 3:
                base_color = base[:3]
        img = getattr(material, "image", None)
        if img is not None:
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            tex_img = arr.astype(np.float32)
            material_info["has_texture_image"] = True

    def bilinear_sample(image: np.ndarray, uv_query: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        u = np.clip(uv_query[:, 0], 0.0, 1.0)
        v = np.clip(uv_query[:, 1], 0.0, 1.0)
        x = u * (w - 1)
        y = (1.0 - v) * (h - 1)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wx = x - x0
        wy = y - y0
        c00 = image[y0, x0]
        c01 = image[y0, x1]
        c10 = image[y1, x0]
        c11 = image[y1, x1]
        c0 = c00 * (1.0 - wx[:, None]) + c01 * wx[:, None]
        c1 = c10 * (1.0 - wx[:, None]) + c11 * wx[:, None]
        return c0 * (1.0 - wy[:, None]) + c1 * wy[:, None]

    def sample(uv_query: np.ndarray) -> np.ndarray:
        if len(uv_query) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if tex_img is None:
            return np.repeat(base_color[None, :], len(uv_query), axis=0)
        return bilinear_sample(tex_img, uv_query)

    return sample, material_info


def fibonacci_sphere(samples: int) -> np.ndarray:
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = 1.0 - (2.0 * i) / max(samples - 1, 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def look_at_ob_in_cam(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - cam_pos
    z /= np.linalg.norm(z) + 1e-8
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(z, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(z, up)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    y /= np.linalg.norm(y) + 1e-8

    cam_in_ob = np.eye(4, dtype=np.float32)
    cam_in_ob[:3, 0] = x
    cam_in_ob[:3, 1] = y
    cam_in_ob[:3, 2] = z
    cam_in_ob[:3, 3] = cam_pos
    return np.linalg.inv(cam_in_ob)


def candidate_axis_rotations() -> List[np.ndarray]:
    rots = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            r = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                r[i, perm[i]] = signs[i]
            if np.linalg.det(r) > 0.0:
                rots.append(r)
    return rots


def sample_mesh_points(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float64)


def similarity_align_prior_to_partial(
    prior_mesh: trimesh.Trimesh,
    partial_mesh: trimesh.Trimesh,
    sample_points: int = 30000,
    icp_iters: int = 50,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    prior_pts = sample_mesh_points(prior_mesh, sample_points)
    partial_pts = sample_mesh_points(partial_mesh, sample_points)

    prior_ext = np.maximum(prior_mesh.extents.astype(np.float64), 1e-6)
    partial_ext = np.maximum(partial_mesh.extents.astype(np.float64), 1e-6)
    scale = float(np.median(partial_ext / prior_ext))

    partial_tree = cKDTree(partial_pts)
    partial_center = partial_pts.mean(axis=0)
    best_err = np.inf
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)

    prior_pts_scaled = prior_pts * scale
    prior_center_scaled = prior_pts_scaled.mean(axis=0)
    for r in candidate_axis_rotations():
        t = partial_center - r @ prior_center_scaled
        moved = (r @ prior_pts_scaled.T).T + t
        dists, _ = partial_tree.query(moved, k=1)
        err = float(dists.mean())
        if err < best_err:
            best_err = err
            best_r = r
            best_t = t

    src = (best_r @ prior_pts_scaled.T).T + best_t
    tgt = partial_pts
    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))

    max_corresp = float(np.linalg.norm(partial_ext) * 0.08)
    reg = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        max_corresp,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iters),
    )
    t_icp = reg.transformation.astype(np.float64)

    aligned = prior_mesh.copy()
    v = aligned.vertices.astype(np.float64) * scale
    v = (best_r @ v.T).T + best_t
    v = (t_icp[:3, :3] @ v.T).T + t_icp[:3, 3]
    aligned.vertices = v.astype(np.float32)

    aligned_pts = sample_mesh_points(aligned, sample_points)
    dists, _ = partial_tree.query(aligned_pts, k=1)
    rmse = float(np.sqrt((dists**2).mean()))

    info = {
        "scale": scale,
        "rotation_init": best_r.tolist(),
        "translation_init": best_t.tolist(),
        "icp_transform": t_icp.tolist(),
        "init_mean_nn_error": best_err,
        "final_rmse_to_partial": rmse,
    }
    return aligned, info


def make_raycast_scene(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def build_pixel_grid_and_dirs(
    k: np.ndarray, h: int, w: int, pixel_stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    us = np.arange(0, w, pixel_stride, dtype=np.float32)
    vs = np.arange(0, h, pixel_stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)
    v = vv.reshape(-1)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs = dirs / norms
    return u.astype(np.int32), v.astype(np.int32), dirs.astype(np.float32)


def transform_rays_to_object(
    dirs_cam: np.ndarray, ob_in_cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cam_in_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
    r = cam_in_ob[:3, :3]
    t = cam_in_ob[:3, 3]
    dirs_obj = (r @ dirs_cam.T).T
    dirs_obj /= np.linalg.norm(dirs_obj, axis=1, keepdims=True) + 1e-8
    origins_obj = np.repeat(t[None, :], len(dirs_obj), axis=0)
    return origins_obj, dirs_obj


def cast_rays(
    scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, dirs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    prim_ids = ans["primitive_ids"].numpy().reshape(-1)
    bary_uv = ans["primitive_uvs"].numpy().reshape(-1, 2).astype(np.float32)
    t_hit = ans["t_hit"].numpy().reshape(-1).astype(np.float32)

    prim_i64 = prim_ids.astype(np.int64, copy=False)
    invalid_u32 = np.iinfo(np.uint32).max
    valid = np.isfinite(t_hit) & (prim_i64 >= 0) & (prim_i64 != invalid_u32)
    hit_points = origins + dirs * t_hit[:, None]
    return prim_i64, bary_uv, t_hit, hit_points, valid


def bary_to_uv(tri_uv: np.ndarray, prim_ids: np.ndarray, bary_uv: np.ndarray) -> np.ndarray:
    uv_tri = tri_uv[prim_ids]  # (N,3,2)
    b1 = bary_uv[:, 0:1]
    b2 = bary_uv[:, 1:2]
    b0 = 1.0 - b1 - b2
    return b0 * uv_tri[:, 0] + b1 * uv_tri[:, 1] + b2 * uv_tri[:, 2]


def accumulate_to_uv(
    tex_sum: np.ndarray,
    tex_w: np.ndarray,
    uv: np.ndarray,
    rgb: np.ndarray,
    w: np.ndarray,
    only_missing: bool = False,
) -> int:
    if len(uv) == 0:
        return 0
    h, w_tex = tex_w.shape
    x = np.clip(np.round(uv[:, 0] * (w_tex - 1)).astype(np.int32), 0, w_tex - 1)
    y = np.clip(np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    valid = np.isfinite(rgb).all(axis=1) & np.isfinite(w) & (w > 0)
    if only_missing:
        valid &= tex_w[y, x] <= 1e-8
    if not np.any(valid):
        return 0
    x = x[valid]
    y = y[valid]
    rgb = rgb[valid].astype(np.float32)
    ww = w[valid].astype(np.float32)

    np.add.at(tex_w, (y, x), ww)
    for c in range(3):
        np.add.at(tex_sum[..., c], (y, x), rgb[:, c] * ww)
    return int(valid.sum())


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, -1)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[..., None], 3, axis=-1)
    if bgr.shape[-1] == 4:
        bgr = bgr[..., :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_m(path: str) -> np.ndarray:
    d = cv2.imread(path, -1)
    if d is None:
        raise FileNotFoundError(path)
    d = d.astype(np.float32)
    if d.max() > 20.0:
        d = d / 1000.0
    return d


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, -1)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = (m.sum(axis=-1) > 0).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    return m


def collect_common_frame_ids(
    rgb_dir: str, depth_dir: str, mask_dir: str, pose_dir: str
) -> List[str]:
    def stems(pattern: str) -> set:
        out = set()
        for p in glob.glob(pattern):
            name = os.path.basename(p)
            stem, _ = os.path.splitext(name)
            out.add(stem)
        return out

    ids = (
        stems(os.path.join(rgb_dir, "*.png"))
        & stems(os.path.join(depth_dir, "*.png"))
        & stems(os.path.join(mask_dir, "*.png"))
        & stems(os.path.join(pose_dir, "*.txt"))
    )
    return sorted(ids)


def project_real_frames_to_uv(
    frame_ids: List[str],
    rgb_dir: str,
    depth_dir: str,
    mask_dir: str,
    pose_dir: str,
    k: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene,
    tri_uv: np.ndarray,
    face_normals: np.ndarray,
    tex_sum: np.ndarray,
    tex_w: np.ndarray,
    pixel_stride: int,
    depth_tol_abs: float,
    depth_tol_rel: float,
) -> int:
    n_written = 0
    u_idx: Optional[np.ndarray] = None
    v_idx: Optional[np.ndarray] = None
    dirs_cam: Optional[np.ndarray] = None

    for i, fid in enumerate(frame_ids):
        rgb = read_rgb(os.path.join(rgb_dir, f"{fid}.png"))
        depth = read_depth_m(os.path.join(depth_dir, f"{fid}.png"))
        mask = read_mask(os.path.join(mask_dir, f"{fid}.png"))
        ob_in_cam = np.loadtxt(os.path.join(pose_dir, f"{fid}.txt")).reshape(4, 4)

        h, w = depth.shape[:2]
        if u_idx is None:
            u_idx, v_idx, dirs_cam = build_pixel_grid_and_dirs(k, h, w, pixel_stride)
        assert u_idx is not None and v_idx is not None and dirs_cam is not None

        origins_obj, dirs_obj = transform_rays_to_object(dirs_cam, ob_in_cam)
        prim_ids, bary_uv, _, hit_obj, hit_valid = cast_rays(scene, origins_obj, dirs_obj)
        if not np.any(hit_valid):
            continue

        sampled_depth = depth[v_idx, u_idx]
        sampled_mask = mask[v_idx, u_idx] > 0
        depth_ok = sampled_depth > 1e-6

        pred_depth = np.full_like(sampled_depth, np.inf, dtype=np.float32)
        hit_valid_idx = np.where(hit_valid)[0]
        hit_cam_valid = (ob_in_cam[:3, :3] @ hit_obj[hit_valid_idx].T).T + ob_in_cam[:3, 3]
        pred_depth[hit_valid_idx] = hit_cam_valid[:, 2]
        tol = np.maximum(depth_tol_abs, depth_tol_rel * sampled_depth)
        depth_consistent = np.abs(pred_depth - sampled_depth) <= tol

        valid = hit_valid & sampled_mask & depth_ok & depth_consistent & (pred_depth > 1e-6)
        if not np.any(valid):
            continue

        vid = np.where(valid)[0]
        pid = prim_ids[vid]
        uv = bary_to_uv(tri_uv, pid, bary_uv[vid])
        rgb_obs = rgb[v_idx[vid], u_idx[vid]].astype(np.float32)

        normals = face_normals[pid].astype(np.float32)
        view = origins_obj[vid] - hit_obj[vid]
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-8
        cos_view = np.clip(np.sum(normals * view, axis=1), 0.0, 1.0)
        depth_err = np.abs(pred_depth[vid] - sampled_depth[vid])
        ww = cos_view * np.exp(-depth_err / (tol[vid] + 1e-8))

        n_written += accumulate_to_uv(tex_sum, tex_w, uv, rgb_obs, ww, only_missing=False)
        if (i + 1) % 50 == 0:
            print(f"[real] processed {i + 1}/{len(frame_ids)} frames")

    return n_written


def fill_missing_from_synthetic_prior_views(
    aligned_prior: trimesh.Trimesh,
    k: np.ndarray,
    h: int,
    w: int,
    scene: o3d.t.geometry.RaycastingScene,
    tri_uv: np.ndarray,
    face_normals: np.ndarray,
    prior_sampler,
    tex_sum: np.ndarray,
    tex_w: np.ndarray,
    n_views: int,
    pixel_stride: int,
) -> int:
    center = aligned_prior.bounds.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(aligned_prior.extents)) * 1.2
    dirs = fibonacci_sphere(n_views)

    u_idx, v_idx, dirs_cam = build_pixel_grid_and_dirs(k, h, w, pixel_stride)
    total = 0

    for i, d in enumerate(dirs):
        if not np.any(tex_w <= 1e-8):
            break
        cam_pos = center + d * radius
        ob_in_cam = look_at_ob_in_cam(cam_pos, center)
        origins_obj, dirs_obj = transform_rays_to_object(dirs_cam, ob_in_cam)
        prim_ids, bary_uv, _, hit_obj, hit_valid = cast_rays(scene, origins_obj, dirs_obj)
        if not np.any(hit_valid):
            continue

        vid = np.where(hit_valid)[0]
        pid = prim_ids[vid]
        uv = bary_to_uv(tri_uv, pid, bary_uv[vid])
        rgb_prior = prior_sampler(uv).astype(np.float32)

        normals = face_normals[pid].astype(np.float32)
        view = origins_obj[vid] - hit_obj[vid]
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-8
        ww = np.clip(np.sum(normals * view, axis=1), 0.0, 1.0)

        written = accumulate_to_uv(
            tex_sum, tex_w, uv, rgb_prior, ww, only_missing=True
        )
        total += written
        if (i + 1) % 6 == 0:
            remain = int((tex_w <= 1e-8).sum())
            print(f"[synthetic] views {i + 1}/{n_views}, remaining missing texels: {remain}")
    return total


def compose_texture(
    tex_sum: np.ndarray, tex_w: np.ndarray, prior_sampler
) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = tex_w.shape
    tex = np.zeros((h, w, 3), dtype=np.float32)
    seen = tex_w > 1e-8
    tex[seen] = tex_sum[seen] / tex_w[seen, None]

    missing = ~seen
    if np.any(missing):
        yy, xx = np.where(missing)
        uv = np.stack(
            [xx.astype(np.float32) / max(w - 1, 1), 1.0 - yy.astype(np.float32) / max(h - 1, 1)],
            axis=1,
        )
        tex[yy, xx] = prior_sampler(uv)

    coverage = float(seen.mean())
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return tex, missing.astype(np.uint8), coverage


def attach_texture_and_export(
    mesh: trimesh.Trimesh, texture: np.ndarray, out_obj: str, out_glb: str
):
    uv = ensure_uv(mesh)
    textured = mesh.copy()
    textured.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture)
    textured.export(out_obj)
    textured.export(out_glb)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Complete partial mesh with SAM3D prior (shape + texture)."
    )
    ap.add_argument(
        "--partial_mesh",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/textured_mesh.obj",
    )
    ap.add_argument(
        "--prior_mesh",
        default="/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/sam3d/object_0.glb",
    )
    ap.add_argument(
        "--rgb_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/color_segmented",
    )
    ap.add_argument(
        "--depth_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/depth_filtered",
    )
    ap.add_argument(
        "--mask_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/mask",
    )
    ap.add_argument(
        "--pose_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/ob_in_cam",
    )
    ap.add_argument(
        "--cam_k",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/cam_K.txt",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/completion_sam3d",
    )
    ap.add_argument("--tex_res", type=int, default=2048)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--pixel_stride_real", type=int, default=2)
    ap.add_argument("--pixel_stride_synth", type=int, default=3)
    ap.add_argument("--depth_tol_abs", type=float, default=0.01)
    ap.add_argument("--depth_tol_rel", type=float, default=0.03)
    ap.add_argument("--sample_points", type=int, default=30000)
    ap.add_argument("--icp_iters", type=int, default=50)
    ap.add_argument("--synthetic_views", type=int, default=24)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading meshes...")
    partial_mesh = load_mesh(args.partial_mesh)
    prior_mesh = load_mesh(args.prior_mesh)
    prior_sampler, prior_material_info = make_prior_texture_sampler(prior_mesh)

    print("Aligning prior mesh to partial mesh...")
    aligned_prior, align_info = similarity_align_prior_to_partial(
        prior_mesh, partial_mesh, sample_points=args.sample_points, icp_iters=args.icp_iters
    )
    aligned_prior.export(os.path.join(args.out_dir, "aligned_prior.obj"))
    aligned_prior.export(os.path.join(args.out_dir, "completed_shape.obj"))

    print("Preparing frame list...")
    frame_ids = collect_common_frame_ids(
        args.rgb_dir, args.depth_dir, args.mask_dir, args.pose_dir
    )
    frame_ids = frame_ids[:: max(args.frame_stride, 1)]
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if len(frame_ids) == 0:
        raise RuntimeError("No valid frame IDs found across rgb/depth/mask/pose.")
    print(f"Using {len(frame_ids)} real frames.")

    k = np.loadtxt(args.cam_k).reshape(3, 3).astype(np.float32)
    first_rgb = read_rgb(os.path.join(args.rgb_dir, f"{frame_ids[0]}.png"))
    h, w = first_rgb.shape[:2]

    scene = make_raycast_scene(aligned_prior)
    uv = ensure_uv(aligned_prior)
    tri_uv = uv[np.asarray(aligned_prior.faces, dtype=np.int64)]
    face_normals = np.asarray(aligned_prior.face_normals, dtype=np.float32)

    tex_sum = np.zeros((args.tex_res, args.tex_res, 3), dtype=np.float64)
    tex_w = np.zeros((args.tex_res, args.tex_res), dtype=np.float64)

    print("Projecting real RGB-D frames to UV...")
    n_real = project_real_frames_to_uv(
        frame_ids=frame_ids,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        pose_dir=args.pose_dir,
        k=k,
        scene=scene,
        tri_uv=tri_uv,
        face_normals=face_normals,
        tex_sum=tex_sum,
        tex_w=tex_w,
        pixel_stride=args.pixel_stride_real,
        depth_tol_abs=args.depth_tol_abs,
        depth_tol_rel=args.depth_tol_rel,
    )
    real_seen = float((tex_w > 1e-8).mean())
    print(f"Real projection wrote {n_real} UV samples; seen coverage={real_seen:.4f}")

    print("Filling unseen UV texels using synthetic prior views...")
    n_synth = fill_missing_from_synthetic_prior_views(
        aligned_prior=aligned_prior,
        k=k,
        h=h,
        w=w,
        scene=scene,
        tri_uv=tri_uv,
        face_normals=face_normals,
        prior_sampler=prior_sampler,
        tex_sum=tex_sum,
        tex_w=tex_w,
        n_views=args.synthetic_views,
        pixel_stride=args.pixel_stride_synth,
    )
    print(f"Synthetic prior fill wrote {n_synth} UV samples.")

    print("Composing final texture...")
    texture, missing_mask, final_real_coverage = compose_texture(
        tex_sum, tex_w, prior_sampler
    )
    imageio.imwrite(os.path.join(args.out_dir, "texture_completed.png"), texture)
    imageio.imwrite(
        os.path.join(args.out_dir, "texture_missing_mask.png"),
        (missing_mask * 255).astype(np.uint8),
    )

    print("Exporting textured meshes...")
    out_obj = os.path.join(args.out_dir, "completed_textured.obj")
    out_glb = os.path.join(args.out_dir, "completed_textured.glb")
    attach_texture_and_export(aligned_prior, texture, out_obj, out_glb)

    meta = {
        "partial_mesh": args.partial_mesh,
        "prior_mesh": args.prior_mesh,
        "n_real_frames": len(frame_ids),
        "n_real_uv_samples": n_real,
        "n_synthetic_uv_samples": n_synth,
        "real_seen_texel_ratio": real_seen,
        "final_seen_texel_ratio_before_fallback": final_real_coverage,
        "prior_material": prior_material_info,
        "alignment": align_info,
    }
    with open(os.path.join(args.out_dir, "completion_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
