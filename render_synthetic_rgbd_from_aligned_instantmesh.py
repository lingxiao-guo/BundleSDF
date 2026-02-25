#!/usr/bin/env python3
"""
Render synthetic RGB-D from aligned InstantMesh using real camera intrinsics.
By default, real RGB-D/mask/pose are used to estimate seen vs unseen mesh areas,
then viewpoints are selected to prioritize unseen coverage and avoid seen areas.

Default outputs:
  - /home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/instantmesh/rgb
  - /home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/instantmesh/depth

Depth is saved as uint16 PNG in millimeters.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


DEFAULT_INSTANTMESH_DIR = (
    "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/instantmesh"
)
DEFAULT_MESH = f"{DEFAULT_INSTANTMESH_DIR}/mesh_aligned.obj"
DEFAULT_CAM_K = "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/cam_K.txt"
DEFAULT_RGB_DIR = f"{DEFAULT_INSTANTMESH_DIR}/rgb"
DEFAULT_DEPTH_DIR = f"{DEFAULT_INSTANTMESH_DIR}/depth"
DEFAULT_POSE_DIR = f"{DEFAULT_INSTANTMESH_DIR}/ob_in_cam"
DEFAULT_REAL_DEPTH_DIR = "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/depth"
DEFAULT_REAL_MASK_DIR = "/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/mask"
DEFAULT_REAL_POSE_DIR = "/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2/ob_in_cam"
DEFAULT_COVERAGE_REPORT = f"{DEFAULT_INSTANTMESH_DIR}/coverage_report.json"


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


def make_color_sampler(mesh: trimesh.Trimesh):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    base_color = np.array([180.0, 180.0, 180.0], dtype=np.float32)
    tri_uv = None
    tri_vcols = None

    uv = getattr(mesh.visual, "uv", None)
    if uv is not None:
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim == 2 and uv.shape[1] == 2 and len(uv) == len(mesh.vertices):
            tri_uv = uv[faces]

    tex_img = None
    material = getattr(mesh.visual, "material", None)
    if material is not None:
        img = getattr(material, "image", None)
        if img is not None:
            tex_img = np.asarray(img)
            if tex_img.ndim == 2:
                tex_img = np.repeat(tex_img[..., None], 3, axis=-1)
            if tex_img.shape[-1] == 4:
                tex_img = tex_img[..., :3]
            tex_img = tex_img.astype(np.float32)
        base = getattr(material, "baseColorFactor", None)
        if base is not None:
            base = np.asarray(base).reshape(-1)
            if base.size >= 3:
                base_color = base[:3].astype(np.float32)

    vcols = getattr(mesh.visual, "vertex_colors", None)
    if vcols is not None and len(vcols) == len(mesh.vertices):
        vcols = np.asarray(vcols, dtype=np.float32)
        if vcols.ndim == 2 and vcols.shape[1] >= 3:
            tri_vcols = vcols[:, :3][faces]

    def bilinear(image: np.ndarray, uv_query: np.ndarray):
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

    def sample(prim_ids: np.ndarray, bary_uv: np.ndarray):
        n = len(prim_ids)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        b1 = bary_uv[:, 0:1]
        b2 = bary_uv[:, 1:2]
        b0 = 1.0 - b1 - b2
        if tex_img is not None and tri_uv is not None:
            uv_hit = (
                b0 * tri_uv[prim_ids, 0]
                + b1 * tri_uv[prim_ids, 1]
                + b2 * tri_uv[prim_ids, 2]
            )
            return bilinear(tex_img, uv_hit).astype(np.float32)
        if tri_vcols is not None:
            return (
                b0 * tri_vcols[prim_ids, 0]
                + b1 * tri_vcols[prim_ids, 1]
                + b2 * tri_vcols[prim_ids, 2]
            ).astype(np.float32)
        return np.repeat(base_color[None, :], n, axis=0).astype(np.float32)

    return sample


def build_renderer(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return {
        "scene": scene,
        "sample_color": make_color_sampler(mesh),
        "center": mesh.bounds.mean(axis=0).astype(np.float32),
        "extents": mesh.extents.astype(np.float32),
    }


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
    cam_in_ob[:3, 3] = cam_pos.astype(np.float32)
    return np.linalg.inv(cam_in_ob)


def get_pixel_grid_dirs(k: np.ndarray, h: int, w: int):
    us = np.arange(w, dtype=np.int32)
    vs = np.arange(h, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1).astype(np.int32)
    v = vv.reshape(-1).astype(np.int32)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    dirs = np.stack(
        [
            (u.astype(np.float32) - cx) / fx,
            (v.astype(np.float32) - cy) / fy,
            np.ones_like(u, dtype=np.float32),
        ],
        axis=1,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return u, v, dirs.astype(np.float32)


def raycast_hits(
    renderer,
    ob_in_cam: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dirs_cam: np.ndarray,
    h: int,
    w: int,
):

    cam_in_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
    r = cam_in_ob[:3, :3]
    t = cam_in_ob[:3, 3]
    dirs_obj = (r @ dirs_cam.T).T
    dirs_obj /= np.linalg.norm(dirs_obj, axis=1, keepdims=True) + 1e-8
    origins_obj = np.repeat(t[None, :], len(dirs_obj), axis=0)
    rays = np.concatenate([origins_obj, dirs_obj], axis=1).astype(np.float32)

    ans = renderer["scene"].cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    )
    prim_ids = ans["primitive_ids"].numpy().reshape(-1).astype(np.int64)
    bary_uv = ans["primitive_uvs"].numpy().reshape(-1, 2).astype(np.float32)
    t_hit = ans["t_hit"].numpy().reshape(-1).astype(np.float32)
    invalid_u32 = np.iinfo(np.uint32).max
    valid = np.isfinite(t_hit) & (prim_ids >= 0) & (prim_ids != invalid_u32)

    prim_map = np.full((h, w), -1, dtype=np.int32)
    depth = np.zeros((h, w), dtype=np.float32)
    if not np.any(valid):
        return prim_map, depth, prim_ids, bary_uv, np.zeros((0,), dtype=np.int64)

    idx = np.where(valid)[0]
    hit_obj = origins_obj[idx] + dirs_obj[idx] * t_hit[idx, None]
    hit_cam = (ob_in_cam[:3, :3] @ hit_obj.T).T + ob_in_cam[:3, 3]
    z = hit_cam[:, 2]
    good = z > 1e-6
    if not np.any(good):
        return prim_map, depth, prim_ids, bary_uv, np.zeros((0,), dtype=np.int64)

    idx = idx[good]
    z = z[good]
    uu = u[idx]
    vv = v[idx]
    prim_map[vv, uu] = prim_ids[idx].astype(np.int32)
    depth[vv, uu] = z.astype(np.float32)
    return prim_map, depth, prim_ids, bary_uv, idx


def raycast_render(renderer, k: np.ndarray, h: int, w: int, ob_in_cam: np.ndarray):
    u, v, dirs_cam = get_pixel_grid_dirs(k, h, w)
    _prim_map, depth, prim_ids, bary_uv, idx = raycast_hits(
        renderer, ob_in_cam, u, v, dirs_cam, h, w
    )
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if len(idx) == 0:
        return rgb, depth

    uu = u[idx]
    vv = v[idx]
    rgb_vals = renderer["sample_color"](prim_ids[idx], bary_uv[idx])
    rgb[vv, uu] = np.clip(rgb_vals, 0, 255).astype(np.uint8)
    return rgb, depth


def infer_image_size(k: np.ndarray):
    cx, cy = float(k[0, 2]), float(k[1, 2])
    w = int(round(2.0 * cx + 1.0))
    h = int(round(2.0 * cy + 1.0))
    return max(w, 1), max(h, 1)


def infer_image_size_from_real(real_depth_dir: str, real_mask_dir: str):
    for d in [real_depth_dir, real_mask_dir]:
        if not d:
            continue
        pdir = Path(d)
        if not pdir.exists():
            continue
        files = sorted(pdir.glob("*.png"))
        if len(files) == 0:
            continue
        img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim >= 2:
            return int(img.shape[1]), int(img.shape[0])
    return None


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    kk = k.copy().astype(np.float32)
    kk[0, 0] *= sx
    kk[1, 1] *= sy
    kk[0, 2] *= sx
    kk[1, 2] *= sy
    return kk


def load_depth_m(path: str, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth: {path}")
    return depth.astype(np.float32) / max(depth_scale, 1e-8)


def load_mask(path: str, thresh: int) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to load mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > thresh).astype(np.uint8)


def collect_common_frame_ids(pose_dir: str, depth_dir: str, mask_dir: str):
    pose_ids = {p.stem for p in Path(pose_dir).glob("*.txt")}
    depth_ids = {p.stem for p in Path(depth_dir).glob("*.png")}
    mask_ids = {p.stem for p in Path(mask_dir).glob("*.png")}
    common = sorted(pose_ids & depth_ids & mask_ids)
    return common


def face_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    tris = np.asarray(mesh.triangles, dtype=np.float32)
    if len(tris) == 0:
        return np.zeros((0,), dtype=np.float32)
    v01 = tris[:, 1] - tris[:, 0]
    v02 = tris[:, 2] - tris[:, 0]
    return 0.5 * np.linalg.norm(np.cross(v01, v02), axis=1)


def confidence_to_weights(
    seen_conf: np.ndarray, seen_threshold: float, boundary_ratio: float
):
    thr = float(np.clip(seen_threshold, 0.0, 1.0))
    margin = float(np.clip(boundary_ratio, 0.0, 0.49))
    seen_w = np.zeros_like(seen_conf, dtype=np.float32)
    unseen_w = np.ones_like(seen_conf, dtype=np.float32)
    if margin <= 1e-8:
        seen_w = (seen_conf >= thr).astype(np.float32)
        unseen_w = 1.0 - seen_w
        return seen_w, unseen_w

    lo = thr - margin
    hi = thr + margin
    low = seen_conf <= lo
    high = seen_conf >= hi
    mid = (~low) & (~high)

    seen_w[high] = 1.0
    unseen_w[high] = 0.0
    seen_w[low] = 0.0
    unseen_w[low] = 1.0
    alpha = np.clip((seen_conf[mid] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    seen_w[mid] = alpha.astype(np.float32)
    unseen_w[mid] = (1.0 - alpha).astype(np.float32)
    return seen_w, unseen_w


def choose_candidate_views(
    cand_dirs: np.ndarray,
    cand_faces: list,
    face_area: np.ndarray,
    seen_w: np.ndarray,
    unseen_w: np.ndarray,
    num_renders: int,
    seen_penalty: float,
    min_unseen_ratio: float,
    min_view_angle_deg: float,
    coverage_decay: float,
):
    n = len(cand_dirs)
    if n == 0:
        return []
    used = np.zeros((n,), dtype=np.uint8)
    chosen = []
    chosen_dirs = []
    remaining_unseen = unseen_w.copy().astype(np.float32)
    min_cos = float(np.cos(np.deg2rad(max(min_view_angle_deg, 0.0))))

    for _ in range(max(int(num_renders), 0)):
        best_idx = -1
        best_gain = 0.0

        for strict in [True, False]:
            best_score = -1e30
            best_idx = -1
            best_gain = 0.0
            for i in range(n):
                if used[i]:
                    continue
                fids = cand_faces[i]
                if fids.size == 0:
                    continue
                if strict and chosen_dirs and min_view_angle_deg > 0:
                    dots = np.clip(np.dot(np.asarray(chosen_dirs), cand_dirs[i]), -1.0, 1.0)
                    if float(np.max(dots)) > min_cos:
                        continue

                areas = face_area[fids]
                unseen_gain = float(np.sum(areas * remaining_unseen[fids]))
                seen_hit = float(np.sum(areas * seen_w[fids]))
                total = unseen_gain + seen_hit
                unseen_ratio = unseen_gain / (total + 1e-8)
                if strict and unseen_ratio < float(min_unseen_ratio):
                    continue

                score = unseen_gain - float(seen_penalty) * seen_hit
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_gain = unseen_gain

            if best_idx >= 0:
                break

        if best_idx < 0:
            break
        if best_gain <= 1e-10 and len(chosen) > 0:
            break

        used[best_idx] = 1
        chosen.append(best_idx)
        chosen_dirs.append(cand_dirs[best_idx])
        fids = cand_faces[best_idx]
        decay = float(np.clip(coverage_decay, 0.0, 1.0))
        if decay >= 1.0:
            remaining_unseen[fids] = 0.0
        else:
            remaining_unseen[fids] *= (1.0 - decay)

    return chosen


def estimate_seen_unseen(
    renderer,
    mesh: trimesh.Trimesh,
    k: np.ndarray,
    w: int,
    h: int,
    args,
):
    frame_ids = collect_common_frame_ids(
        args.real_pose_dir, args.real_depth_dir, args.real_mask_dir
    )
    if int(args.frame_stride) > 1:
        frame_ids = frame_ids[:: int(args.frame_stride)]
    if int(args.max_real_frames) > 0:
        frame_ids = frame_ids[: int(args.max_real_frames)]
    if len(frame_ids) == 0:
        return None

    n_faces = len(mesh.faces)
    visible_votes = np.zeros((n_faces,), dtype=np.float32)
    seen_votes = np.zeros((n_faces,), dtype=np.float32)

    down = max(int(args.estimation_downscale), 1)
    w_est = max(1, w // down)
    h_est = max(1, h // down)
    sx = float(w_est) / float(w)
    sy = float(h_est) / float(h)
    k_est = scale_intrinsics(k, sx, sy)
    u_est, v_est, dirs_est = get_pixel_grid_dirs(k_est, h_est, w_est)

    for i, fid in enumerate(frame_ids):
        pose_path = os.path.join(args.real_pose_dir, f"{fid}.txt")
        depth_path = os.path.join(args.real_depth_dir, f"{fid}.png")
        mask_path = os.path.join(args.real_mask_dir, f"{fid}.png")

        ob_in_cam = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        mask = load_mask(mask_path, int(args.mask_threshold))
        depth_m = load_depth_m(depth_path, float(args.real_depth_scale))
        if mask.shape[0] != h_est or mask.shape[1] != w_est:
            mask = cv2.resize(mask, (w_est, h_est), interpolation=cv2.INTER_NEAREST)
        if depth_m.shape[0] != h_est or depth_m.shape[1] != w_est:
            depth_m = cv2.resize(depth_m, (w_est, h_est), interpolation=cv2.INTER_NEAREST)

        prim_map, depth_render, _prim_ids, _bary, _idx = raycast_hits(
            renderer, ob_in_cam, u_est, v_est, dirs_est, h_est, w_est
        )
        valid = prim_map >= 0
        if not np.any(valid):
            continue

        real_valid = depth_m > 1e-6
        obj_px = mask > 0
        considered = valid & real_valid & obj_px
        if not np.any(considered):
            continue

        f_visible = prim_map[considered].astype(np.int64)
        np.add.at(visible_votes, f_visible, 1.0)

        d_pred = depth_render[considered]
        d_real = depth_m[considered]
        tol = float(args.depth_abs_tol_m) + float(args.depth_rel_tol) * d_pred
        good = np.abs(d_real - d_pred) <= tol
        if np.any(good):
            f_seen = f_visible[good]
            np.add.at(seen_votes, f_seen, 1.0)

        if (i + 1) % 20 == 0 or (i + 1) == len(frame_ids):
            print(f"[coverage] processed {i + 1}/{len(frame_ids)} real frames")

    seen_conf = np.zeros((n_faces,), dtype=np.float32)
    nonzero = visible_votes > 0
    seen_conf[nonzero] = seen_votes[nonzero] / np.maximum(visible_votes[nonzero], 1e-8)
    seen_w, unseen_w = confidence_to_weights(
        seen_conf, float(args.seen_threshold), float(args.boundary_ratio)
    )
    f_area = face_areas(mesh)

    total_area = float(np.sum(f_area)) + 1e-8
    seen_area = float(np.sum(f_area * seen_w))
    unseen_area = float(np.sum(f_area * unseen_w))

    summary = {
        "num_real_frames": int(len(frame_ids)),
        "num_faces": int(n_faces),
        "total_face_area": total_area,
        "seen_area_weighted": seen_area,
        "unseen_area_weighted": unseen_area,
        "seen_area_ratio": seen_area / total_area,
        "unseen_area_ratio": unseen_area / total_area,
        "seen_threshold": float(args.seen_threshold),
        "boundary_ratio": float(args.boundary_ratio),
        "depth_abs_tol_m": float(args.depth_abs_tol_m),
        "depth_rel_tol": float(args.depth_rel_tol),
        "estimation_downscale": int(args.estimation_downscale),
        "frame_stride": int(args.frame_stride),
        "max_real_frames": int(args.max_real_frames),
        "median_seen_conf": float(np.median(seen_conf)),
    }
    return {
        "seen_conf": seen_conf,
        "seen_w": seen_w,
        "unseen_w": unseen_w,
        "face_area": f_area,
        "summary": summary,
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Render synthetic RGB-D from aligned InstantMesh with real cam_K."
    )
    ap.add_argument("--mesh", type=str, default=DEFAULT_MESH)
    ap.add_argument("--cam_k", type=str, default=DEFAULT_CAM_K)
    ap.add_argument("--rgb_dir", type=str, default=DEFAULT_RGB_DIR)
    ap.add_argument("--depth_dir", type=str, default=DEFAULT_DEPTH_DIR)
    ap.add_argument(
        "--pose_dir",
        type=str,
        default=DEFAULT_POSE_DIR,
        help="Directory to save object poses in camera frame (ob_in_cam/*.txt).",
    )
    ap.add_argument("--num_renders", type=int, default=24)
    ap.add_argument(
        "--width",
        type=int,
        default=0,
        help="Override image width. Default uses real depth/mask size if available, else cam_K.",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=0,
        help="Override image height. Default uses real depth/mask size if available, else cam_K.",
    )
    ap.add_argument(
        "--distance_scale",
        type=float,
        default=1.2,
        help="Scale factor on minimum camera distance that fits object in FoV.",
    )
    ap.add_argument("--real_depth_dir", type=str, default=DEFAULT_REAL_DEPTH_DIR)
    ap.add_argument("--real_mask_dir", type=str, default=DEFAULT_REAL_MASK_DIR)
    ap.add_argument("--real_pose_dir", type=str, default=DEFAULT_REAL_POSE_DIR)
    ap.add_argument(
        "--real_depth_scale",
        type=float,
        default=1000.0,
        help="Depth unit scale for real depth PNGs (1000 => mm to meters).",
    )
    ap.add_argument(
        "--frame_stride",
        type=int,
        default=8,
        help="Use every N-th real frame for seen/unseen estimation.",
    )
    ap.add_argument(
        "--max_real_frames",
        type=int,
        default=180,
        help="Max number of real frames used for estimation. 0 means all.",
    )
    ap.add_argument(
        "--estimation_downscale",
        type=int,
        default=2,
        help="Downscale factor for coverage estimation raycasting.",
    )
    ap.add_argument("--mask_threshold", type=int, default=0)
    ap.add_argument(
        "--depth_abs_tol_m",
        type=float,
        default=0.02,
        help="Absolute depth consistency tolerance in meters.",
    )
    ap.add_argument(
        "--depth_rel_tol",
        type=float,
        default=0.03,
        help="Relative depth consistency tolerance.",
    )
    ap.add_argument(
        "--seen_threshold",
        type=float,
        default=0.55,
        help="Face seen-confidence threshold in [0,1].",
    )
    ap.add_argument(
        "--boundary_ratio",
        type=float,
        default=0.12,
        help="Uncertain band half-width around seen threshold (confidence boundary).",
    )
    ap.add_argument(
        "--num_candidates",
        type=int,
        default=0,
        help="Number of candidate views before selecting num_renders. 0 means auto.",
    )
    ap.add_argument(
        "--seen_penalty",
        type=float,
        default=1.2,
        help="Penalty weight when a candidate view overlaps seen area.",
    )
    ap.add_argument(
        "--min_unseen_ratio",
        type=float,
        default=0.55,
        help="Preferred min unseen ratio per selected view.",
    )
    ap.add_argument(
        "--min_view_angle_deg",
        type=float,
        default=12.0,
        help="Discourage selecting nearly identical viewpoints.",
    )
    ap.add_argument(
        "--coverage_decay",
        type=float,
        default=1.0,
        help="How strongly selected views consume unseen area [0,1].",
    )
    ap.add_argument(
        "--coverage_report",
        type=str,
        default=DEFAULT_COVERAGE_REPORT,
        help="JSON file to save seen/unseen summary and selected-view stats.",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    mesh_path = str(Path(args.mesh).expanduser().resolve())
    cam_k_path = str(Path(args.cam_k).expanduser().resolve())
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    if not os.path.exists(cam_k_path):
        raise FileNotFoundError(cam_k_path)

    k = np.loadtxt(cam_k_path).reshape(3, 3).astype(np.float32)
    w0, h0 = infer_image_size(k)
    inferred_real_size = infer_image_size_from_real(
        args.real_depth_dir, args.real_mask_dir
    )
    if inferred_real_size is not None:
        rw, rh = inferred_real_size
    else:
        rw, rh = w0, h0
    w = int(args.width) if int(args.width) > 0 else rw
    h = int(args.height) if int(args.height) > 0 else rh

    rgb_dir = str(Path(args.rgb_dir).expanduser().resolve())
    depth_dir = str(Path(args.depth_dir).expanduser().resolve())
    pose_dir = str(Path(args.pose_dir).expanduser().resolve())
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    mesh = load_mesh(mesh_path)
    renderer = build_renderer(mesh)
    center = renderer["center"].astype(np.float32)
    extents = renderer["extents"].astype(np.float32)
    r_obj = 0.5 * float(np.linalg.norm(extents))

    fx, fy = float(k[0, 0]), float(k[1, 1])
    fov_x = 2.0 * np.arctan(float(w) / (2.0 * fx))
    fov_y = 2.0 * np.arctan(float(h) / (2.0 * fy))
    min_fov = float(min(fov_x, fov_y))
    min_dist = r_obj / max(np.sin(0.5 * min_fov), 1e-3)
    cam_dist = max(0.15, float(args.distance_scale) * min_dist)

    if int(args.num_candidates) > 0:
        n_candidates = int(args.num_candidates)
    else:
        n_candidates = max(256, int(args.num_renders) * 32)
    cand_dirs = fibonacci_sphere(n_candidates)

    coverage = estimate_seen_unseen(
        renderer=renderer,
        mesh=mesh,
        k=k,
        w=w,
        h=h,
        args=args,
    )

    if coverage is None:
        print(
            "No overlapping real pose/depth/mask frames found. "
            "Fallback to uniform sampling."
        )
        selected_idx = list(range(min(int(args.num_renders), len(cand_dirs))))
    else:
        print(
            f"Coverage estimate: seen_ratio={coverage['summary']['seen_area_ratio']:.4f}, "
            f"unseen_ratio={coverage['summary']['unseen_area_ratio']:.4f}"
        )
        est_w = max(1, w // max(1, int(args.estimation_downscale)))
        est_h = max(1, h // max(1, int(args.estimation_downscale)))
        k_est = scale_intrinsics(k, float(est_w) / float(w), float(est_h) / float(h))
        u_est, v_est, dirs_est = get_pixel_grid_dirs(k_est, est_h, est_w)

        cand_faces = []
        for i, d in enumerate(cand_dirs):
            cam_pos = center + d * cam_dist
            ob_in_cam = look_at_ob_in_cam(cam_pos, center)
            prim_map, _depth, _prim, _bary, _idx = raycast_hits(
                renderer, ob_in_cam, u_est, v_est, dirs_est, est_h, est_w
            )
            fids = np.unique(prim_map[prim_map >= 0]).astype(np.int64)
            cand_faces.append(fids)
            if (i + 1) % 64 == 0 or (i + 1) == len(cand_dirs):
                print(f"[candidate] built {i + 1}/{len(cand_dirs)} views")

        selected_idx = choose_candidate_views(
            cand_dirs=cand_dirs,
            cand_faces=cand_faces,
            face_area=coverage["face_area"],
            seen_w=coverage["seen_w"],
            unseen_w=coverage["unseen_w"],
            num_renders=int(args.num_renders),
            seen_penalty=float(args.seen_penalty),
            min_unseen_ratio=float(args.min_unseen_ratio),
            min_view_angle_deg=float(args.min_view_angle_deg),
            coverage_decay=float(args.coverage_decay),
        )
        if len(selected_idx) < int(args.num_renders):
            for i in range(len(cand_dirs)):
                if i not in selected_idx:
                    selected_idx.append(i)
                if len(selected_idx) >= int(args.num_renders):
                    break

    dirs = cand_dirs[selected_idx]
    print(
        f"Rendering {len(dirs)} selected views | image={w}x{h} | "
        f"distance={cam_dist:.4f}m | mesh={mesh_path}"
    )

    for i, d in enumerate(dirs):
        cam_pos = center + d * cam_dist
        ob_in_cam = look_at_ob_in_cam(cam_pos, center)
        rgb, depth_m = raycast_render(renderer, k, h, w, ob_in_cam)

        name = f"{i:06d}.png"
        cv2.imwrite(os.path.join(rgb_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, name), depth_mm)
        np.savetxt(
            os.path.join(pose_dir, f"{i:06d}.txt"),
            ob_in_cam.reshape(4, 4),
            fmt="%.8f",
        )

    if coverage is not None:
        report = {
            "coverage": coverage["summary"],
            "selection": {
                "num_candidates": int(len(cand_dirs)),
                "num_selected": int(len(selected_idx)),
                "selected_indices": [int(x) for x in selected_idx],
                "seen_penalty": float(args.seen_penalty),
                "min_unseen_ratio": float(args.min_unseen_ratio),
                "min_view_angle_deg": float(args.min_view_angle_deg),
                "coverage_decay": float(args.coverage_decay),
            },
        }
        report_path = str(Path(args.coverage_report).expanduser().resolve())
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved coverage report to: {report_path}")

    print(f"Saved RGB to: {rgb_dir}")
    print(f"Saved depth(mm) to: {depth_dir}")
    print(f"Saved ob_in_cam poses to: {pose_dir}")


if __name__ == "__main__":
    main()
