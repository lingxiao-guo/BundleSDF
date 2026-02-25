#!/usr/bin/env python3
"""
Offline co-reconstruction with real + synthetic RGB-D.

Pipeline:
1) Read real RGB-D + tracked poses from an existing real-only BundleSDF run.
2) Render synthetic RGB-D from a registered prior mesh (targeting unseen views).
3) Reconstruct using CompletionBuffer + BundleSDF global NeRF with fixed real poses.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from typing import List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import trimesh
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from bundlesdf import BundleSdf  # noqa: E402
from completion_buffer import CompletionBuffer  # noqa: E402


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


def get_camera_dirs(k: np.ndarray, h: int, w: int):
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


def raycast_render(renderer, k: np.ndarray, h: int, w: int, ob_in_cam: np.ndarray):
    u, v, dirs_cam = get_camera_dirs(k, h, w)
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

    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if not np.any(valid):
        return color, depth, mask

    idx = np.where(valid)[0]
    hit_obj = origins_obj[idx] + dirs_obj[idx] * t_hit[idx, None]
    hit_cam = (ob_in_cam[:3, :3] @ hit_obj.T).T + ob_in_cam[:3, 3]
    z = hit_cam[:, 2]
    good = z > 1e-6
    if not np.any(good):
        return color, depth, mask

    idx = idx[good]
    z = z[good]
    uu = u[idx]
    vv = v[idx]
    mask[vv, uu] = 1
    depth[vv, uu] = z.astype(np.float32)
    rgb = renderer["sample_color"](prim_ids[idx], bary_uv[idx])
    color[vv, uu] = np.clip(rgb, 0, 255).astype(np.uint8)
    return color, depth, mask


def fibonacci_sphere(samples: int):
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


def collect_common_frame_ids(rgb_dir: str, depth_dir: str, mask_dir: str, pose_dir: str):
    def stems(pattern: str):
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


def read_rgb(path: str):
    bgr = cv2.imread(path, -1)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[..., None], 3, axis=-1)
    if bgr.shape[-1] == 4:
        bgr = bgr[..., :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_m(path: str):
    d = cv2.imread(path, -1)
    if d is None:
        raise FileNotFoundError(path)
    d = d.astype(np.float32)
    if d.max() > 20.0:
        d /= 1000.0
    return d


def read_mask(path: str):
    m = cv2.imread(path, -1)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        return (m.sum(axis=-1) > 0).astype(np.uint8)
    return (m > 0).astype(np.uint8)


def to_object_view_dirs(cam_in_obs: np.ndarray, center: np.ndarray):
    if len(cam_in_obs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cam_pos = cam_in_obs[:, :3, 3]
    dirs = cam_pos - center[None, :]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return (dirs / norms).astype(np.float32)


def select_unseen_dirs(
    real_cam_in_obs: np.ndarray,
    center: np.ndarray,
    n_views: int,
    candidate_count: int,
) -> np.ndarray:
    cand = fibonacci_sphere(max(candidate_count, n_views))
    real_dirs = to_object_view_dirs(real_cam_in_obs, center)
    if len(real_dirs) == 0:
        return cand[:n_views]

    sim_real = cand @ real_dirs.T
    max_sim_real = sim_real.max(axis=1)
    order = np.argsort(max_sim_real)  # low similarity to real first

    selected = []
    for idx in order:
        d = cand[idx]
        if len(selected) > 0:
            sims = np.dot(np.asarray(selected), d)
            if float(np.max(sims)) > 0.97:
                continue
        selected.append(d)
        if len(selected) >= n_views:
            break
    if len(selected) < n_views:
        for idx in order:
            selected.append(cand[idx])
            if len(selected) >= n_views:
                break
    return np.asarray(selected[:n_views], dtype=np.float32)


def prepare_configs(stage1_out: str, out_dir: str, freeze_real_poses: bool):
    cfg_track_src = os.path.join(stage1_out, "config_bundletrack.yml")
    cfg_nerf_src = os.path.join(stage1_out, "config_nerf.yml")
    if not os.path.exists(cfg_track_src):
        raise FileNotFoundError(cfg_track_src)
    if not os.path.exists(cfg_nerf_src):
        raise FileNotFoundError(cfg_nerf_src)

    cfg_track = yaml.load(open(cfg_track_src, "r"), Loader=yaml.FullLoader)
    cfg_nerf = yaml.load(open(cfg_nerf_src, "r"), Loader=yaml.FullLoader)

    cfg_track["debug_dir"] = out_dir
    cfg_nerf["datadir"] = f"{out_dir}/nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf["ckpt_dir"] = f"{out_dir}/ckpt/nerf_sdf"
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    if freeze_real_poses:
        cfg_nerf["optimize_poses"] = 0

    os.makedirs(cfg_nerf["datadir"], exist_ok=True)
    os.makedirs(cfg_nerf["ckpt_dir"], exist_ok=True)

    cfg_track_out = os.path.join(out_dir, "config_bundletrack.yml")
    cfg_nerf_out = os.path.join(out_dir, "config_nerf.yml")
    yaml.dump(cfg_track, open(cfg_track_out, "w"))
    yaml.dump(cfg_nerf, open(cfg_nerf_out, "w"))

    cam_k_src = os.path.join(stage1_out, "cam_K.txt")
    cam_k_out = os.path.join(out_dir, "cam_K.txt")
    if os.path.exists(cam_k_src):
        shutil.copy(cam_k_src, cam_k_out)

    return cfg_track_out, cfg_nerf_out


def parse_args():
    ap = argparse.ArgumentParser(description="Offline co-reconstruction with registered prior mesh.")
    ap.add_argument(
        "--stage1_out",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2_realonly",
    )
    ap.add_argument(
        "--prior_mesh",
        default="/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/instantmesh/mesh.obj",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/ubuntu/projects/BundleSDF/outputs/real_0212_trash/object_2_completion_registered_offline",
    )
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_real_frames", type=int, default=-1)
    ap.add_argument("--synthetic_views", type=int, default=48)
    ap.add_argument("--synthetic_candidates", type=int, default=384)
    ap.add_argument("--tex_res", type=int, default=2048)
    ap.add_argument("--freeze_real_poses", type=int, default=1)
    ap.add_argument("--synthetic_frame_weight", type=float, default=0.35)
    ap.add_argument("--synthetic_rgb_weight", type=float, default=0.05)
    ap.add_argument("--synthetic_only_uncovered_texels", type=int, default=1)
    ap.add_argument("--max_pool_size", type=int, default=30)
    ap.add_argument("--save_synthetic_rgbd", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rgb_dir = os.path.join(args.stage1_out, "color_segmented")
    depth_dir = os.path.join(args.stage1_out, "depth_filtered")
    mask_dir = os.path.join(args.stage1_out, "mask")
    pose_dir = os.path.join(args.stage1_out, "ob_in_cam")
    cam_k_path = os.path.join(args.stage1_out, "cam_K.txt")

    for d in [rgb_dir, depth_dir, mask_dir, pose_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(d)
    if not os.path.exists(cam_k_path):
        raise FileNotFoundError(cam_k_path)

    frame_ids = collect_common_frame_ids(rgb_dir, depth_dir, mask_dir, pose_dir)
    frame_ids = frame_ids[:: max(int(args.frame_stride), 1)]
    if int(args.max_real_frames) > 0:
        frame_ids = frame_ids[: int(args.max_real_frames)]
    if len(frame_ids) == 0:
        raise RuntimeError("No common frame ids found across RGB/Depth/Mask/Pose.")
    print(f"Using {len(frame_ids)} real frames.")

    k = np.loadtxt(cam_k_path).reshape(3, 3).astype(np.float32)
    first_rgb = read_rgb(os.path.join(rgb_dir, f"{frame_ids[0]}.png"))
    h, w = first_rgb.shape[:2]

    completion_buffer = CompletionBuffer(max_pool_size=int(args.max_pool_size))
    real_cam_in_obs = []
    for fid in frame_ids:
        rgb = read_rgb(os.path.join(rgb_dir, f"{fid}.png"))
        depth = read_depth_m(os.path.join(depth_dir, f"{fid}.png"))
        mask = read_mask(os.path.join(mask_dir, f"{fid}.png"))
        ob_in_cam = np.loadtxt(os.path.join(pose_dir, f"{fid}.txt")).reshape(4, 4)
        cam_in_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
        real_cam_in_obs.append(cam_in_ob.copy())
        completion_buffer.append_real(
            K=k,
            rgb=rgb,
            depth=depth.astype(np.float32),
            mask=mask.astype(np.uint8),
            occ_mask=None,
            cam_in_ob=cam_in_ob,
            frame_id=fid,
        )

    prior_mesh = load_mesh(args.prior_mesh)
    renderer = build_renderer(prior_mesh)
    center = renderer["center"].astype(np.float32)
    radius = float(max(0.15, np.linalg.norm(renderer["extents"]) * 1.2))

    synth_dirs = select_unseen_dirs(
        real_cam_in_obs=np.asarray(real_cam_in_obs, dtype=np.float32),
        center=center,
        n_views=int(args.synthetic_views),
        candidate_count=int(args.synthetic_candidates),
    )
    print(f"Rendering {len(synth_dirs)} synthetic views for unseen coverage.")

    synth_entries = []
    synth_dir = os.path.join(args.out_dir, "synthetic_rgbd")
    if int(args.save_synthetic_rgbd):
        for sub in ["color", "depth", "mask", "ob_in_cam", "cam_in_ob"]:
            os.makedirs(os.path.join(synth_dir, sub), exist_ok=True)

    for i, d in enumerate(synth_dirs):
        cam_pos = center + d * radius
        ob_in_cam = look_at_ob_in_cam(cam_pos, center)
        color, depth, mask = raycast_render(renderer, k, h, w, ob_in_cam)
        frame_id = f"synth_{i:03d}"
        cam_in_ob = np.linalg.inv(ob_in_cam).astype(np.float32)

        synth_entries.append(
            {
                "K": k.copy(),
                "rgb": color.copy(),
                "depth": depth.copy(),
                "mask": mask.copy(),
                "occ_mask": None,
                "cam_in_ob": cam_in_ob.copy(),
                "frame_id": frame_id,
            }
        )

        if int(args.save_synthetic_rgbd):
            cv2.imwrite(
                os.path.join(synth_dir, "color", f"{frame_id}.png"),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
            )
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(synth_dir, "depth", f"{frame_id}.png"), depth_mm)
            cv2.imwrite(
                os.path.join(synth_dir, "mask", f"{frame_id}.png"),
                (mask > 0).astype(np.uint8) * 255,
            )
            np.savetxt(
                os.path.join(synth_dir, "ob_in_cam", f"{frame_id}.txt"),
                ob_in_cam.reshape(4, 4),
                fmt="%.8f",
            )
            np.savetxt(
                os.path.join(synth_dir, "cam_in_ob", f"{frame_id}.txt"),
                cam_in_ob.reshape(4, 4),
                fmt="%.8f",
            )

    completion_buffer.replace_synthetic(synth_entries)

    cfg_track, cfg_nerf = prepare_configs(
        stage1_out=args.stage1_out,
        out_dir=args.out_dir,
        freeze_real_poses=bool(args.freeze_real_poses),
    )
    tracker = BundleSdf(
        cfg_track_dir=cfg_track,
        cfg_nerf_dir=cfg_nerf,
        start_nerf_keyframes=5,
        use_gui=False,
    )

    print("Running co-reconstruction from real + synthetic RGB-D...")
    mesh, rebuilt, metadata = completion_buffer.build_sdf(
        tracker=tracker,
        checkpoint_path=None,
        interpolate_missing_vertices=False,
        tex_res=int(args.tex_res),
        optimize_poses_override=0 if bool(args.freeze_real_poses) else None,
        synthetic_frame_weight=float(args.synthetic_frame_weight),
        synthetic_rgb_weight=float(args.synthetic_rgb_weight),
        synthetic_only_uncovered_texels=bool(args.synthetic_only_uncovered_texels),
    )
    tracker.on_finish()

    if mesh is None:
        raise RuntimeError("Co-reconstruction returned no mesh.")

    out_obj = os.path.join(args.out_dir, "textured_mesh.obj")
    out_glb = os.path.join(args.out_dir, "textured_mesh.glb")
    mesh.export(out_obj)
    mesh.export(out_glb)

    meta = {
        "stage1_out": args.stage1_out,
        "prior_mesh": args.prior_mesh,
        "n_real_frames": len(frame_ids),
        "n_synth_frames": len(synth_entries),
        "freeze_real_poses": bool(args.freeze_real_poses),
        "synthetic_frame_weight": float(args.synthetic_frame_weight),
        "synthetic_rgb_weight": float(args.synthetic_rgb_weight),
        "synthetic_only_uncovered_texels": bool(args.synthetic_only_uncovered_texels),
        "rebuilt": bool(rebuilt),
        "build_metadata": metadata,
        "output_mesh_obj": out_obj,
        "output_mesh_glb": out_glb,
    }
    with open(os.path.join(args.out_dir, "completion_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    imageio.imwrite(
        os.path.join(args.out_dir, "synthetic_views_preview.png"),
        synth_entries[0]["rgb"] if len(synth_entries) > 0 else np.zeros((h, w, 3), dtype=np.uint8),
    )
    print(f"Done. Completed mesh written to: {out_obj}")


if __name__ == "__main__":
    main()
