#!/usr/bin/env python3
"""
Simple color/texture transfer for affine-aligned meshes.

Modes:
1) project: transfer real mesh colors by geometric projection
2) main_color: paint target with dominant color from real mesh
3) both: run both

Additionally for project mode:
- export textured OBJ+MTL+PNG by baking projected colors onto UV texture.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
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

    info = {
        "has_texture_uv": int(tri_uv is not None and tex_img is not None),
        "has_vertex_color": int(tri_vcols is not None),
        "base_color": base_color.tolist(),
    }
    return sample, info


def sample_surface_with_face_bary(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    points, face_ids = trimesh.sample.sample_surface(mesh, int(n_points))
    tris = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)[face_ids]]
    bary = trimesh.triangles.points_to_barycentric(tris, points)
    bary_uv = np.stack([bary[:, 1], bary[:, 2]], axis=1).astype(np.float32)
    return points.astype(np.float32), face_ids.astype(np.int64), bary_uv


def sample_surface_with_colors(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    points, face_ids, bary_uv = sample_surface_with_face_bary(mesh, n_points, rng)
    sample_fn, _ = make_color_sampler(mesh)
    colors = sample_fn(face_ids, bary_uv).astype(np.float32)
    return points, colors


def dominant_color(colors: np.ndarray, bins: int) -> np.ndarray:
    c = np.clip(colors.astype(np.float32), 0.0, 255.0)
    b = max(2, int(bins))
    q = np.floor(c / (256.0 / float(b))).astype(np.int32)
    q = np.clip(q, 0, b - 1)
    key = q[:, 0] + b * q[:, 1] + (b * b) * q[:, 2]
    uniq, cnt = np.unique(key, return_counts=True)
    best_key = int(uniq[np.argmax(cnt)])
    mask = key == best_key
    if int(np.count_nonzero(mask)) == 0:
        return np.array([180, 180, 180], dtype=np.uint8)
    main = np.mean(c[mask], axis=0)
    return np.clip(np.round(main), 0, 255).astype(np.uint8)


def apply_vertex_colors(mesh: trimesh.Trimesh, rgb: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    if rgb.ndim != 2 or rgb.shape[0] != len(out.vertices) or rgb.shape[1] != 3:
        raise ValueError("rgb must be (num_vertices, 3)")
    rgba = np.concatenate(
        [rgb.astype(np.uint8), 255 * np.ones((len(rgb), 1), dtype=np.uint8)], axis=1
    )
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=rgba)
    return out


def ensure_uv_from_mesh(
    target_mesh: trimesh.Trimesh,
    uv_ref_mesh: Optional[trimesh.Trimesh],
) -> Tuple[np.ndarray, str]:
    uv = getattr(target_mesh.visual, "uv", None)
    if uv is not None:
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim == 2 and uv.shape == (len(target_mesh.vertices), 2):
            return uv, "target_uv"

    if uv_ref_mesh is None:
        raise RuntimeError(
            "Target mesh has no UV. Provide --uv_ref_mesh (e.g. sam3d/mesh_aligned.obj)."
        )

    uv_ref = getattr(uv_ref_mesh.visual, "uv", None)
    if uv_ref is None:
        raise RuntimeError("uv_ref_mesh has no UV.")
    uv_ref = np.asarray(uv_ref, dtype=np.float32)
    if uv_ref.ndim != 2 or uv_ref.shape[1] != 2 or len(uv_ref) != len(uv_ref_mesh.vertices):
        raise RuntimeError("uv_ref_mesh has invalid UV shape.")

    ref_faces = np.asarray(uv_ref_mesh.faces, dtype=np.int64)
    ref_tri_v = np.asarray(uv_ref_mesh.vertices, dtype=np.float64)[ref_faces]
    ref_tri_uv = uv_ref[ref_faces]
    tgt_v = np.asarray(target_mesh.vertices, dtype=np.float64)

    try:
        closest, _dist, tri_ids = trimesh.proximity.closest_point(uv_ref_mesh, tgt_v)
        tri_ids = tri_ids.astype(np.int64)
        tris = ref_tri_v[tri_ids]
        bary = trimesh.triangles.points_to_barycentric(tris, closest)
        b0 = bary[:, 0:1]
        b1 = bary[:, 1:2]
        b2 = bary[:, 2:3]
        uv_out = (
            b0 * ref_tri_uv[tri_ids, 0]
            + b1 * ref_tri_uv[tri_ids, 1]
            + b2 * ref_tri_uv[tri_ids, 2]
        )
    except Exception:
        # Fallback: nearest vertex UV transfer.
        tree = cKDTree(np.asarray(uv_ref_mesh.vertices, dtype=np.float64))
        _d, idx = tree.query(tgt_v, k=1)
        uv_out = uv_ref[idx]

    return np.clip(uv_out.astype(np.float32), 0.0, 1.0), "uv_ref_transfer"


def bake_projected_texture(
    target_mesh: trimesh.Trimesh,
    target_uv: np.ndarray,
    real_mesh: trimesh.Trimesh,
    tex_res: int,
    sample_points: int,
    knn: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    sample_fn, _ = make_color_sampler(real_mesh)

    src_pts, src_faces, src_bary = sample_surface_with_face_bary(real_mesh, sample_points, rng=rng)
    src_cols = sample_fn(src_faces, src_bary).astype(np.float32)
    src_tree = cKDTree(src_pts)

    tgt_pts, tgt_faces, tgt_bary = sample_surface_with_face_bary(target_mesh, sample_points, rng=rng)
    tri_uv = target_uv[np.asarray(target_mesh.faces, dtype=np.int64)]
    b1 = tgt_bary[:, 0:1]
    b2 = tgt_bary[:, 1:2]
    b0 = 1.0 - b1 - b2
    uv = (
        b0 * tri_uv[tgt_faces, 0]
        + b1 * tri_uv[tgt_faces, 1]
        + b2 * tri_uv[tgt_faces, 2]
    ).astype(np.float32)

    k = max(1, int(knn))
    d, idx = src_tree.query(tgt_pts, k=k)
    if k == 1:
        col = src_cols[idx]
        d0 = d
    else:
        ww = 1.0 / (d + 1e-6)
        ww = ww / np.maximum(np.sum(ww, axis=1, keepdims=True), 1e-12)
        col = np.sum(src_cols[idx] * ww[:, :, None], axis=1)
        d0 = d[:, 0]

    h = w = int(tex_res)
    tex_sum = np.zeros((h, w, 3), dtype=np.float64)
    tex_w = np.zeros((h, w), dtype=np.float64)
    uu = np.clip(np.round(uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    vv = np.clip(np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    np.add.at(tex_sum, (vv, uu, 0), col[:, 0])
    np.add.at(tex_sum, (vv, uu, 1), col[:, 1])
    np.add.at(tex_sum, (vv, uu, 2), col[:, 2])
    np.add.at(tex_w, (vv, uu), 1.0)

    valid = tex_w > 0
    tex = np.zeros((h, w, 3), dtype=np.float32)
    tex[valid] = (tex_sum[valid] / tex_w[valid, None]).astype(np.float32)

    if np.any(~valid):
        # Fill missing texels from nearest valid texel.
        _, indices = distance_transform_edt(~valid, return_indices=True)
        tex[~valid] = tex[indices[0][~valid], indices[1][~valid]]

    stats = {
        "mean_nn_dist": float(np.mean(d0)),
        "p95_nn_dist": float(np.percentile(d0, 95)),
        "coverage_before_fill": float(np.mean(valid)),
    }
    return np.clip(np.round(tex), 0, 255).astype(np.uint8), stats


def save_colored_mesh(mesh: trimesh.Trimesh, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))


def save_textured_obj(
    mesh: trimesh.Trimesh, uv: np.ndarray, tex_img: np.ndarray, out_obj: Path
) -> Tuple[Path, Path]:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    out_mtl = out_obj.with_suffix(".mtl")
    out_tex = out_obj.with_name(f"{out_obj.stem}_texture.png")

    imageio.imwrite(str(out_tex), tex_img.astype(np.uint8))

    with open(out_mtl, "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {out_tex.name}\n")

    v = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if vn.shape != v.shape:
        vn = np.zeros_like(v)
        vn[:, 2] = 1.0
    uv = np.asarray(uv, dtype=np.float64)

    with open(out_obj, "w") as f:
        f.write("# Custom textured OBJ writer\n")
        f.write(f"mtllib {out_mtl.name}\n")
        f.write(f"o {out_obj.stem}\n")
        for vv in v:
            f.write(f"v {vv[0]:.8f} {vv[1]:.8f} {vv[2]:.8f}\n")
        for vt in uv:
            f.write(f"vt {vt[0]:.8f} {vt[1]:.8f}\n")
        for nn in vn:
            f.write(f"vn {nn[0]:.8f} {nn[1]:.8f} {nn[2]:.8f}\n")
        f.write("usemtl material_0\n")
        for tri in faces:
            i0 = int(tri[0]) + 1
            i1 = int(tri[1]) + 1
            i2 = int(tri[2]) + 1
            f.write(
                f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n"
            )
    return out_mtl, out_tex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Simple projection and main-color transfer for affine-aligned mesh."
    )
    ap.add_argument("--target_mesh", required=True, type=str)
    ap.add_argument("--real_mesh", required=True, type=str)
    ap.add_argument("--mode", default="both", choices=["project", "main_color", "both"])
    ap.add_argument("--sample_points", type=int, default=200000)
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--main_color_bins", type=int, default=16)
    ap.add_argument("--tex_res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uv_ref_mesh", type=str, default="")

    ap.add_argument("--out_project_glb", type=str, default="")
    ap.add_argument("--out_main_glb", type=str, default="")
    ap.add_argument("--out_project_ply", type=str, default="")
    ap.add_argument("--out_main_ply", type=str, default="")
    ap.add_argument("--out_project_obj", type=str, default="")
    ap.add_argument("--out_main_obj", type=str, default="")
    return ap.parse_args()


def default_out(path: Path, suffix: str, ext: str) -> Path:
    return (path.parent / f"{path.stem}_{suffix}.{ext}").resolve()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    target_mesh_path = Path(args.target_mesh).expanduser().resolve()
    real_mesh_path = Path(args.real_mesh).expanduser().resolve()
    uv_ref_mesh_path = (
        Path(args.uv_ref_mesh).expanduser().resolve()
        if args.uv_ref_mesh
        else (target_mesh_path.parent / "mesh_aligned.obj").resolve()
    )
    if not target_mesh_path.exists():
        raise FileNotFoundError(target_mesh_path)
    if not real_mesh_path.exists():
        raise FileNotFoundError(real_mesh_path)

    out_project_glb = (
        Path(args.out_project_glb).expanduser().resolve()
        if args.out_project_glb
        else default_out(target_mesh_path, "projected_color", "glb")
    )
    out_main_glb = (
        Path(args.out_main_glb).expanduser().resolve()
        if args.out_main_glb
        else default_out(target_mesh_path, "main_color", "glb")
    )
    out_project_ply = (
        Path(args.out_project_ply).expanduser().resolve()
        if args.out_project_ply
        else default_out(target_mesh_path, "projected_color", "ply")
    )
    out_main_ply = (
        Path(args.out_main_ply).expanduser().resolve()
        if args.out_main_ply
        else default_out(target_mesh_path, "main_color", "ply")
    )
    out_project_obj = (
        Path(args.out_project_obj).expanduser().resolve()
        if args.out_project_obj
        else default_out(target_mesh_path, "projected_color_textured", "obj")
    )
    out_main_obj = (
        Path(args.out_main_obj).expanduser().resolve()
        if args.out_main_obj
        else default_out(target_mesh_path, "main_color_textured", "obj")
    )

    target_mesh = load_mesh(str(target_mesh_path))
    real_mesh = load_mesh(str(real_mesh_path))
    _sample_fn, sampler_info = make_color_sampler(real_mesh)

    print(f"Loaded target mesh: {target_mesh_path}")
    print(f"Loaded real mesh: {real_mesh_path}")
    print(
        "Real mesh color source: "
        f"texture_uv={sampler_info['has_texture_uv']} "
        f"vertex_color={sampler_info['has_vertex_color']} "
        f"base_color={sampler_info['base_color']}"
    )

    if args.mode in ("project", "both"):
        # Existing quick outputs (vertex color).
        src_pts, src_cols = sample_surface_with_colors(
            real_mesh, int(args.sample_points), rng=rng
        )
        tree = cKDTree(src_pts)
        tgt_v = np.asarray(target_mesh.vertices, dtype=np.float32)
        k = max(1, int(args.knn))
        d, idx = tree.query(tgt_v, k=k)
        if k == 1:
            cols = src_cols[idx]
        else:
            ww = 1.0 / (d + 1e-6)
            ww = ww / np.maximum(np.sum(ww, axis=1, keepdims=True), 1e-12)
            cols = np.sum(src_cols[idx] * ww[:, :, None], axis=1)
        cols_u8 = np.clip(np.round(cols), 0, 255).astype(np.uint8)
        mesh_proj = apply_vertex_colors(target_mesh, cols_u8)
        try:
            save_colored_mesh(mesh_proj, out_project_glb)
            save_colored_mesh(mesh_proj, out_project_ply)
            print(f"[project] vertex-color glb: {out_project_glb}")
            print(f"[project] vertex-color ply: {out_project_ply}")
        except PermissionError as e:
            print(f"[project] warning: skip vertex-color outputs due to permission: {e}")

        # New textured OBJ output.
        uv_ref_mesh = load_mesh(str(uv_ref_mesh_path)) if uv_ref_mesh_path.exists() else None
        target_uv, uv_source = ensure_uv_from_mesh(target_mesh, uv_ref_mesh)
        tex_img, tex_stats = bake_projected_texture(
            target_mesh=target_mesh,
            target_uv=target_uv,
            real_mesh=real_mesh,
            tex_res=int(args.tex_res),
            sample_points=int(args.sample_points),
            knn=int(args.knn),
            rng=rng,
        )
        out_mtl, out_tex = save_textured_obj(target_mesh, target_uv, tex_img, out_project_obj)
        print(f"[project] textured obj: {out_project_obj}")
        print(f"[project] textured mtl: {out_mtl}")
        print(f"[project] textured png: {out_tex}")
        print(
            "[project] textured stats: "
            f"uv_source={uv_source} "
            f"coverage_before_fill={tex_stats['coverage_before_fill']:.4f} "
            f"mean_nn_dist={tex_stats['mean_nn_dist']:.6f} "
            f"p95_nn_dist={tex_stats['p95_nn_dist']:.6f}"
        )

    if args.mode in ("main_color", "both"):
        _, real_sample_cols = sample_surface_with_colors(
            real_mesh, n_points=max(20000, int(args.sample_points // 3)), rng=rng
        )
        main_col = dominant_color(real_sample_cols, bins=int(args.main_color_bins))
        cols_main = np.repeat(main_col[None, :], len(target_mesh.vertices), axis=0)
        mesh_main = apply_vertex_colors(target_mesh, cols_main)
        try:
            save_colored_mesh(mesh_main, out_main_glb)
            save_colored_mesh(mesh_main, out_main_ply)
            print(f"[main_color] written glb: {out_main_glb}")
            print(f"[main_color] written ply: {out_main_ply}")
        except PermissionError as e:
            print(f"[main_color] warning: skip outputs due to permission: {e}")
        uv_ref_mesh = load_mesh(str(uv_ref_mesh_path)) if uv_ref_mesh_path.exists() else None
        target_uv, uv_source = ensure_uv_from_mesh(target_mesh, uv_ref_mesh)
        tex_main = np.zeros((int(args.tex_res), int(args.tex_res), 3), dtype=np.uint8)
        tex_main[:, :, :] = main_col[None, None, :]
        out_mtl_main, out_tex_main = save_textured_obj(
            target_mesh, target_uv, tex_main, out_main_obj
        )
        print(f"[main_color] textured obj: {out_main_obj}")
        print(f"[main_color] textured mtl: {out_mtl_main}")
        print(f"[main_color] textured png: {out_tex_main}")
        print(f"[main_color] uv_source={uv_source}")
        print(f"[main_color] rgb={main_col.tolist()} bins={int(args.main_color_bins)}")


if __name__ == "__main__":
    main()
