# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
import os, sys
import shutil

code_dir_run_custom = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir_run_custom)
from bundlesdf import *
from segmentation_utils import Segmenter

print(f"code_dir resolved to: {code_dir_run_custom}")

config_path = f"{code_dir_run_custom}/BundleTrack/config_ho3d.yml"
print(f"Looking for config at: {config_path}")
print(f"Config file exists: {os.path.exists(config_path)}")

DEFAULT_SYN_BASE = (
    "/home/ubuntu/projects/BundleSDF/data/real_0212_trash/object_2/sam3d"
)
DEFAULT_SYN_RGB_DIR = f"{DEFAULT_SYN_BASE}/rgb"
DEFAULT_SYN_DEPTH_DIR = f"{DEFAULT_SYN_BASE}/depth"
DEFAULT_SYN_POSE_DIR = f"{DEFAULT_SYN_BASE}/ob_in_cam"


def prepare_syn_rgbd_for_global_refine(
    out_folder,
    syn_rgb_dir,
    syn_depth_dir,
    syn_pose_dir,
    syn_prefix="syn_",
):
    out_folder = os.path.abspath(out_folder)
    color_dir = f"{out_folder}/color_segmented"
    depth_dir = f"{out_folder}/depth_filtered"
    mask_dir = f"{out_folder}/mask"
    pose_dir = f"{out_folder}/ob_in_cam"

    # Global refine (without reader) consumes these dirs directly.
    for d in [color_dir, depth_dir, mask_dir, pose_dir]:
        if not os.path.exists(d):
            raise RuntimeError(
                f"Required directory missing for global refine: {d}. "
                f"Run tracking first to produce real-frame outputs."
            )

    syn_rgb_files = sorted(glob.glob(f"{syn_rgb_dir}/*.png"))
    syn_depth_files = sorted(glob.glob(f"{syn_depth_dir}/*.png"))
    syn_pose_files = sorted(glob.glob(f"{syn_pose_dir}/*.txt"))
    if len(syn_rgb_files) == 0:
        raise RuntimeError(f"No synthetic RGB files found in {syn_rgb_dir}")
    if len(syn_depth_files) == 0:
        raise RuntimeError(f"No synthetic depth files found in {syn_depth_dir}")
    if len(syn_pose_files) == 0:
        raise RuntimeError(f"No synthetic pose files found in {syn_pose_dir}")

    rgb_ids = {os.path.basename(f).replace(".png", "") for f in syn_rgb_files}
    depth_ids = {os.path.basename(f).replace(".png", "") for f in syn_depth_files}
    pose_ids = {os.path.basename(f).replace(".txt", "") for f in syn_pose_files}
    common_ids = sorted(rgb_ids & depth_ids & pose_ids)
    if len(common_ids) == 0:
        raise RuntimeError(
            "No common frame ids across synthetic rgb/depth/pose directories."
        )

    # Remove stale synthetic files from previous runs (real frames are untouched).
    for d, ext in [(color_dir, "png"), (depth_dir, "png"), (mask_dir, "png"), (pose_dir, "txt")]:
        for f in glob.glob(f"{d}/{syn_prefix}*.{ext}"):
            os.remove(f)

    added = 0
    for fid in common_ids:
        out_id = f"{syn_prefix}{fid}"
        src_rgb = f"{syn_rgb_dir}/{fid}.png"
        src_depth = f"{syn_depth_dir}/{fid}.png"
        src_pose = f"{syn_pose_dir}/{fid}.txt"
        dst_rgb = f"{color_dir}/{out_id}.png"
        dst_depth = f"{depth_dir}/{out_id}.png"
        dst_mask = f"{mask_dir}/{out_id}.png"
        dst_pose = f"{pose_dir}/{out_id}.txt"

        shutil.copy2(src_rgb, dst_rgb)

        depth = cv2.imread(src_depth, -1)
        if depth is None:
            raise RuntimeError(f"Failed to read synthetic depth: {src_depth}")
        if depth.dtype != np.uint16:
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
        cv2.imwrite(dst_depth, depth)

        syn_mask = (depth > 0).astype(np.uint8) * 255
        cv2.imwrite(dst_mask, syn_mask)

        ob_in_cam = np.loadtxt(src_pose).reshape(4, 4)
        np.savetxt(dst_pose, ob_in_cam, fmt="%.8f")
        added += 1

    print(
        f"[run_custom.py] Added {added} synthetic frames for global refine "
        f"using saved poses ({syn_prefix}* ids)."
    )
    return added


def run_one_video(
    video_dir="/home/bowen/debug/2022-11-18-15-10-24_milk",
    out_folder="/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk/",
    use_segmenter=False,
    use_gui=False,
    debug_level=2,
    interpolate_missing_vertices=False,
    use_syn_rgbd=0,
    syn_rgb_dir=DEFAULT_SYN_RGB_DIR,
    syn_depth_dir=DEFAULT_SYN_DEPTH_DIR,
    syn_pose_dir=DEFAULT_SYN_POSE_DIR,
    syn_prefix="syn_",
    mesh_out_name="textured_mesh.obj",
):
    set_seed(0)
    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")
    cfg_bundletrack = yaml.load(
        open(f"{code_dir_run_custom}/BundleTrack/config_ho3d.yml", "r")
    )
    cfg_bundletrack["SPDLOG"] = debug_level  # Higher means more logging
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["depth_processing"]["zfar"] = 4.0
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_bundletrack["bundle"]["max_BA_frames"] = 10  # TODO
    cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    cfg_bundletrack["feature_corres"]["map_points"] = True
    cfg_bundletrack["feature_corres"]["resize"] = 400
    cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
    cfg_bundletrack["keyframe"]["min_rot"] = 5
    cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
    cfg_bundletrack["p2p"]["max_dist"] = 0.02
    cfg_bundletrack["p2p"]["max_normal_angle"] = 45
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))
    cfg_nerf = yaml.load(open(f"{code_dir_run_custom}/config.yml", "r"))
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf["ckpt_dir"] = f"{out_folder}/ckpt/nerf_sdf"
    cfg_nerf["i_weights"] = cfg_nerf["n_step"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))
    if use_segmenter:
        segmenter = Segmenter()
    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=use_gui,
    )
    # Note: specifying a shorter_side will downsample the images such that the shorter
    # side of the image has the specified number of pixels.
    # reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)
    reader = YcbineoatReader(video_dir=video_dir)
    for i in range(0, len(reader.color_files), 1):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)[..., :3]
        H0, W0 = color.shape[:2]
        depth = reader.get_depth(i)
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        if i == 0:
            mask = reader.get_mask(0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
        else:
            if use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
            else:
                mask = reader.get_mask(i)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg_bundletrack["erode_mask"] > 0:
            kernel = np.ones(
                (cfg_bundletrack["erode_mask"], cfg_bundletrack["erode_mask"]), np.uint8
            )
            mask = cv2.erode(mask.astype(np.uint8), kernel)
        # Skip frames that would produce an empty point cloud after depth/mask filtering.
        znear = 0.01
        zfar = cfg_bundletrack["depth_processing"].get("zfar", np.inf)
        # Heuristic: if most masked pixels are too close, skip the frame to avoid tracking failures.
        masked = mask > 0
        if masked.sum() > 0:
            too_close = (depth > 0) & (depth < znear) & masked
            if too_close.sum() / masked.sum() > 0.5:
                print(
                    f"[run_custom.py] skip frame {reader.id_strs[i]}: too-close depth (znear={znear})"
                )
                continue
        valid = (depth >= znear) & (depth <= zfar) & masked
        if valid.sum() == 0:
            print(
                f"[run_custom.py] skip frame {reader.id_strs[i]}: empty masked depth (zfar={zfar})"
            )
            continue
        id_str = reader.id_strs[i]
        pose_in_model = np.eye(4)
        K = reader.K.copy()
        tracker.run(
            color,
            depth,
            K,
            id_str,
            mask=mask,
            occ_mask=None,
            pose_in_model=pose_in_model,
        )
    tracker.on_finish()
    run_one_video_global_nerf(
        video_dir=video_dir,
        out_folder=out_folder,
        interpolate_missing_vertices=interpolate_missing_vertices,
        use_syn_rgbd=use_syn_rgbd,
        syn_rgb_dir=syn_rgb_dir,
        syn_depth_dir=syn_depth_dir,
        syn_pose_dir=syn_pose_dir,
        syn_prefix=syn_prefix,
        mesh_out_name=mesh_out_name,
    )


def run_one_video_global_nerf(
    video_dir,
    out_folder="/home/bowen/debug/bundlesdf_scan_coffee_415",
    interpolate_missing_vertices=False,
    use_syn_rgbd=0,
    syn_rgb_dir=DEFAULT_SYN_RGB_DIR,
    syn_depth_dir=DEFAULT_SYN_DEPTH_DIR,
    syn_pose_dir=DEFAULT_SYN_POSE_DIR,
    syn_prefix="syn_",
    mesh_out_name="textured_mesh.obj",
):
    set_seed(0)
    out_folder += "/"  #!NOTE there has to be a / in the end
    cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml", "r"))
    cfg_bundletrack["debug_dir"] = out_folder
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))
    cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml", "r"))
    cfg_nerf["n_step"] = 3000
    cfg_nerf["N_samples"] = 128
    cfg_nerf["N_samples_around_depth"] = 256
    cfg_nerf["first_frame_weight"] = 1
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["finest_res"] = 512
    cfg_nerf["num_levels"] = 16
    cfg_nerf["mesh_resolution"] = 0.002
    cfg_nerf["n_train_image"] = 600
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["frame_features"] = 2
    cfg_nerf["rgb_weight"] = 100
    cfg_nerf["i_img"] = np.inf
    cfg_nerf["i_mesh"] = cfg_nerf["i_img"]
    cfg_nerf["i_nerf_normals"] = cfg_nerf["i_img"]
    cfg_nerf["i_save_ray"] = cfg_nerf["i_img"]
    cfg_nerf["i_weights"] = cfg_nerf["n_step"]
    cfg_nerf["datadir"] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = copy.deepcopy(cfg_nerf["datadir"])
    cfg_nerf["ckpt_dir"] = f"{out_folder}/ckpt/nerf_sdf"
    os.makedirs(cfg_nerf["datadir"], exist_ok=True)
    cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    reader = YcbineoatReader(video_dir=video_dir, downscale=1)
    if use_syn_rgbd:
        prepare_syn_rgbd_for_global_refine(
            out_folder=out_folder,
            syn_rgb_dir=syn_rgb_dir,
            syn_depth_dir=syn_depth_dir,
            syn_pose_dir=syn_pose_dir,
            syn_prefix=syn_prefix,
        )
        # Use saved outputs directly, so synthetic frames use provided ob_in_cam poses.
        reader = None

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5
    )
    tracker.cfg_nerf = cfg_nerf
    tracker.run_global_nerf(
        reader=reader,
        get_texture=True,
        tex_res=2048,
        use_all_frames=True,
        interpolate_missing_vertices=interpolate_missing_vertices,
        mesh_out_name=mesh_out_name,
    )
    tracker.on_finish()
    print(f"Done")


def postprocess_mesh(out_folder):
    mesh_files = sorted(
        glob.glob(f"{out_folder}/**/nerf/*normalized_space.obj", recursive=True)
    )
    print(f"Using {mesh_files[-1]}")
    os.makedirs(f"{out_folder}/mesh/", exist_ok=True)
    print(f"\nSaving meshes to {out_folder}/mesh/\n")
    mesh = trimesh.load(mesh_files[-1])
    with open(f"{os.path.dirname(mesh_files[-1])}/config.yml", "r") as ff:
        cfg = yaml.load(ff)
    tf = np.eye(4)
    tf[:3, 3] = cfg["translation"]
    tf1 = np.eye(4)
    tf1[:3, :3] *= cfg["sc_factor"]
    tf = tf1 @ tf
    mesh.apply_transform(np.linalg.inv(tf))
    mesh.export(f"{out_folder}/mesh/mesh_real_scale.obj")
    components = trimesh_split(mesh, min_edge=1000)
    best_component = None
    best_size = 0
    for component in components:
        dists = np.linalg.norm(component.vertices, axis=-1)
        if len(component.vertices) > best_size:
            best_size = len(component.vertices)
            best_component = component
    mesh = trimesh_clean(best_component)
    mesh.export(f"{out_folder}/mesh/mesh_biggest_component.obj")
    mesh = trimesh.smoothing.filter_laplacian(
        mesh,
        lamb=0.5,
        iterations=3,
        implicit_time_integration=False,
        volume_constraint=True,
        laplacian_operator=None,
    )
    mesh.export(f"{out_folder}/mesh/mesh_biggest_component_smoothed.obj")


def draw_pose():
    K = np.loadtxt(f"{args.out_folder}/cam_K.txt").reshape(3, 3)
    color_files = sorted(glob.glob(f"{args.out_folder}/color/*"))
    mesh = trimesh.load(f"{args.out_folder}/textured_mesh.obj")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    out_dir = f"{args.out_folder}/pose_vis"
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Saving to {out_dir}")
    for color_file in color_files:
        color = imageio.imread(color_file)
        pose = np.loadtxt(
            color_file.replace(".png", ".txt").replace("color", "ob_in_cam")
        )
        pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(
            K, color, ob_in_cam=pose, bbox=bbox, line_color=(255, 255, 0)
        )
        id_str = os.path.basename(color_file).replace(".png", "")
        imageio.imwrite(f"{out_dir}/{id_str}.png", vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="run_video",
        help="run_video/global_refine/draw_pose",
    )
    parser.add_argument(
        "--video_dir", type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk",
    )
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=1)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="interval of frames to run; 1 means using every frame",
    )
    parser.add_argument(
        "--debug_level", type=int, default=2, help="higher means more logging"
    )
    parser.add_argument(
        "--interpolate_missing_vertices",
        type=int,
        default=0,
        help="If specified, interpolate missing vertices in the mesh.",
    )
    parser.add_argument(
        "--use_syn_rgbd",
        type=int,
        default=0,
        help="If 1, append synthetic RGB-D + saved ob_in_cam for global NeRF/SDF refine.",
    )
    parser.add_argument("--syn_rgb_dir", type=str, default=DEFAULT_SYN_RGB_DIR)
    parser.add_argument("--syn_depth_dir", type=str, default=DEFAULT_SYN_DEPTH_DIR)
    parser.add_argument("--syn_pose_dir", type=str, default=DEFAULT_SYN_POSE_DIR)
    parser.add_argument(
        "--syn_prefix",
        type=str,
        default="syn_",
        help="Prefix for injected synthetic frame ids in out_folder.",
    )
    parser.add_argument(
        "--mesh_out_name",
        type=str,
        default="textured_mesh.obj",
        help="Output mesh filename written under out_folder (global refine result).",
    )
    args = parser.parse_args()
    if args.mode == "run_video":
        run_one_video(
            video_dir=args.video_dir,
            out_folder=args.out_folder,
            use_segmenter=args.use_segmenter,
            use_gui=args.use_gui,
            debug_level=args.debug_level,
            interpolate_missing_vertices=args.interpolate_missing_vertices,
            use_syn_rgbd=args.use_syn_rgbd,
            syn_rgb_dir=args.syn_rgb_dir,
            syn_depth_dir=args.syn_depth_dir,
            syn_pose_dir=args.syn_pose_dir,
            syn_prefix=args.syn_prefix,
            mesh_out_name=args.mesh_out_name,
        )
    elif args.mode == "global_refine":
        run_one_video_global_nerf(
            video_dir=args.video_dir,
            out_folder=args.out_folder,
            interpolate_missing_vertices=args.interpolate_missing_vertices,
            use_syn_rgbd=args.use_syn_rgbd,
            syn_rgb_dir=args.syn_rgb_dir,
            syn_depth_dir=args.syn_depth_dir,
            syn_pose_dir=args.syn_pose_dir,
            syn_prefix=args.syn_prefix,
            mesh_out_name=args.mesh_out_name,
        )
    elif args.mode == "draw_pose":
        draw_pose()
    else:
        raise RuntimeError
