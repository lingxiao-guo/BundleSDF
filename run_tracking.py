import argparse
import glob
import logging
import os
import sys
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import copy

code_dir_run_tracking = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir_run_tracking)

from bundlesdf import BundleSdf, LoftrRunner, my_cpp, run_gui, set_seed, yaml
from BundleTrack.scripts.data_reader import YcbineoatReader
from nerf_runner import NerfRunner
from nerf_helpers import preprocess_data
from segmentation_utils import Segmenter
from Utils import glcam_in_cvcam, get_optimized_poses_in_real_world
import multiprocessing
import open3d as o3d


def _find_pose_only_config(ckpt_path, cfg_path=None):
    if cfg_path:
        return cfg_path

    ckpt_path = Path(ckpt_path).resolve()
    ckpt_dir = ckpt_path.parent
    direct = ckpt_dir / "config.yml"
    if direct.exists():
        return str(direct)

    out_folder = ckpt_dir.parent.parent
    final_cfg = out_folder / "final" / "nerf" / "config.yml"
    if final_cfg.exists():
        return str(final_cfg)

    candidates = list(out_folder.glob("*/nerf/config.yml"))
    numeric = []
    for path in candidates:
        frame_id = path.parent.parent.name
        if frame_id.isdigit():
            numeric.append((int(frame_id), path))
    if numeric:
        numeric.sort(key=lambda x: x[0])
        return str(numeric[-1][1])

    raise FileNotFoundError(
        f"Could not locate config.yml for ckpt {ckpt_path}. "
        "Pass --pose_only_cfg to specify it explicitly."
    )


def _run_pose_only_refine(
    ckpt_path,
    cfg_path,
    rgbs,
    depths,
    masks,
    cam_in_obs,
    K,
    out_folder,
    pose_only_steps,
    pose_only_n_rand,
    frame_ids,
):
    cfg = yaml.load(open(cfg_path, "r"))
    cfg["translation"] = np.array(cfg["translation"], dtype=np.float32)
    cfg["sc_factor"] = float(cfg["sc_factor"])
    cfg["bounding_box"] = np.array(cfg["bounding_box"]).reshape(2, 3)
    cfg["use_octree"] = 1
    cfg["denoise_depth_use_octree_cloud"] = False
    cfg["save_octree_clouds"] = False
    cfg["n_step"] = int(pose_only_steps)
    cfg["i_weights"] = int(1e9)
    cfg["i_img"] = int(1e9)
    cfg["i_mesh"] = int(1e9)
    cfg["i_nerf_normals"] = int(1e9)
    cfg["i_save_ray"] = int(1e9)
    if pose_only_n_rand is not None:
        cfg["N_rand"] = int(pose_only_n_rand)

    pose_only_dir = f"{out_folder}/pose_only/nerf"
    os.makedirs(pose_only_dir, exist_ok=True)
    cfg["save_dir"] = pose_only_dir
    cfg["ckpt_dir"] = pose_only_dir

    glcam_in_obs = cam_in_obs @ glcam_in_cvcam
    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)
    masks = np.asarray(masks)
    rgbs, depths, masks, normal_maps, poses = preprocess_data(
        rgbs, depths, masks, None, glcam_in_obs, cfg["sc_factor"], cfg["translation"]
    )

    dummy_pcd = o3d.geometry.PointCloud()
    dummy_pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float32))

    nerf = NerfRunner(
        cfg,
        rgbs,
        depths=depths,
        masks=masks,
        normal_maps=normal_maps,
        poses=poses,
        K=K,
        build_octree_pcd=dummy_pcd,
    )
    nerf.load_weights(ckpt_path)

    for k, model in nerf.models.items():
        if k == "pose_array" or model is None:
            continue
        for p in model.parameters():
            p.requires_grad = False

    if nerf.models["pose_array"] is None:
        raise RuntimeError("pose_array is not initialized; ensure optimize_poses=1.")

    nerf.optimizer = torch.optim.Adam(
        [{"name": "pose_array", "params": nerf.models["pose_array"].parameters(), "lr": cfg["lrate_pose"]}],
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-15,
    )
    nerf.param_groups_init = copy.deepcopy(nerf.optimizer.param_groups)

    nerf.train()

    optimized_cvcam_in_obs, _ = get_optimized_poses_in_real_world(
        poses, nerf.models["pose_array"], cfg["sc_factor"], cfg["translation"]
    )

    refined_dir = f"{out_folder}/ob_in_cam_refined"
    os.makedirs(refined_dir, exist_ok=True)
    for frame_id, cam_in_ob in zip(frame_ids, optimized_cvcam_in_obs):
        ob_in_cam = np.linalg.inv(cam_in_ob)
        np.savetxt(f"{refined_dir}/{frame_id}.txt", ob_in_cam)




class BundleSdfTracking(BundleSdf):
    def __init__(self, cfg_track_dir, use_gui=False):
        with open(cfg_track_dir, "r") as ff:
            self.cfg_track = yaml.load(ff)
        self.debug_dir = self.cfg_track["debug_dir"]
        self.SPDLOG = self.cfg_track["SPDLOG"]
        self.use_gui = use_gui
        self.translation = None
        self.sc_factor = None

        self.manager = multiprocessing.Manager()
        if self.use_gui:
            self.gui_lock = multiprocessing.Lock()
            self.gui_dict = self.manager.dict()
            self.gui_dict["join"] = False
            self.gui_dict["started"] = False
            self.gui_worker = multiprocessing.Process(
                target=run_gui, args=(self.gui_dict, self.gui_lock)
            )
            self.gui_worker.start()
        else:
            self.gui_lock = None
            self.gui_dict = None

        yml = my_cpp.YamlLoadFile(cfg_track_dir)
        self.bundler = my_cpp.Bundler(yml)
        self.loftr = LoftrRunner()
        self.cnt = -1
        self.K = None
        self.mesh = None

    def on_finish(self):
        if self.use_gui:
            with self.gui_lock:
                self.gui_dict["join"] = True
            self.gui_worker.join()

    def run(
        self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)
    ):
        self.cnt += 1

        if self.K is None:
            self.K = K

        if self.use_gui:
            while 1:
                with self.gui_lock:
                    started = self.gui_dict["started"]
                if not started:
                    time.sleep(1)
                    logging.info("Waiting for GUI")
                    continue
                break

        H, W = color.shape[:2]

        percentile = self.cfg_track["depth_processing"]["percentile"]
        if percentile < 100:
            logging.info("percentile denoise start")
            valid = (depth >= 0.1) & (mask > 0)
            thres = np.percentile(depth[valid], percentile)
            depth[depth >= thres] = 0
            logging.info("percentile denoise done")

        frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
        os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

        logging.info(f"processNewFrame start {frame._id_str}")
        self.process_new_frame(frame)
        logging.info(f"processNewFrame done {frame._id_str}")

        self.bundler.saveNewframeResult()
        if self.SPDLOG >= 2 and occ_mask is not None:
            os.makedirs(f"{self.debug_dir}/occ_mask/", exist_ok=True)
            cv2.imwrite(f"{self.debug_dir}/occ_mask/{frame._id_str}.png", occ_mask)

        if self.use_gui:
            ob_in_cam = np.linalg.inv(frame._pose_in_model)
            with self.gui_lock:
                self.gui_dict["color"] = color[..., ::-1]
                self.gui_dict["mask"] = mask
                self.gui_dict["ob_in_cam"] = ob_in_cam
                self.gui_dict["id_str"] = frame._id_str
                self.gui_dict["K"] = self.K
                self.gui_dict["n_keyframe"] = len(self.bundler._keyframes)


def run_one_video_tracking(
    video_dir,
    out_folder,
    use_segmenter=False,
    use_gui=False,
    debug_level=2,
    stride=1,
    overwrite=False,
    pose_only_ckpt=None,
    pose_only_cfg=None,
    pose_only_steps=50,
    pose_only_n_rand=None,
):
    set_seed(0)
    if overwrite:
        os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")
    else:
        os.makedirs(out_folder, exist_ok=True)

    cfg_bundletrack = yaml.load(
        open(f"{code_dir_run_tracking}/BundleTrack/config_ho3d.yml", "r")
    )
    cfg_bundletrack["SPDLOG"] = debug_level
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_bundletrack["bundle"]["max_BA_frames"] = 10
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

    segmenter = Segmenter() if use_segmenter else None

    tracker = BundleSdfTracking(cfg_track_dir=cfg_track_dir, use_gui=use_gui)
    reader = YcbineoatReader(video_dir=video_dir)
    mask_files = sorted(glob.glob(f"{video_dir}/masks/*.png"))
    mask_ids = {os.path.basename(p).replace(".png", "") for p in mask_files}
    color_ids = [os.path.basename(p).replace(".png", "") for p in reader.color_files]
    valid_ids = [id_str for id_str in color_ids if id_str in mask_ids]
    if len(valid_ids) == 0:
        raise RuntimeError(
            f"No matching mask files found under {video_dir}/masks for rgb frames."
        )
    pose_only_rgbs = []
    pose_only_depths = []
    pose_only_masks = []
    pose_only_ids = []
    for i in range(0, len(reader.color_files), stride):
        color_file = reader.color_files[i]
        id_str = reader.id_strs[i]
        if id_str not in mask_ids:
            continue
        color = cv2.imread(color_file)[..., :3]
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

        zfar = cfg_bundletrack["depth_processing"].get("zfar", np.inf)
        valid = (depth >= 0.1) & (depth <= zfar) & (mask > 0)
        if valid.sum() == 0:
            print(
                f"[run_tracking.py] skip frame {reader.id_strs[i]}: empty masked depth (zfar={zfar})"
            )
            continue

        tracker.run(color, depth, reader.K, id_str, mask=mask)
        ob_src = f"{out_folder}/ob_in_cam/{id_str}.txt"
        if os.path.exists(ob_src):
            ob_dst_dir = f"{out_folder}/ob_in_cam_global_video"
            os.makedirs(ob_dst_dir, exist_ok=True)
            shutil.copyfile(ob_src, f"{ob_dst_dir}/{id_str}.txt")
        if pose_only_ckpt:
            pose_only_rgbs.append(color.copy())
            pose_only_depths.append(depth.copy())
            pose_only_masks.append(mask.copy())
            pose_only_ids.append(id_str)

    tracker.on_finish()

    if pose_only_ckpt:
        cfg_path = _find_pose_only_config(pose_only_ckpt, pose_only_cfg)
        pose_dir = f"{out_folder}/ob_in_cam_global_video"
        cam_in_obs = []
        for frame_id in pose_only_ids:
            ob_in_cam_path = f"{pose_dir}/{frame_id}.txt"
            ob_in_cam = np.loadtxt(ob_in_cam_path).reshape(4, 4)
            cam_in_obs.append(np.linalg.inv(ob_in_cam))
        cam_in_obs = np.array(cam_in_obs)
        _run_pose_only_refine(
            pose_only_ckpt,
            cfg_path,
            pose_only_rgbs,
            pose_only_depths,
            pose_only_masks,
            cam_in_obs,
            reader.K,
            out_folder,
            pose_only_steps,
            pose_only_n_rand,
            pose_only_ids,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk_track/",
    )
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=0)
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
        "--overwrite",
        type=int,
        default=0,
        help="If set to 1, delete out_folder before running.",
    )
    parser.add_argument(
        "--pose_only_ckpt",
        type=str,
        default=None,
        help="Path to model_latest.pth to refine poses with a fixed SDF.",
    )
    parser.add_argument(
        "--pose_only_cfg",
        type=str,
        default=None,
        help="Path to config.yml used to train the SDF (sc_factor/translation).",
    )
    parser.add_argument(
        "--pose_only_steps",
        type=int,
        default=50,
        help="Pose-only optimization steps.",
    )
    parser.add_argument(
        "--pose_only_n_rand",
        type=int,
        default=None,
        help="Override N_rand for pose-only optimization.",
    )
    args = parser.parse_args()
    run_one_video_tracking(
        video_dir=args.video_dir,
        out_folder=args.out_folder,
        use_segmenter=bool(args.use_segmenter),
        use_gui=bool(args.use_gui),
        debug_level=args.debug_level,
        stride=args.stride,
        overwrite=bool(args.overwrite),
        pose_only_ckpt=args.pose_only_ckpt,
        pose_only_cfg=args.pose_only_cfg,
        pose_only_steps=args.pose_only_steps,
        pose_only_n_rand=args.pose_only_n_rand,
    )
