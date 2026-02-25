import numpy as np


def _rotation_distance_deg(cam_in_ob_a, cam_in_ob_b):
    """Rotation distance in degrees between two camera-to-object poses."""
    ra = cam_in_ob_a[:3, :3]
    rb = cam_in_ob_b[:3, :3]
    cos_theta = (np.trace(ra.T @ rb) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


class CompletionBuffer:
    """
    Buffer for online completion:
      - keeps real/synthetic RGB-D-mask observations
      - picks diverse real frames
      - appends all synthetic frames
      - triggers global NeRF rebuild through BundleSdf.run_global_nerf(frame_payload=...)
    """

    def __init__(self, max_pool_size=20):
        self.max_pool_size = int(max_pool_size)

        self.Ks = []
        self.rgbs = []
        self.depths = []
        self.masks = []
        self.occ_masks = []
        self.cam_in_obs = []
        self.is_real_images = []
        self.frame_ids = []

        self.latest_mesh = None
        self.pool_changed = False
        self.last_rebuild_frame = -1

    def _append(
        self,
        K,
        rgb,
        depth,
        mask,
        occ_mask,
        cam_in_ob,
        is_real_image,
        frame_id,
    ):
        self.Ks.append(np.asarray(K).copy())
        self.rgbs.append(np.asarray(rgb).copy())
        self.depths.append(np.asarray(depth).copy())
        self.masks.append(np.asarray(mask).copy())
        if occ_mask is None:
            self.occ_masks.append(None)
        else:
            self.occ_masks.append(np.asarray(occ_mask).copy())
        self.cam_in_obs.append(np.asarray(cam_in_ob).copy())
        self.is_real_images.append(bool(is_real_image))
        self.frame_ids.append(str(frame_id))
        self.pool_changed = True

    def append_real(self, K, rgb, depth, mask, occ_mask, cam_in_ob, frame_id):
        self._append(
            K=K,
            rgb=rgb,
            depth=depth,
            mask=mask,
            occ_mask=occ_mask,
            cam_in_ob=cam_in_ob,
            is_real_image=True,
            frame_id=frame_id,
        )

    def append_synthetic(self, K, rgb, depth, mask, occ_mask, cam_in_ob, frame_id):
        self._append(
            K=K,
            rgb=rgb,
            depth=depth,
            mask=mask,
            occ_mask=occ_mask,
            cam_in_ob=cam_in_ob,
            is_real_image=False,
            frame_id=frame_id,
        )

    def _get_real_indices(self):
        return [i for i, flag in enumerate(self.is_real_images) if flag]

    def _get_synthetic_indices(self):
        return [i for i, flag in enumerate(self.is_real_images) if not flag]

    def real_count(self):
        return len(self._get_real_indices())

    def synthetic_count(self):
        return len(self._get_synthetic_indices())

    def replace_synthetic(self, synthetic_entries):
        keep_ids = self._get_real_indices()
        self._keep_only(keep_ids)
        for entry in synthetic_entries:
            self.append_synthetic(
                K=entry["K"],
                rgb=entry["rgb"],
                depth=entry["depth"],
                mask=entry["mask"],
                occ_mask=entry["occ_mask"],
                cam_in_ob=entry["cam_in_ob"],
                frame_id=entry["frame_id"],
            )
        self.pool_changed = True

    def _keep_only(self, keep_ids):
        self.Ks = [self.Ks[i] for i in keep_ids]
        self.rgbs = [self.rgbs[i] for i in keep_ids]
        self.depths = [self.depths[i] for i in keep_ids]
        self.masks = [self.masks[i] for i in keep_ids]
        self.occ_masks = [self.occ_masks[i] for i in keep_ids]
        self.cam_in_obs = [self.cam_in_obs[i] for i in keep_ids]
        self.is_real_images = [self.is_real_images[i] for i in keep_ids]
        self.frame_ids = [self.frame_ids[i] for i in keep_ids]

    def _select_diverse_real_indices(self):
        real_ids = self._get_real_indices()
        if len(real_ids) <= self.max_pool_size:
            return real_ids

        selected = [real_ids[0]]
        if real_ids[-1] != real_ids[0]:
            selected.append(real_ids[-1])

        candidates = [idx for idx in real_ids if idx not in selected]
        while len(selected) < self.max_pool_size and len(candidates) > 0:
            best_id = -1
            best_score = -1.0
            for cand in candidates:
                cand_pose = self.cam_in_obs[cand]
                min_dist = min(
                    _rotation_distance_deg(cand_pose, self.cam_in_obs[s])
                    for s in selected
                )
                if min_dist > best_score:
                    best_score = min_dist
                    best_id = cand
            selected.append(best_id)
            candidates = [idx for idx in candidates if idx != best_id]

        selected = sorted(selected)
        return selected

    def build_payload(self):
        picked_real = self._select_diverse_real_indices()
        picked_synth = self._get_synthetic_indices()
        picked = picked_real + picked_synth
        if len(picked) == 0:
            return None, {}

        occ_masks = []
        has_any_occ = False
        for i in picked:
            occ = self.occ_masks[i]
            if occ is None:
                occ_masks.append(np.zeros_like(self.masks[i], dtype=np.uint8))
            else:
                has_any_occ = True
                occ_masks.append(np.asarray(occ))

        payload = {
            "K": self.Ks[picked[0]],
            "Ks": np.asarray([self.Ks[i] for i in picked]),
            "rgbs": np.asarray([self.rgbs[i] for i in picked]),
            "depths": np.asarray([self.depths[i] for i in picked]),
            "masks": np.asarray([self.masks[i] for i in picked]),
            "occ_masks": np.asarray(occ_masks) if has_any_occ else None,
            "cam_in_obs": np.asarray([self.cam_in_obs[i] for i in picked]),
            "poses": np.asarray([self.cam_in_obs[i] for i in picked]),
            "pose_inits": np.asarray([self.cam_in_obs[i] for i in picked]),
            "normal_maps": None,
            "is_real_image": np.asarray([self.is_real_images[i] for i in picked]),
            "is_keyframe": np.ones((len(picked)), dtype=np.uint8),
            "frame_ids": [self.frame_ids[i] for i in picked],
        }
        metadata = {
            "picked_ids": picked,
            "picked_frame_ids": [self.frame_ids[i] for i in picked],
            "picked_real_count": len(picked_real),
            "picked_synth_count": len(picked_synth),
            "total_real_count": len(self._get_real_indices()),
            "total_synth_count": len(self._get_synthetic_indices()),
        }
        return payload, metadata

    def build_sdf(
        self,
        tracker,
        checkpoint_path=None,
        interpolate_missing_vertices=False,
        tex_res=1024,
        optimize_poses_override=None,
        synthetic_frame_weight=0.35,
        synthetic_rgb_weight=0.1,
        synthetic_only_uncovered_texels=True,
    ):
        del checkpoint_path  # Kept for interface compatibility.

        payload, metadata = self.build_payload()
        if payload is None:
            return None, False, metadata

        if (not self.pool_changed) and (self.latest_mesh is not None):
            return self.latest_mesh, False, metadata

        old_optimize_poses = tracker.cfg_nerf.get("optimize_poses", None)
        if optimize_poses_override is not None:
            tracker.cfg_nerf["optimize_poses"] = int(optimize_poses_override)
        try:
            mesh, optimized_cvcam_in_obs, offset = tracker.run_global_nerf(
                frame_payload=payload,
                get_texture=True,
                tex_res=tex_res,
                use_all_frames=False,
                interpolate_missing_vertices=interpolate_missing_vertices,
                return_mesh=True,
                synthetic_frame_weight=synthetic_frame_weight,
                synthetic_rgb_weight=synthetic_rgb_weight,
                synthetic_only_uncovered_texels=synthetic_only_uncovered_texels,
            )
        finally:
            if old_optimize_poses is not None:
                tracker.cfg_nerf["optimize_poses"] = old_optimize_poses

        pose_writeback_real_count = 0
        pose_delta_real_rot_deg = []
        pose_delta_real_trans_m = []
        if optimized_cvcam_in_obs is not None:
            optimized_cvcam_in_obs = np.asarray(optimized_cvcam_in_obs)
            picked_ids = metadata.get("picked_ids", [])
            if (
                optimized_cvcam_in_obs.ndim == 3
                and optimized_cvcam_in_obs.shape[-2:] == (4, 4)
                and len(picked_ids) == len(optimized_cvcam_in_obs)
            ):
                for payload_idx, pool_idx in enumerate(picked_ids):
                    # Only update real-frame poses. Synthetic poses must stay
                    # consistent with their rendered RGB-D observations.
                    if bool(self.is_real_images[pool_idx]):
                        old_pose = self.cam_in_obs[pool_idx]
                        new_pose = optimized_cvcam_in_obs[payload_idx]
                        pose_delta_real_rot_deg.append(
                            _rotation_distance_deg(old_pose, new_pose)
                        )
                        pose_delta_real_trans_m.append(
                            float(
                                np.linalg.norm(
                                    old_pose[:3, 3].astype(np.float64)
                                    - new_pose[:3, 3].astype(np.float64)
                                )
                            )
                        )
                        self.cam_in_obs[pool_idx] = (
                            new_pose.astype(np.float32).copy()
                        )
                        pose_writeback_real_count += 1

        self.latest_mesh = mesh
        self.pool_changed = False
        metadata.update(
            {
                "optimized_pose_count": int(
                    0
                    if optimized_cvcam_in_obs is None
                    else len(optimized_cvcam_in_obs)
                ),
                "pose_writeback_real_count": int(pose_writeback_real_count),
                "pose_delta_real_rot_deg_mean": float(
                    0.0 if len(pose_delta_real_rot_deg) == 0 else np.mean(pose_delta_real_rot_deg)
                ),
                "pose_delta_real_rot_deg_max": float(
                    0.0 if len(pose_delta_real_rot_deg) == 0 else np.max(pose_delta_real_rot_deg)
                ),
                "pose_delta_real_trans_m_mean": float(
                    0.0 if len(pose_delta_real_trans_m) == 0 else np.mean(pose_delta_real_trans_m)
                ),
                "pose_delta_real_trans_m_max": float(
                    0.0 if len(pose_delta_real_trans_m) == 0 else np.max(pose_delta_real_trans_m)
                ),
                "has_offset": bool(offset is not None),
            }
        )
        return mesh, True, metadata
