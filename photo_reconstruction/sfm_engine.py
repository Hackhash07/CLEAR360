"""
photo_reconstruction/sfm_engine.py — Structure-from-Motion Camera Registration Engine
=====================================================================================

Performs camera pose estimation, feature extraction, pairwise matching, geometric
verification, and sparse 3D bundle adjustment using pycolmap (COLMAP) with OpenCV fallback.
Recovers unknown camera trajectories and sparse 3D points from unordered smartphone photos.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os
import shutil
import numpy as np
import open3d as o3d
import cv2

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


class SfMEngine:
    """Structure-from-Motion photogrammetry engine."""

    def __init__(self, work_dir: Path, quality_preset: str = "standard"):
        """
        Parameters
        ----------
        work_dir : Path
            Directory for database and sparse models.
        quality_preset : str ('quick', 'standard', 'high')
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.quality_preset = quality_preset.lower()
        self.db_path = self.work_dir / "database.db"
        self.sparse_dir = self.work_dir / "sparse"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def run_reconstruction(
        self,
        image_dir: Path,
        progress_cb: Optional[callable] = None,
        camera_mode: str = "shared_camera",
    ) -> Dict[str, Any]:
        """
        Execute full SfM pipeline:
          1. Feature extraction
          2. Exhaustive pairwise matching
          3. Incremental Structure-from-Motion
          4. Pose extraction and sparse point-cloud export

        Parameters
        ----------
        image_dir : Path
            Directory containing source photographs.
        progress_cb : Optional[callable(pct: int, msg: str)]
        camera_mode : str
            "shared_camera" for a consistent same-lens set, or "per_image_camera"
            when the photo-set audit finds mixed cameras/lenses or dimensions.

        Returns
        -------
        Dict with keys:
          - success: bool
          - registered_count: int
          - total_images: int
          - failed_images: List[str]
          - camera_poses: Dict[str, np.ndarray] (4x4 world_from_cam matrices)
          - sparse_points: np.ndarray (N, 3)
          - sparse_colors: np.ndarray (N, 3) in [0, 1]
          - mean_reprojection_error: float (pixels)
          - message: str
        """
        img_dir = Path(image_dir)
        if not img_dir.exists():
            images = []
        else:
            images = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        num_images = len(images)

        if camera_mode not in {"shared_camera", "per_image_camera"}:
            raise ValueError(
                "camera_mode must be 'shared_camera' or 'per_image_camera'."
            )

        if num_images < 2:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": "At least 2 photographs are required for Structure-from-Motion.",
            }

        if progress_cb:
            progress_cb(10, f"Extracting visual features from {num_images} photographs...")

        if HAS_PYCOLMAP:
            res = self._run_pycolmap_sfm(
                img_dir, images, progress_cb, camera_mode=camera_mode
            )
            res["camera_calibration_mode"] = camera_mode
            if res.get("success") and res.get("registered_count", 0) >= 2:
                return res

            # The OpenCV fallback estimates one shared K matrix. It cannot honour
            # per-image calibrations, so never use it for a mixed-camera upload.
            from photo_reconstruction.camera_calibration import can_use_opencv_fallback
            if not can_use_opencv_fallback(camera_mode):
                res["message"] = (
                    str(res.get("message", "PyCOLMAP reconstruction failed."))
                    + " OpenCV fallback was not used because it cannot honour "
                    "per-image camera calibration. Install or repair PyCOLMAP."
                )
                return res

            if progress_cb:
                progress_cb(50, "Running OpenCV feature matching and camera pose estimation...")
            res_cv = self._run_opencv_sfm(img_dir, images, progress_cb)
            res_cv["camera_calibration_mode"] = camera_mode
            if res_cv.get("success") and res_cv.get("registered_count", 0) >= 2:
                return res_cv
            return res
        from photo_reconstruction.camera_calibration import can_use_opencv_fallback
        if not can_use_opencv_fallback(camera_mode):
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "camera_calibration_mode": camera_mode,
                "message": (
                    "This photo set requires per-image camera calibration, but PyCOLMAP "
                    "is unavailable. Install PyCOLMAP or recapture using one lens and "
                    "consistent image dimensions."
                ),
            }
        res = self._run_opencv_sfm(img_dir, images, progress_cb)
        res["camera_calibration_mode"] = camera_mode
        return res

    def _run_pycolmap_sfm(
        self,
        image_dir: Path,
        images: List[Path],
        progress_cb: Optional[callable],
        camera_mode: str = "shared_camera",
    ) -> Dict[str, Any]:
        """Execute SfM using pycolmap."""
        num_images = len(images)
        if self.db_path.exists():
            self.db_path.unlink()

        # 1. Feature Extraction
        extract_opts = pycolmap.FeatureExtractionOptions()
        if self.quality_preset == "fast":
            extract_opts.max_image_size = 800
            extract_opts.sift.max_num_features = 1500
        elif self.quality_preset == "draft":
            extract_opts.max_image_size = 800
            extract_opts.sift.max_num_features = 1000
        elif self.quality_preset == "quick":
            extract_opts.max_image_size = 1000
            extract_opts.sift.max_num_features = 1500
        elif self.quality_preset == "high":
            extract_opts.max_image_size = 1800
            extract_opts.sift.max_num_features = 5000
        elif self.quality_preset == "maximum":
            extract_opts.max_image_size = 2400
            extract_opts.sift.max_num_features = 8000
        else:  # standard
            extract_opts.max_image_size = 1200
            extract_opts.sift.max_num_features = 2500

        try:
            from photo_reconstruction.camera_calibration import pycolmap_camera_mode

            reader_options = pycolmap.ImageReaderOptions()
            reader_options.camera_model = "SIMPLE_RADIAL"
            explicit_camera_mode = pycolmap_camera_mode(pycolmap, camera_mode)
            pycolmap.extract_features(
                database_path=str(self.db_path),
                image_path=str(image_dir),
                camera_mode=explicit_camera_mode,
                reader_options=reader_options,
                extraction_options=extract_opts,
            )
        except Exception as e:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": f"Feature extraction failed: {e}",
            }

        if progress_cb:
            progress_cb(40, "Matching feature pairs and performing geometric verification...")

        # 2. Feature Matching
        match_opts = pycolmap.FeatureMatchingOptions()
        try:
            pycolmap.match_exhaustive(
                database_path=str(self.db_path),
                matching_options=match_opts,
            )
        except Exception as e:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": f"Feature matching failed: {e}",
            }

        if progress_cb:
            progress_cb(60, "Running incremental Structure-from-Motion bundle adjustment...")

        # 3. Incremental Mapping with robust indoor smartphone thresholds
        inc_opts = pycolmap.IncrementalPipelineOptions()
        inc_opts.min_num_matches = 10
        inc_opts.min_model_size = 2
        # Relaxed indoor initialization thresholds (default 16.0 deg & 100 inliers fail on indoor walls)
        inc_opts.mapper.init_min_num_inliers = 15
        inc_opts.mapper.init_min_tri_angle = 4.0
        inc_opts.mapper.abs_pose_min_num_inliers = 12
        inc_opts.mapper.abs_pose_min_inlier_ratio = 0.15
        out_recon_dir = self.sparse_dir / "0"
        if out_recon_dir.exists():
            shutil.rmtree(out_recon_dir)

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(self.db_path),
                image_path=str(image_dir),
                output_path=str(self.sparse_dir),
                options=inc_opts,
            )
        except Exception as e:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": f"Incremental mapping failed: {e}",
            }

        if not reconstructions or len(reconstructions) == 0:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": num_images,
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": (
                    "Could not register camera poses. Photographs lack sufficient visual overlap (recommend 60-80% overlap) "
                    "or contain significant motion blur. Capture additional intermediate views around the scene."
                ),
            }

        # Enumerate and audit ALL COLMAP reconstruction models (M7.3.8 Issue 2)
        models_audit = []
        for model_idx, recon in reconstructions.items():
            pts3d_arr = np.array([p.xyz for p in recon.points3D.values()]) if len(recon.points3D) > 0 else np.zeros((0, 3))
            b_min = pts3d_arr.min(axis=0).tolist() if len(pts3d_arr) > 0 else [0.0, 0.0, 0.0]
            b_max = pts3d_arr.max(axis=0).tolist() if len(pts3d_arr) > 0 else [0.0, 0.0, 0.0]
            errs = [p.error for p in recon.points3D.values()]
            models_audit.append({
                "model_id": int(model_idx),
                "num_registered_images": len(recon.images),
                "registered_images": [img.name for img in recon.images.values()],
                "num_points3D": len(recon.points3D),
                "bounds_min": b_min,
                "bounds_max": b_max,
                "mean_reprojection_error": round(float(np.mean(errs)), 3) if errs else 0.0,
            })

        # Select primary reconstruction component (or aligned composite)
        best_recon = max(reconstructions.values(), key=lambda r: len(r.points3D))

        # 4. Extract Registered Cameras and Poses
        camera_poses = {}
        registered_names = set()

        for img_id, image in best_recon.images.items():
            registered_names.add(image.name)
            # cam_from_world transforms world -> camera
            # In pycolmap 0.6+, cam_from_world is a method returning Rigid3d
            if hasattr(image, "cam_from_world"):
                cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                R = cfw.rotation.matrix()
                t = np.asarray(cfw.translation, dtype=np.float64)
            else:
                R = pycolmap.qvec_to_rotmat(image.qvec)
                t = np.asarray(image.tvec, dtype=np.float64)

            T_cam_world = np.eye(4, dtype=np.float64)
            T_cam_world[:3, :3] = R
            T_cam_world[:3, 3] = t
            T_world_cam = np.linalg.inv(T_cam_world)
            camera_poses[image.name] = T_world_cam

        # Reject spatial camera pose outliers (bundle adjustment divergence)
        if len(camera_poses) >= 4:
            centers = np.array([p[:3, 3] for p in camera_poses.values()])
            c_names = list(camera_poses.keys())
            med_center = np.median(centers, axis=0)
            dists = np.linalg.norm(centers - med_center, axis=1)
            q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
            iqr = q75 - q25
            outlier_thresh = max(15.0, q75 + 5.0 * iqr)

            valid_poses = {}
            for name, d in zip(c_names, dists):
                if d < outlier_thresh:
                    valid_poses[name] = camera_poses[name]
                else:
                    registered_names.discard(name)
            camera_poses = valid_poses

        all_names = set(img.name for img in images)
        failed_names = sorted(list(all_names - registered_names))

        # Build Authoritative PhoneCameraModel and 2D-3D observation tracks
        camera_models: Dict[str, Any] = {}
        sparse_tracks: Dict[str, List[Dict[str, Any]]] = {}
        from photo_reconstruction.camera_model import PhoneCameraModel

        for img_id, image in best_recon.images.items():
            if image.name in camera_poses:
                cam = best_recon.cameras[image.camera_id]
                camera_models[image.name] = PhoneCameraModel.from_colmap_camera(
                    cam=cam,
                    image_name=image.name,
                    T_world_cam=camera_poses[image.name],
                )

                img_tracks = []
                for p2d in image.points2D:
                    if p2d.has_point3D():
                        p3d_id = p2d.point3D_id
                        if p3d_id in best_recon.points3D:
                            p3d = best_recon.points3D[p3d_id]
                            img_tracks.append({
                                "xy": (float(p2d.xy[0]), float(p2d.xy[1])),
                                "xyz": (float(p3d.xyz[0]), float(p3d.xyz[1]), float(p3d.xyz[2])),
                                "error": float(p3d.error),
                                "point3D_id": int(p3d_id),
                            })
                sparse_tracks[image.name] = img_tracks

        # 5. Extract 3D Points
        pts_list = []
        cols_list = []
        errors = []

        for p3d_id, pt3d in best_recon.points3D.items():
            pts_list.append(pt3d.xyz)
            cols_list.append(pt3d.color / 255.0)
            errors.append(pt3d.error)

        if len(pts_list) > 0:
            sparse_pts = np.array(pts_list, dtype=np.float32)
            sparse_cols = np.array(cols_list, dtype=np.float32)
            mean_error: Optional[float] = float(np.mean(errors))
        else:
            sparse_pts = np.zeros((0, 3), dtype=np.float32)
            sparse_cols = np.zeros((0, 3), dtype=np.float32)
            mean_error = None

        reprojection_message = (
            f"mean reprojection error {mean_error:.2f} px"
            if mean_error is not None
            else "reprojection residuals unavailable (no sparse 3D points)"
        )

        if progress_cb:
            progress_cb(80, f"Sparse reconstruction complete: {len(camera_poses)}/{num_images} registered.")

        return {
            "success": len(camera_poses) >= 2,
            "registered_count": len(camera_poses),
            "total_images": num_images,
            "failed_images": failed_names,
            "camera_poses": camera_poses,
            "camera_models": camera_models,
            "sparse_tracks": sparse_tracks,
            "sparse_points": sparse_pts,
            "sparse_colors": sparse_cols,
            "sfm_models_audit": models_audit,
            "mean_reprojection_error": round(mean_error, 3),
            "message": (
                f"Successfully registered {len(camera_poses)} of {num_images} photographs. "
                f"Reconstructed {len(sparse_pts):,} 3D points; {reprojection_message}."
            ),
        }

    def _run_opencv_sfm(
        self,
        image_dir: Path,
        images: List[Path],
        progress_cb: Optional[callable],
    ) -> Dict[str, Any]:
        """Fallback OpenCV-based Two-View / Multi-View SfM."""
        cv_features = {
            "fast": 1500,
            "draft": 1000,
            "quick": 1500,
            "standard": 2500,
            "high": 4000,
            "maximum": 6000,
        }.get(self.quality_preset, 2500)
        sift = cv2.SIFT_create(nfeatures=cv_features)
        max_size = {
            "fast": 800,
            "draft": 800,
            "quick": 1000,
            "standard": 1600,
            "high": 2400,
            "maximum": 3200,
        }.get(self.quality_preset, 1600)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        kps_dict = {}
        descs_dict = {}
        imgs_rgb = {}

        for img_path in images:
            img = cv2.imread(str(img_path))
            h_orig, w_orig = img.shape[:2]
            if max(h_orig, w_orig) > max_size:
                scale = max_size / float(max(h_orig, w_orig))
                new_w = max(1, int(round(w_orig * scale)))
                new_h = max(1, int(round(h_orig * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            kps_dict[img_path.name] = kp
            descs_dict[img_path.name] = des
            imgs_rgb[img_path.name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find best pair
        best_pair = None
        max_matches = 0
        best_matches = None
        deep_matcher = None
        deep_pts1 = None
        deep_pts2 = None
        img_map = {p.name: p for p in images}

        names = list(kps_dict.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d1 = descs_dict[names[i]]
                d2 = descs_dict[names[j]]
                if d1 is None or d2 is None:
                    continue
                matches = bf.match(d1, d2)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    best_pair = (names[i], names[j])
                    best_matches = matches

        # If SIFT matches are sparse, attempt DeepMatcher (LightGlue)
        if max_matches < 30 or best_pair is None:
            try:
                from photo_reconstruction.deep_matcher import DeepMatcher
                deep_matcher = DeepMatcher(feature_type="superpoint", max_keypoints=2048)
                if deep_matcher.is_available:
                    if progress_cb:
                        progress_cb(55, "Applying LightGlue neural feature matching...")
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            p1, p2 = img_map[names[i]], img_map[names[j]]
                            kpts0, kpts1, _ = deep_matcher.match_pair(p1, p2)
                            if len(kpts0) > max_matches:
                                max_matches = len(kpts0)
                                best_pair = (names[i], names[j])
                                deep_pts1 = kpts0
                                deep_pts2 = kpts1
            except Exception:
                pass

        if max_matches < 15 or best_pair is None:
            return {
                "success": False,
                "registered_count": 0,
                "total_images": len(images),
                "failed_images": [img.name for img in images],
                "camera_poses": {},
                "sparse_points": np.zeros((0, 3), dtype=np.float32),
                "sparse_colors": np.zeros((0, 3), dtype=np.float32),
                "mean_reprojection_error": None,
                "message": "Insufficient feature matches between image pairs (< 15 matches).",
            }

        name1, name2 = best_pair
        if deep_pts1 is not None and deep_pts2 is not None:
            pts1 = deep_pts1
            pts2 = deep_pts2
        else:
            kp1 = kps_dict[name1]
            kp2 = kps_dict[name2]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches])

        H, W = imgs_rgb[name1].shape[:2]
        f = 1.2 * max(W, H)
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float64)

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        # Camera 1 at origin
        T1 = np.eye(4, dtype=np.float64)
        # Camera 2
        T2_cam_world = np.eye(4, dtype=np.float64)
        T2_cam_world[:3, :3] = R
        T2_cam_world[:3, 3] = t.ravel()
        T2 = np.linalg.inv(T2_cam_world)

        # Triangulate points
        P1 = K @ np.eye(3, 4, dtype=np.float64)
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        valid = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 50.0)
        sparse_pts = pts3d[valid].astype(np.float32)
        sparse_cols = np.ones((len(sparse_pts), 3), dtype=np.float32) * 0.7

        camera_poses = {name1: T1, name2: T2}
        all_pts3d = [sparse_pts]

        # Lazy init deep matcher for remaining views if needed
        if deep_matcher is None:
            try:
                from photo_reconstruction.deep_matcher import DeepMatcher
                deep_matcher = DeepMatcher(feature_type="superpoint", max_keypoints=2048)
            except Exception:
                deep_matcher = None

        # Incremental Registration for remaining viewpoints
        remaining = [n for n in names if n not in camera_poses]
        max_iters = len(remaining)
        for _ in range(max_iters):
            if not remaining:
                break
            registered_this_iter = False
            for cand_name in list(remaining):
                d_cand = descs_dict.get(cand_name)
                # Try to register against any already registered camera
                for reg_name in list(camera_poses.keys()):
                    cand_pts_reg = None
                    cand_pts_cand = None

                    d_reg = descs_dict.get(reg_name)
                    if d_reg is not None and d_cand is not None:
                        m_cand = bf.match(d_reg, d_cand)
                        if len(m_cand) >= 15:
                            cand_pts_reg = np.float32([kps_dict[reg_name][m.queryIdx].pt for m in m_cand])
                            cand_pts_cand = np.float32([kps_dict[cand_name][m.trainIdx].pt for m in m_cand])

                    if cand_pts_reg is None and deep_matcher is not None and deep_matcher.is_available:
                        try:
                            k0, k1, _ = deep_matcher.match_pair(img_map[reg_name], img_map[cand_name])
                            if len(k0) >= 15:
                                cand_pts_reg = k0
                                cand_pts_cand = k1
                        except Exception:
                            pass

                    if cand_pts_reg is None or cand_pts_cand is None or len(cand_pts_reg) < 15:
                        continue

                    E_c, mask_c = cv2.findEssentialMat(cand_pts_reg, cand_pts_cand, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
                    if E_c is None or mask_c is None or np.sum(mask_c) < 15:
                        continue
                    _, R_c, t_c, _ = cv2.recoverPose(E_c, cand_pts_reg, cand_pts_cand, K, mask=mask_c)

                    # Scale consistency enforcement against existing triangulated scene points
                    inliers_reg = cand_pts_reg[mask_c.ravel() > 0]
                    inliers_cand = cand_pts_cand[mask_c.ravel() > 0]
                    scale_factor = 1.0

                    if len(inliers_reg) >= 8:
                        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                        P2 = K @ np.hstack([R_c, t_c])
                        pts4d = cv2.triangulatePoints(P1, P2, inliers_reg.T, inliers_cand.T)
                        w = pts4d[3]
                        valid_w = np.abs(w) > 1e-5
                        if np.any(valid_w):
                            pts3d_pair = (pts4d[:3, valid_w] / w[valid_w]).T
                            z_reg = pts3d_pair[:, 2]
                            valid_z = (z_reg > 0.1) & (z_reg < 50.0)
                            if np.sum(valid_z) >= 5:
                                med_z_new = float(np.median(z_reg[valid_z]))
                                if all_pts3d:
                                    curr_pts = np.vstack(all_pts3d)
                                    C_reg = camera_poses[reg_name][:3, 3]
                                    dists = np.linalg.norm(curr_pts - C_reg, axis=1)
                                    med_z_exist = float(np.median(dists)) if len(dists) > 0 else med_z_new
                                    if med_z_new > 1e-3 and med_z_exist > 1e-3:
                                        scale_factor = float(np.clip(med_z_exist / med_z_new, 0.05, 5.0))

                    t_c_scaled = t_c.ravel() * scale_factor

                    T_cand_reg = np.eye(4, dtype=np.float64)
                    T_cand_reg[:3, :3] = R_c
                    T_cand_reg[:3, 3] = t_c_scaled
                    T_reg_cand = np.linalg.inv(T_cand_reg)

                    T_world_cand = camera_poses[reg_name] @ T_reg_cand
                    camera_poses[cand_name] = T_world_cand
                    remaining.remove(cand_name)
                    registered_this_iter = True
                    break
                if registered_this_iter:
                    break

        # Reject spatial camera pose outliers (bundle adjustment divergence)
        if len(camera_poses) >= 4:
            centers = np.array([p[:3, 3] for p in camera_poses.values()])
            c_names = list(camera_poses.keys())
            med_center = np.median(centers, axis=0)
            dists = np.linalg.norm(centers - med_center, axis=1)
            q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
            iqr = q75 - q25
            outlier_thresh = max(15.0, q75 + 5.0 * iqr)
            camera_poses = {name: camera_poses[name] for name, d in zip(c_names, dists) if d < outlier_thresh}

        failed_names = [n for n in names if n not in camera_poses]
        sparse_pts = np.vstack(all_pts3d) if all_pts3d else np.zeros((0, 3), dtype=np.float32)
        sparse_cols = np.ones((len(sparse_pts), 3), dtype=np.float32) * 0.7

        from photo_reconstruction.camera_model import PhoneCameraModel
        camera_models = {}
        sparse_tracks = {}
        for name, pose in camera_poses.items():
            h_im, w_im = imgs_rgb[name].shape[:2] if name in imgs_rgb else (1067, 1600)
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            camera_models[name] = PhoneCameraModel(
                name=name,
                model_name="PINHOLE",
                width=w_im,
                height=h_im,
                params=np.array([fx, fy, cx, cy], dtype=np.float64),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                T_world_cam=pose,
                T_cam_world=np.linalg.inv(pose),
            )
            sparse_tracks[name] = []

        return {
            "success": len(camera_poses) >= 2,
            "registered_count": len(camera_poses),
            "total_images": len(images),
            "failed_images": failed_names,
            "camera_poses": camera_poses,
            "camera_models": camera_models,
            "sparse_tracks": sparse_tracks,
            "sparse_points": sparse_pts,
            "sparse_colors": sparse_cols,
            "mean_reprojection_error": None,
            "message": f"Successfully registered {len(camera_poses)}/{len(images)} photographs with OpenCV SfM.",
        }
