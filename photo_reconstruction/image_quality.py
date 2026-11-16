"""
photo_reconstruction/image_quality.py — Automated Image Quality Assessment
===========================================================================

Evaluates uploaded smartphone photographs for:
  1. Motion and defocus blur (Laplacian operator variance).
  2. Over/under exposure and dynamic range clipping.
  3. Visual texture richness and feature detectability (SIFT/ORB keypoints).
Outputs composite quality scores (0-100) and actionable feedback for the user.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


class ImageQualityAssessor:
    """Automated photogrammetry suitability analyzer."""

    BLUR_THRESHOLD_GOOD = 100.0
    BLUR_THRESHOLD_WARN = 50.0

    EXPOSURE_MIN_MEAN = 45.0
    EXPOSURE_MAX_MEAN = 210.0
    MAX_CLIPPING_RATIO = 0.08  # Max 8% clipped pixels

    MIN_KEYPOINTS_GOOD = 800
    MIN_KEYPOINTS_WARN = 300

    @classmethod
    def evaluate(cls, rgb_image: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate single photograph for 3D reconstruction quality.

        Returns
        -------
        Dict with keys:
          - quality_score: float in [0.0, 100.0]
          - status: 'EXCELLENT', 'GOOD', 'WARNING', or 'REJECT'
          - blur_score: float (Laplacian variance)
          - is_blurry: bool
          - exposure_mean: float [0, 255]
          - clipping_ratio: float [0, 1]
          - keypoint_count: int
          - recommendations: List[str]
        """
        H, W = rgb_image.shape[:2]
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or H < 2 or W < 2:
            raise ValueError("Image quality assessment expects an RGB image with at least 2x2 pixels.")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        recommendations = []

        # 1. Blur Detection (Laplacian Variance)
        # Bound the longest side, not the width: portrait images must receive
        # the same analysis scale as landscape images.
        target_max_dimension = 1280
        scale = min(1.0, target_max_dimension / float(max(H, W)))
        if scale < 1.0:
            analysis_width = max(2, int(round(W * scale)))
            analysis_height = max(2, int(round(H * scale)))
            gray_scaled = cv2.resize(
                gray,
                (analysis_width, analysis_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray_scaled = gray
            analysis_height, analysis_width = gray_scaled.shape[:2]

        laplacian = cv2.Laplacian(gray_scaled, cv2.CV_64F)
        blur_score = float(laplacian.var())

        if blur_score >= cls.BLUR_THRESHOLD_GOOD:
            blur_status = "SHARP"
            blur_subscore = 100.0
        elif blur_score >= cls.BLUR_THRESHOLD_WARN:
            blur_status = "MODERATE_BLUR"
            blur_subscore = 65.0
            recommendations.append("Slight motion/lens blur detected. Hold phone steady while capturing.")
        else:
            blur_status = "BLURRY"
            blur_subscore = 25.0
            recommendations.append("Significant blur detected. Image may fail camera pose registration.")

        # 2. Exposure & Dynamic Range Analysis
        mean_exp = float(np.mean(gray))
        under_exposed = np.sum(gray < 10) / float(gray.size)
        over_exposed = np.sum(gray > 245) / float(gray.size)
        clipping_ratio = float(under_exposed + over_exposed)

        exp_subscore = 100.0
        if mean_exp < cls.EXPOSURE_MIN_MEAN:
            exp_subscore -= 40.0 * (1.0 - mean_exp / cls.EXPOSURE_MIN_MEAN)
            recommendations.append("Image is underexposed. Improve environmental lighting.")
        elif mean_exp > cls.EXPOSURE_MAX_MEAN:
            exp_subscore -= 40.0 * ((mean_exp - cls.EXPOSURE_MAX_MEAN) / (255.0 - cls.EXPOSURE_MAX_MEAN))
            recommendations.append("Image is overexposed/washed out. Reduce direct glare.")

        if clipping_ratio > cls.MAX_CLIPPING_RATIO:
            exp_subscore -= 20.0
            recommendations.append(f"High pixel clipping ({clipping_ratio * 100:.1f}%). Avoid harsh backlight.")

        exp_subscore = max(10.0, exp_subscore)

        # 3. Texture Richness (Feature Detectability via FAST/ORB)
        detector = cv2.FastFeatureDetector_create(threshold=25)
        kps = detector.detect(gray_scaled)
        kp_count = len(kps)
        analysis_pixels = max(1, analysis_width * analysis_height)
        reference_pixels = 1280 * 720
        normalized_kp_count = float(kp_count * reference_pixels / analysis_pixels)

        if normalized_kp_count >= cls.MIN_KEYPOINTS_GOOD:
            feat_subscore = 100.0
        elif normalized_kp_count >= cls.MIN_KEYPOINTS_WARN:
            feat_subscore = 60.0
            recommendations.append("Moderate feature density. Ensure surfaces contain visible visual texture.")
        else:
            feat_subscore = 30.0
            recommendations.append("Low texture density. Blank or featureless walls hinder geometric matching.")

        # Composite Quality Score (Weighted: 45% blur, 30% exposure, 25% features)
        quality_score = float(0.45 * blur_subscore + 0.30 * exp_subscore + 0.25 * feat_subscore)
        quality_score = np.clip(quality_score, 0.0, 100.0)

        if quality_score >= 80.0:
            status = "EXCELLENT"
        elif quality_score >= 65.0:
            status = "GOOD"
        elif quality_score >= 45.0:
            status = "WARNING"
        else:
            status = "REJECT"

        return {
            "quality_score": round(quality_score, 1),
            "status": status,
            "blur_score": round(blur_score, 1),
            "is_blurry": bool(blur_score < cls.BLUR_THRESHOLD_WARN),
            "exposure_mean": round(mean_exp, 1),
            "clipping_ratio": round(clipping_ratio, 3),
            "keypoint_count": kp_count,
            "normalized_keypoint_count": round(normalized_kp_count, 1),
            "analysis_dimensions": {
                "width": analysis_width,
                "height": analysis_height,
                "max_dimension": max(analysis_width, analysis_height),
                "scale_from_input": round(scale, 4),
            },
            "recommendations": recommendations,
        }

