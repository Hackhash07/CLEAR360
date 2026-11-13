"""
CLEAR360 Depth Accuracy Metrics
Compare AI-estimated depth to ground-truth Blender depth.

Standard monocular depth estimation benchmark metrics:
  - Absolute Relative Error (AbsRel):  mean(|d_pred - d_gt| / d_gt)
  - Root Mean Squared Error (RMSE):    sqrt(mean((d_pred - d_gt)^2))
  - Root Mean Squared Log Error:       sqrt(mean((log(d_pred) - log(d_gt))^2))
  - Threshold Accuracy (delta < thr):  fraction of pixels satisfying max(d_pred/d_gt, d_gt/d_pred) < thr
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_valid_mask(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Build a boolean mask of finite, positive pixels to prevent division by zero."""
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0.0) & (pred > 0.0)
    if mask is not None:
        valid = valid & mask.astype(bool)
    return valid


def abs_rel(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Compute Absolute Relative Error (AbsRel).

    Formula:
        AbsRel = (1 / |V|) * sum_{i in V} (|pred_i - gt_i| / gt_i)

    Parameters
    ----------
    pred : np.ndarray
        Predicted depth map.
    gt : np.ndarray
        Ground-truth depth map.
    mask : np.ndarray, optional
        Optional boolean mask selecting regions of interest.

    Returns
    -------
    float
        AbsRel error. Returns 0.0 if no valid pixels exist.
    """
    valid = _get_valid_mask(pred, gt, mask)
    if not np.any(valid):
        return 0.0
    p = pred[valid]
    g = gt[valid]
    return float(np.mean(np.abs(p - g) / g))


def rmse(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Compute Root Mean Squared Error (RMSE) in metres.

    Formula:
        RMSE = sqrt((1 / |V|) * sum_{i in V} (pred_i - gt_i)^2)

    Parameters
    ----------
    pred : np.ndarray
        Predicted depth map in metres.
    gt : np.ndarray
        Ground-truth depth map in metres.
    mask : np.ndarray, optional
        Optional boolean mask selecting regions of interest.

    Returns
    -------
    float
        RMSE in metres. Returns 0.0 if no valid pixels exist.
    """
    valid = _get_valid_mask(pred, gt, mask)
    if not np.any(valid):
        return 0.0
    p = pred[valid]
    g = gt[valid]
    return float(np.sqrt(np.mean((p - g) ** 2)))


def rmse_log(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Compute Root Mean Squared Error of log depths (RMSE log).

    Formula:
        RMSE_log = sqrt((1 / |V|) * sum_{i in V} (log(pred_i) - log(gt_i))^2)

    Parameters
    ----------
    pred : np.ndarray
        Predicted depth map.
    gt : np.ndarray
        Ground-truth depth map.
    mask : np.ndarray, optional
        Optional boolean mask selecting regions of interest.

    Returns
    -------
    float
        Log RMSE. Returns 0.0 if no valid pixels exist.
    """
    valid = _get_valid_mask(pred, gt, mask)
    if not np.any(valid):
        return 0.0
    p = pred[valid]
    g = gt[valid]
    return float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))


def threshold_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 1.25,
) -> float:
    """Compute Threshold Accuracy (fraction of pixels with ratio < threshold).

    Formula:
        delta = max(pred / gt, gt / pred)
        Accuracy = (1 / |V|) * sum_{i in V} [delta_i < threshold]

    Parameters
    ----------
    pred : np.ndarray
        Predicted depth map.
    gt : np.ndarray
        Ground-truth depth map.
    mask : np.ndarray, optional
        Optional boolean mask selecting regions of interest.
    threshold : float, optional
        Accuracy ratio threshold (default 1.25).

    Returns
    -------
    float
        Fraction of valid pixels satisfying the threshold condition (between 0.0 and 1.0).
    """
    valid = _get_valid_mask(pred, gt, mask)
    if not np.any(valid):
        return 0.0
    p = pred[valid]
    g = gt[valid]
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < threshold))


def compute_all_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Compute all standard depth evaluation metrics in a single pass.

    Metrics computed:
      - abs_rel: Mean absolute relative error
      - rmse: Root mean squared error in metres
      - rmse_log: Root mean squared log error
      - delta_1: Percentage of pixels where delta < 1.25
      - delta_2: Percentage of pixels where delta < 1.25^2 (1.5625)
      - delta_3: Percentage of pixels where delta < 1.25^3 (1.953125)

    Parameters
    ----------
    pred : np.ndarray
        Predicted depth map.
    gt : np.ndarray
        Ground-truth depth map.
    mask : np.ndarray, optional
        Optional boolean mask.

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to their computed values.
    """
    valid = _get_valid_mask(pred, gt, mask)
    if not np.any(valid):
        return {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
            "delta_1": 0.0,
            "delta_2": 0.0,
            "delta_3": 0.0,
        }

    p = pred[valid]
    g = gt[valid]

    abs_rel_val = float(np.mean(np.abs(p - g) / g))
    rmse_val = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log_val = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))

    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < (1.25**2)))
    d3 = float(np.mean(ratio < (1.25**3)))

    return {
        "abs_rel": abs_rel_val,
        "rmse": rmse_val,
        "rmse_log": rmse_log_val,
        "delta_1": d1,
        "delta_2": d2,
        "delta_3": d3,
    }


def per_camera_report(
    predictions: dict[str, np.ndarray],
    ground_truths: dict[str, np.ndarray],
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Generate per-camera depth accuracy metrics dictionary.

    Parameters
    ----------
    predictions : dict[str, np.ndarray]
        Mapping from camera name (e.g. 'front', 'rear') to predicted depth maps.
    ground_truths : dict[str, np.ndarray]
        Mapping from camera name to ground-truth depth maps.
    masks : dict[str, np.ndarray], optional
        Optional mapping from camera name to boolean masks.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary of metrics per camera, plus an 'overall' summary average.
    """
    report: dict[str, dict[str, float]] = {}

    for cam_name, gt in ground_truths.items():
        if cam_name not in predictions:
            continue
        pred = predictions[cam_name]
        mask = masks.get(cam_name) if masks is not None else None
        report[cam_name] = compute_all_metrics(pred, gt, mask)

    # Compute overall macro average if at least one camera was evaluated
    if report:
        metric_keys = ["abs_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3"]
        overall: dict[str, float] = {}
        for k in metric_keys:
            vals = [m[k] for m in report.values() if k in m]
            overall[k] = float(np.mean(vals)) if vals else 0.0
        report["overall"] = overall

    return report
