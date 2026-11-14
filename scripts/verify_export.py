#!/usr/bin/env python3
"""
CLEAR360 M1 Export Verifier
============================
Verifies that the Blender headless export completed successfully by checking:
  1. All expected RGB PNGs exist and are non-zero in size (error if missing).
  2. All expected depth EXRs exist (error if missing).
  3. All expected position EXRs exist (warning if missing, not error).
  4. All expected semantic PNGs exist (warning if missing, not error).
  5. Loads a sample frame (frame 0, front camera) and validates shapes and depth range.
  6. Loads cameras.yaml and validates all 4 cameras with expected intrinsics.
  7. Prints a colour-coded PASS / WARN / FAIL summary table.

Usage:
    python scripts/verify_export.py [--data-root DATA_ROOT]

DATA_ROOT defaults to the current working directory (project root).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ── Constants ─────────────────────────────────────────────────────────────────
CAMERAS = ["front", "rear", "left", "right"]
NUM_FRAMES = 20
EXPECTED_SHAPE_RGB = (720, 1280, 3)
EXPECTED_SHAPE_DEPTH = (720, 1280)

# Expected intrinsic values (from sim/constants.py CAMERA_INTRINSICS)
EXPECTED_FX = 369.4
EXPECTED_CX = 640.0
EXPECTED_CY = 360.0

# ANSI colour codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class CheckResult(NamedTuple):
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    detail: str


def _fmt_status(status: str) -> str:
    """Return ANSI-coloured status label."""
    if status == "PASS":
        return f"{GREEN}{BOLD}PASS{RESET}"
    if status == "WARN":
        return f"{YELLOW}{BOLD}WARN{RESET}"
    return f"{RED}{BOLD}FAIL{RESET}"


# ── File-existence checks ─────────────────────────────────────────────────────

def check_rgb_frames(data_root: Path) -> CheckResult:
    """Verify all 80 RGB PNGs (4 cameras × 20 frames) exist and are non-empty."""
    missing: list[str] = []
    empty: list[str] = []
    for cam in CAMERAS:
        for frame in range(NUM_FRAMES):
            p = data_root / "data" / "sim_frames" / cam / f"frame_{frame:04d}.png"
            if not p.exists():
                missing.append(str(p.relative_to(data_root)))
            elif p.stat().st_size == 0:
                empty.append(str(p.relative_to(data_root)))

    total = len(CAMERAS) * NUM_FRAMES
    if missing or empty:
        detail = ""
        if missing:
            detail += f"Missing ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}"
        if empty:
            detail += f" Zero-size ({len(empty)}): {empty[:3]}"
        return CheckResult("RGB frames", "FAIL", detail)
    return CheckResult("RGB frames", "PASS", f"All {total} PNGs present and non-empty")


def check_depth_exrs(data_root: Path) -> CheckResult:
    """Verify all 80 depth EXRs exist."""
    missing: list[str] = []
    for cam in CAMERAS:
        for frame in range(NUM_FRAMES):
            p = data_root / "data" / "gt_depth" / cam / f"frame_{frame:04d}.exr"
            if not p.exists():
                missing.append(str(p.relative_to(data_root)))

    total = len(CAMERAS) * NUM_FRAMES
    if missing:
        detail = f"Missing ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}"
        return CheckResult("Depth EXRs", "FAIL", detail)
    return CheckResult("Depth EXRs", "PASS", f"All {total} EXRs present")


def check_position_exrs(data_root: Path) -> CheckResult:
    """Verify all 80 position EXRs exist — warning only if missing."""
    missing: list[str] = []
    for cam in CAMERAS:
        for frame in range(NUM_FRAMES):
            p = data_root / "data" / "gt_position" / cam / f"frame_{frame:04d}.exr"
            if not p.exists():
                missing.append(str(p.relative_to(data_root)))

    total = len(CAMERAS) * NUM_FRAMES
    if missing:
        detail = f"Missing ({len(missing)}/{total}): {missing[:3]}{'...' if len(missing) > 3 else ''}"
        return CheckResult("Position EXRs", "WARN", detail)
    return CheckResult("Position EXRs", "PASS", f"All {total} EXRs present")


def check_semantic_pngs(data_root: Path) -> CheckResult:
    """Verify all 80 semantic PNGs exist — warning only if missing."""
    missing: list[str] = []
    for cam in CAMERAS:
        for frame in range(NUM_FRAMES):
            p = data_root / "data" / "gt_semantic" / cam / f"frame_{frame:04d}.png"
            if not p.exists():
                missing.append(str(p.relative_to(data_root)))

    total = len(CAMERAS) * NUM_FRAMES
    if missing:
        detail = f"Missing ({len(missing)}/{total}): {missing[:3]}{'...' if len(missing) > 3 else ''}"
        return CheckResult("Semantic PNGs", "WARN", detail)
    return CheckResult("Semantic PNGs", "PASS", f"All {total} PNGs present")


# ── Sample-frame content checks ───────────────────────────────────────────────

def check_sample_frame(data_root: Path) -> list[CheckResult]:
    """Load frame_0000 from the front camera and validate shapes and depth range."""
    results: list[CheckResult] = []

    # --- RGB shape ---
    rgb_path = data_root / "data" / "sim_frames" / "front" / "frame_0000.png"
    if not rgb_path.exists():
        results.append(CheckResult("Sample RGB shape", "FAIL", f"File not found: {rgb_path}"))
    else:
        try:
            if cv2 is None:
                raise ImportError("OpenCV cv2 not installed")
            img = cv2.imread(str(rgb_path))
            if img is None:
                results.append(CheckResult("Sample RGB shape", "FAIL", "cv2.imread returned None"))
            else:
                # OpenCV loads BGR; convert channel count to 3 for check
                shape = img.shape  # (H, W, C) or (H, W)
                if shape == EXPECTED_SHAPE_RGB:
                    results.append(
                        CheckResult("Sample RGB shape", "PASS", f"Shape {shape} ✓")
                    )
                else:
                    results.append(
                        CheckResult(
                            "Sample RGB shape",
                            "FAIL",
                            f"Expected {EXPECTED_SHAPE_RGB}, got {shape}",
                        )
                    )
        except ImportError:
            results.append(
                CheckResult("Sample RGB shape", "WARN", "opencv-python not installed; skipped")
            )

    # --- Depth EXR ---
    depth_path = data_root / "data" / "gt_depth" / "front" / "frame_0000.exr"
    if not depth_path.exists():
        results.append(CheckResult("Sample depth content", "FAIL", f"File not found: {depth_path}"))
    else:
        depth = _load_depth_exr(depth_path)
        if depth is None:
            results.append(
                CheckResult("Sample depth content", "WARN", "OpenEXR not installed; skipped")
            )
        else:
            if depth.shape != EXPECTED_SHAPE_DEPTH:
                results.append(
                    CheckResult(
                        "Sample depth content",
                        "FAIL",
                        f"Expected shape {EXPECTED_SHAPE_DEPTH}, got {depth.shape}",
                    )
                )
            elif depth.dtype != np.float32:
                results.append(
                    CheckResult(
                        "Sample depth content", "FAIL", f"Expected float32, got {depth.dtype}"
                    )
                )
            else:
                valid = depth[np.isfinite(depth) & (depth > 0)]
                if len(valid) == 0:
                    results.append(
                        CheckResult("Sample depth content", "FAIL", "All depth values are zero/inf")
                    )
                else:
                    dmin, dmax = float(valid.min()), float(valid.max())
                    results.append(
                        CheckResult(
                            "Sample depth content",
                            "PASS",
                            f"Shape {depth.shape}, dtype float32, depth range [{dmin:.2f}, {dmax:.2f}] m",
                        )
                    )

    return results


def _load_depth_exr(path: Path) -> np.ndarray | None:
    """Load a depth EXR, returning (H,W) float32 or None if OpenEXR unavailable."""
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        raw = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return depth
    except ImportError:
        return None


# ── cameras.yaml check ────────────────────────────────────────────────────────

def check_cameras_yaml(data_root: Path) -> list[CheckResult]:
    """Load cameras.yaml and validate 4 cameras with expected intrinsics."""
    results: list[CheckResult] = []

    yaml_path = data_root / "cameras.yaml"
    if not yaml_path.exists():
        yaml_path = data_root / "config" / "cameras.yaml"

    if not yaml_path.exists():
        results.append(
            CheckResult("cameras.yaml exists", "FAIL", f"Not found at {yaml_path}")
        )
        return results

    try:
        import yaml  # type: ignore
    except ImportError:
        results.append(
            CheckResult("cameras.yaml", "WARN", "PyYAML not installed; skipped")
        )
        return results

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    # Check top-level structure
    if not isinstance(data, dict) or "cameras" not in data:
        results.append(
            CheckResult("cameras.yaml structure", "FAIL", "Missing 'cameras' key at top level")
        )
        return results

    cameras = data["cameras"]
    found_names = set(cameras.keys()) if isinstance(cameras, dict) else set()
    expected_names = set(CAMERAS)

    if found_names != expected_names:
        results.append(
            CheckResult(
                "cameras.yaml camera names",
                "FAIL",
                f"Expected {expected_names}, found {found_names}",
            )
        )
        return results

    results.append(
        CheckResult("cameras.yaml camera names", "PASS", f"Found all 4: {sorted(found_names)}")
    )

    # Validate intrinsics for each camera
    intrinsic_errors: list[str] = []
    for cam_name, cam_data in cameras.items():
        intrinsics = cam_data.get("intrinsics", cam_data.get("K", {}))
        if isinstance(intrinsics, dict):
            fx = intrinsics.get("fx", None)
            cx = intrinsics.get("cx", None)
            cy = intrinsics.get("cy", None)
        elif isinstance(intrinsics, (list, tuple)) and len(intrinsics) == 3:
            # 3x3 matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            fx = intrinsics[0][0]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
        else:
            fx = cx = cy = None

        if fx is None or abs(fx - EXPECTED_FX) > 1.0:
            intrinsic_errors.append(f"{cam_name}: fx={fx} (expected ~{EXPECTED_FX})")
        if cx is None or abs(cx - EXPECTED_CX) > 1.0:
            intrinsic_errors.append(f"{cam_name}: cx={cx} (expected {EXPECTED_CX})")
        if cy is None or abs(cy - EXPECTED_CY) > 1.0:
            intrinsic_errors.append(f"{cam_name}: cy={cy} (expected {EXPECTED_CY})")

    if intrinsic_errors:
        results.append(
            CheckResult(
                "cameras.yaml intrinsics",
                "FAIL",
                "; ".join(intrinsic_errors),
            )
        )
    else:
        results.append(
            CheckResult(
                "cameras.yaml intrinsics",
                "PASS",
                f"All 4 cameras have correct fx≈{EXPECTED_FX}, cx={EXPECTED_CX}, cy={EXPECTED_CY}",
            )
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify CLEAR360 M1 Blender export output."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current working directory).",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    print(f"\n{BOLD}CLEAR360 Export Verifier{RESET}")
    print(f"Data root: {data_root}\n")

    all_results: list[CheckResult] = []

    # File existence checks
    all_results.append(check_rgb_frames(data_root))
    all_results.append(check_depth_exrs(data_root))
    all_results.append(check_position_exrs(data_root))
    all_results.append(check_semantic_pngs(data_root))

    # Sample frame content
    all_results.extend(check_sample_frame(data_root))

    # cameras.yaml
    all_results.extend(check_cameras_yaml(data_root))

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = max(len(r.name) for r in all_results) + 2
    print(f"\n{'Check':<{col_w}}  {'Status':<8}  Detail")
    print("─" * (col_w + 60))
    for r in all_results:
        print(f"{r.name:<{col_w}}  {_fmt_status(r.status):<8}  {r.detail}")
    print("─" * (col_w + 60))

    # ── Overall outcome ───────────────────────────────────────────────────────
    statuses = [r.status for r in all_results]
    if "FAIL" in statuses:
        print(f"\n{RED}{BOLD}Overall: FAIL{RESET}\n")
        return 1
    if "WARN" in statuses:
        print(f"\n{YELLOW}{BOLD}Overall: WARN (some optional outputs missing){RESET}\n")
        return 0
    print(f"\n{GREEN}{BOLD}Overall: PASS{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
