"""Parse stage timings and optional per-command peak-RSS data from OpenMVS logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_STAGE_RE = re.compile(
    r"\[CLEAR360 stage timing\]\s+([^:]+):\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"seconds;\s+exit_code=(-?\d+)"
)
_RSS_RE = re.compile(r"^\s*([\d,]+)\s+maximum resident set size\s*$")
_CONFIG_PREFIX = "Pipeline configuration: "


def summarize_openmvs_log(log_path: str | Path) -> dict[str, Any]:
    """Return measured external-stage times and optional peak RSS from a run log."""
    path = Path(log_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    configuration: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = []
    pending_peak_rss: int | None = None

    for line in lines:
        if line.startswith(_CONFIG_PREFIX):
            try:
                parsed = json.loads(line[len(_CONFIG_PREFIX):])
                if isinstance(parsed, dict):
                    configuration = parsed
            except json.JSONDecodeError:
                # Older or partially written logs may contain an incomplete config.
                continue

        rss_match = _RSS_RE.match(line)
        if rss_match:
            pending_peak_rss = int(rss_match.group(1).replace(",", ""))
            continue

        stage_match = _STAGE_RE.search(line)
        if stage_match:
            stages.append({
                "tool": stage_match.group(1).strip(),
                "duration_seconds": float(stage_match.group(2)),
                "exit_code": int(stage_match.group(3)),
                "peak_rss_bytes": pending_peak_rss,
            })
            pending_peak_rss = None

    return {
        "log_file": str(path),
        "configuration": configuration,
        "stages": stages,
        "external_stage_total_seconds": round(
            sum(stage["duration_seconds"] for stage in stages), 2
        ),
        "all_stages_succeeded": bool(stages) and all(
            stage["exit_code"] == 0 for stage in stages
        ),
        "memory_profiled_stage_count": sum(
            stage["peak_rss_bytes"] is not None for stage in stages
        ),
    }


def find_latest_openmvs_log(project_root: str | Path) -> Path | None:
    """Find the newest saved OpenMVS run log beneath the project data folder."""
    root = Path(project_root) / "data" / "phone_reconstruction"
    candidates = list(root.glob("session_*/openmvs_pipeline.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def format_openmvs_summary(report: dict[str, Any]) -> str:
    """Format a compact human-readable stage table."""
    rows = [
        f"OpenMVS benchmark log: {report['log_file']}",
        "",
        f"{'Stage':<26} {'Time':>10} {'Peak RSS':>14} {'Exit':>8}",
        "-" * 62,
    ]
    for stage in report["stages"]:
        rss = stage["peak_rss_bytes"]
        rss_text = f"{rss / (1024 ** 2):.0f} MiB" if rss is not None else "not recorded"
        rows.append(
            f"{stage['tool']:<26} "
            f"{stage['duration_seconds']:>8.1f}s "
            f"{rss_text:>14} "
            f"{stage['exit_code']:>8}"
        )
    rows.extend([
        "-" * 62,
        f"Sum of external stages: {report['external_stage_total_seconds']:.1f}s",
        f"All recorded stages succeeded: {report['all_stages_succeeded']}",
        (
            f"Stages with peak RSS: {report['memory_profiled_stage_count']}"
        ),
    ])
    config = report.get("configuration") or {}
    if config:
        rows.append(
            "Profile: "
            + str(config.get("performance_profile", "unknown"))
            + " | photos: "
            + str(config.get("input_photo_count", "unknown"))
        )
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize the latest CLEAR360 COLMAP + OpenMVS run log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to openmvs_pipeline.log (defaults to newest saved session log).",
    )
    args = parser.parse_args()

    if args.log_file:
        selected_log = Path(args.log_file)
    else:
        selected_log = find_latest_openmvs_log(Path(__file__).resolve().parents[1])
        if selected_log is None:
            parser.error("No OpenMVS logs found under data/phone_reconstruction.")

    print(format_openmvs_summary(summarize_openmvs_log(selected_log)))
