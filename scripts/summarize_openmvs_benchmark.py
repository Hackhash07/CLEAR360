"""Run with: python scripts/summarize_openmvs_benchmark.py [path/to/openmvs_pipeline.log]."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photo_reconstruction.benchmark_report import (  # noqa: E402
    find_latest_openmvs_log,
    format_openmvs_summary,
    summarize_openmvs_log,
)


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python scripts/summarize_openmvs_benchmark.py [openmvs_pipeline.log]")
        return 2
    selected = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else find_latest_openmvs_log(ROOT)
    )
    if selected is None or not selected.is_file():
        print("No OpenMVS log found. Run High Mode first or pass a log path.")
        return 2
    report = summarize_openmvs_log(selected)
    print(format_openmvs_summary(report))
    return 0 if report["all_stages_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
