"""RoadVision Phase 1 Python Environment Diagnostic Script.

Verifies Python version, core CV/ML/PDF/DB libraries, CUDA availability,
and optional camera capture availability without mutating user settings.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import sys
import tempfile
from pathlib import Path


def check_environment(output_json: Path | None = None, check_camera: bool = False) -> dict:
    """Collect structured diagnostics about the Python runtime and packages."""
    # Avoid writing Ultralytics settings into restricted AppData paths if sandboxed.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(Path(tempfile.gettempdir()) / "roadvision_yolo_cfg"),
    )

    report: dict = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "packages": {},
        "cuda_available": False,
        "cuda_device": None,
        "camera_available": None,
    }

    for pkg in ("numpy", "cv2", "PIL", "torch", "torchvision", "ultralytics", "reportlab"):
        try:
            mod = __import__(pkg)
            report["packages"][pkg] = getattr(mod, "__version__", getattr(mod, "Version", "installed"))
        except Exception as exc:  # noqa: BLE001
            report["packages"][pkg] = f"ERROR: {exc}"

    try:
        import torch  # type: ignore

        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        report["cuda_error"] = str(exc)

    if check_camera:
        try:
            import cv2  # type: ignore

            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)
            opened = bool(cap.isOpened())
            frame_shape = None
            if opened:
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame_shape = list(frame.shape)
            cap.release()
            report["camera_available"] = opened
            report["camera_frame_shape"] = frame_shape
        except Exception as exc:  # noqa: BLE001
            report["camera_available"] = False
            report["camera_error"] = str(exc)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="RoadVision Phase 1 Python environment check.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON report.")
    parser.add_argument("--check-camera", action="store_true", help="Probe default camera index 0.")
    args = parser.parse_args()

    report = check_environment(output_json=args.output_json, check_camera=args.check_camera)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
