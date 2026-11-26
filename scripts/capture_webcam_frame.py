"""RoadVision Camera Snapshot Bridge (OpenCV CAP_DSHOW).

Captures a real camera frame into a PNG file for MATLAB RoadVision inspection
when MATLAB's Image Acquisition Toolbox MEX is blocked by Windows App Control.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import cv2


def capture_snapshot(output_path: Path, camera_index: int = 0) -> None:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} could not be opened.")
    try:
        # Warm up 3 frames for auto-exposure stabilization
        frame = None
        for _ in range(4):
            ok, frm = cap.read()
            if ok and frm is not None:
                frame = frm
        if frame is None:
            raise RuntimeError("Failed to grab frame from camera.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Captured frame {frame.shape} -> {output_path}")
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture snapshot from local camera.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    try:
        capture_snapshot(args.output, args.camera_index)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"CAMERA ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
