"""Create a realistic Indian Road Inspection demo video from held-out RDD2022 India test images.

Generates an AVI file with smooth sliding dashcam crops across real Indian road frames
carrying verified defects so processRoadVideo.m & RoadDefectTracker can demonstrate
Kalman + Hungarian multi-frame unique defect counting vs. raw frame detections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_demo_video(test_img_dir: Path, test_lbl_dir: Path, output_avi: Path, num_scenes: int = 6, frames_per_scene: int = 12) -> Path:
    india_imgs = sorted(test_img_dir.glob("India__*.jpg"))
    # Select real held-out India images that actually contain labeled defects
    damaged: list[Path] = []
    for p in india_imgs:
        lbl = test_lbl_dir / f"{p.stem}.txt"
        if lbl.exists() and len(lbl.read_text(encoding="utf-8").strip()) > 0:
            damaged.append(p)
        if len(damaged) >= num_scenes:
            break

    if not damaged:
        raise RuntimeError(f"No damaged India test images found in {test_img_dir}")

    output_avi.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = 640, 640
    writer = cv2.VideoWriter(str(output_avi), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create AVI writer at {output_avi}")

    try:
        for img_path in damaged:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            up = cv2.resize(im, (720, 720), interpolation=cv2.INTER_LINEAR)
            # Simulate steady forward/lateral dashcam pan over frames_per_scene consecutive frames
            for f in range(frames_per_scene):
                dx = int(round(4 + f * 3.5))
                dy = int(round(4 + f * 3.0))
                crop = up[dy : dy + out_h, dx : dx + out_w]
                writer.write(crop)
    finally:
        writer.release()

    print(f"Created demo video ({len(damaged) * frames_per_scene} frames from {len(damaged)} India scenes) -> {output_avi}")
    return output_avi


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate India road demo video from held-out test split.")
    parser.add_argument("--test-img-dir", type=Path, default=Path("data/processed/yolo_dataset/images/test"))
    parser.add_argument("--test-lbl-dir", type=Path, default=Path("data/processed/yolo_dataset/labels/test"))
    parser.add_argument("--output-avi", type=Path, default=Path("reports/roadvision_india_demo.avi"))
    args = parser.parse_args()
    build_demo_video(args.test_img_dir, args.test_lbl_dir, args.output_avi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
