"""RoadVision YOLOv8 Inference Worker (Image & Batch Video modes).

Runs single-image or video-stream inference using Ultralytics YOLOv8 (.pt/.onnx)
and outputs a clean JSON payload with bounding boxes in [x, y, w, h] pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "roadvision_yolo_cfg"))

import cv2
import torch
from ultralytics import YOLO


def _extract_boxes(res) -> list[dict]:
    names = res.names
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for idx, (box, sc, cid) in enumerate(zip(xyxy, confs, clss), start=1):
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append(
                {
                    "detection_id": idx,
                    "class_id": int(cid),
                    "class_code": str(names[int(cid)]),
                    "confidence": round(float(sc), 5),
                    "bbox_xywh": [
                        round(x1, 2),
                        round(y1, 2),
                        round(max(1.0, x2 - x1), 2),
                        round(max(1.0, y2 - y1), 2),
                    ],
                }
            )
    return dets


def run_image_inference(weights: Path, image_path: Path, conf: float, iou: float, imgsz: int) -> dict:
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "weights": str(weights),
        "image": str(image_path),
        "device": f"cuda:{device}" if device != "cpu" else "cpu",
        "inference_ms": round(elapsed_ms, 2),
        "detections": _extract_boxes(results[0]),
    }


def run_video_inference(weights: Path, video_path: Path, stride: int, conf: float, iou: float, imgsz: int) -> dict:
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frames_out = []
    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % max(1, stride) != 0:
                continue
            res = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
            frames_out.append({"frame_index": frame_idx, "detections": _extract_boxes(res)})
    finally:
        cap.release()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "weights": str(weights),
        "video": str(video_path),
        "device": f"cuda:{device}" if device != "cpu" else "cpu",
        "total_time_ms": round(elapsed_ms, 2),
        "frames": frames_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RoadVision YOLOv8 JSON inference worker.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.video is not None:
        payload = run_video_inference(args.weights, args.video, args.stride, args.conf, args.iou, args.imgsz)
    elif args.image is not None:
        payload = run_image_inference(args.weights, args.image, args.conf, args.iou, args.imgsz)
    else:
        raise ValueError("Provide either --image or --video.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
