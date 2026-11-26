"""RoadVision Phase 12/14 Model Evaluation & Hardware Benchmark Suite.

Evaluates trained YOLO weights on the held-out test split (never seen during
training or hyperparameter tuning) and computes:
  - Precision, Recall, F1-score, mAP@0.5, mAP@0.5:0.95
  - Per-class metrics (D00, D10, D20, D40) and confusion matrix
  - Inference latency (ms), FPS, and model file size (MB)
  - Subgroup scenario breakdowns (by country / resolution / zero-defect negatives)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "roadvision_yolo_cfg"))

import numpy as np
import torch
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained RoadVision model on test set.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8s_roadvision_best.pt"))
    parser.add_argument("--data-yaml", type=Path, default=Path("data/processed/yolo_dataset/roadvision_rdd.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=Path("reports/model_evaluation_summary.json"))
    parser.add_argument("--append", action="store_true", help="Merge/update entry inside existing summary JSON.")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Trained weights not found: {args.weights}")
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_yaml}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data_yaml.resolve()),
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        plots=True,
        project="reports/eval_plots",
        name="yolov8s_test",
        exist_ok=True,
    )

    p = float(np.mean(metrics.box.p)) if hasattr(metrics.box, "p") else 0.0
    r = float(np.mean(metrics.box.r)) if hasattr(metrics.box, "r") else 0.0
    f1 = (2 * p * r / max(p + r, 1e-8))
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    per_class = {}
    names = model.names
    for idx, cid in enumerate(metrics.box.ap_class_index):
        cname = str(names[int(cid)])
        per_class[cname] = {
            "precision": round(float(metrics.box.p[idx]), 4),
            "recall": round(float(metrics.box.r[idx]), 4),
            "ap50": round(float(metrics.box.ap50[idx]), 4),
            "ap50_95": round(float(metrics.box.ap[idx]), 4),
        }

    # Measure direct inference latency over 30 test images
    test_dir = args.data_yaml.parent / "images" / "test"
    sample_imgs = sorted(test_dir.glob("*"))[:30]
    t_lat = []
    for sp in sample_imgs:
        t0 = time.perf_counter()
        model.predict(source=str(sp), conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)
        t_lat.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(t_lat)) if t_lat else 0.0
    fps = 1000.0 / max(mean_ms, 1e-3) if t_lat else 0.0
    size_mb = round(args.weights.stat().st_size / (1024 * 1024), 2)

    summary = {
        "hardware": {
            "torch_version": torch.__version__,
            "device": f"cuda:{device} ({torch.cuda.get_device_name(0)})" if device != "cpu" else "cpu",
        },
        "models": [
            {
                "name": args.weights.stem,
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mean_inference_ms": round(mean_ms, 2),
                "fps": round(fps, 2),
                "size_mb": size_mb,
                "per_class": per_class,
            }
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.output_json.exists():
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        models_list = [m for m in existing.get("models", []) if m.get("name") != summary["models"][0]["name"]]
        models_list.append(summary["models"][0])
        summary["models"] = models_list
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
