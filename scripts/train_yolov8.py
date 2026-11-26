"""RoadVision Phase 6 Reproducible YOLOv8 Training Script.

Trains YOLOv8s (or configurable YOLO variant) on the verified, leakage-free
RDD2022 YOLO dataset with fixed random seeds, checkpoint saving, resume
support, and JSON configuration logging.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "roadvision_yolo_cfg"))

import numpy as np
import torch
from ultralytics import YOLO


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RoadVision YOLOv8 detector.")
    parser.add_argument("--data-yaml", type=Path, default=Path("data/processed/yolo_dataset/roadvision_rdd.yaml"))
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO weights or architecture.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", type=str, default="roadvision_yolov8s")
    parser.add_argument("--output-stem", type=str, default="yolov8s_roadvision_best")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt checkpoint.")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX.")
    args = parser.parse_args()

    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_yaml}")

    set_deterministic_seed(args.seed)
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA unavailable in this Python environment; training on CPU will be slower.")

    # Auto-detect checkpoint if resuming
    model_arg = args.model
    resume_flag = False
    last_ckpt = args.project / args.name / "weights" / "last.pt"
    if args.resume and last_ckpt.exists():
        model_arg = str(last_ckpt)
        resume_flag = True
        print(f"Resuming interrupted training from {last_ckpt}")

    model = YOLO(model_arg)
    train_cfg = {
        "data": str(args.data_yaml.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "deterministic": True,
        "device": device,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "resume": resume_flag,
        "save": True,
        "save_period": 5,
        "amp": torch.cuda.is_available(),
        "patience": 8,
        "workers": args.workers,
    }

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_out = models_dir / f"{args.name}_train_config.json"
    meta_out.write_text(
        json.dumps(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "train_config": train_cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results = model.train(**train_cfg)
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        target_pt = models_dir / f"{args.output_stem}.pt"
        shutil.copy2(best_pt, target_pt)
        print(f"Saved best model -> {target_pt}")
        if args.export_onnx:
            best_model = YOLO(str(target_pt))
            onnx_file = Path(best_model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True))
            target_onnx = models_dir / f"{args.output_stem}.onnx"
            if onnx_file.resolve() != target_onnx.resolve():
                shutil.copy2(onnx_file, target_onnx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
