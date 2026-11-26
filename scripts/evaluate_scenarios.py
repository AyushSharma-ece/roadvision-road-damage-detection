"""RoadVision Phase 14/15 Multi-Scenario & Real-World Condition Evaluation Script.

Evaluates a trained YOLOv8 detector across diverse operational scenarios using
verified held-out test images (never seen during training):
  - Country / road environment breakdown (e.g., India, Japan, Czech, Norway, UnitedStates)
  - Undamaged / zero-defect negative road images (false-positive rate)
  - Multi-defect dense road images
  - Simulated low-light, motion blur, and low-resolution stress slices
Never fabricates detections; records actual counts and recall/false-positive rates.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "roadvision_yolo_cfg"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-8, area_a + area_b - inter)


def load_yolo_gt(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        out.append((cid, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def eval_image_set(model: YOLO, items: list[dict], labels_dir: Path, conf: float, device: str, transform: str | None = None) -> dict:
    tp = fp = fn = 0
    neg_imgs = neg_fp_imgs = 0
    lat_ms = []

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_dir = Path(tmpd)
        for idx, item in enumerate(items):
            src = Path(item["img_path"])
            lbl = labels_dir / f"{src.stem}.txt"
            im = cv2.imread(str(src))
            if im is None:
                continue
            h, w = im.shape[:2]
            gt = load_yolo_gt(lbl, w, h)

            infer_src = str(src)
            if transform == "low_light":
                dark = np.clip(im.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
                p = tmp_dir / f"dark_{idx}.jpg"
                cv2.imwrite(str(p), dark)
                infer_src = str(p)
            elif transform == "motion_blur":
                blur = cv2.GaussianBlur(im, (9, 9), 2.5)
                p = tmp_dir / f"blur_{idx}.jpg"
                cv2.imwrite(str(p), blur)
                infer_src = str(p)
            elif transform == "low_res":
                small = cv2.resize(im, (max(64, w // 3), max(64, h // 3)), interpolation=cv2.INTER_AREA)
                up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                p = tmp_dir / f"lowres_{idx}.jpg"
                cv2.imwrite(str(p), up)
                infer_src = str(p)

            t0 = time.perf_counter()
            res = model.predict(source=infer_src, conf=conf, imgsz=640, device=device, verbose=False)[0]
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                for b, c, s in zip(xyxy, clss, confs):
                    preds.append((int(c), b, float(s)))
                preds.sort(key=lambda x: x[2], reverse=True)

            if not gt:
                neg_imgs += 1
                if preds:
                    neg_fp_imgs += 1

            used = [False] * len(gt)
            for pc, pb, _ in preds:
                best_j = -1
                best_iou = 0.0
                for j, (gc, gb) in enumerate(gt):
                    if used[j] or gc != pc:
                        continue
                    iou = box_iou_xyxy(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= 0.5:
                    used[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(1 for u in used if not u)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return {
        "images_evaluated": len(lat_ms),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "negative_images": neg_imgs,
        "negative_fp_images": neg_fp_imgs,
        "mean_inference_ms": round(float(np.mean(lat_ms)), 2) if lat_ms else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scenario evaluation on held-out RDD2022 test split.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8s_roadvision_best.pt"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/processed/yolo_dataset"))
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--max-per-group", type=int, default=80)
    parser.add_argument("--output-json", type=Path, default=Path("reports/scenario_evaluation.json"))
    args = parser.parse_args()

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))

    test_img_dir = args.yolo_dir / "images" / "test"
    test_lbl_dir = args.yolo_dir / "labels" / "test"
    by_country: dict[str, list[dict]] = collections.defaultdict(list)
    negatives: list[dict] = []
    multi_defect: list[dict] = []

    for img_p in sorted(test_img_dir.glob("*")):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        country = img_p.name.split("__", 1)[0] if "__" in img_p.name else "Unknown"
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        lines = [ln for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()] if lbl_p.exists() else []
        rec = {"img_path": str(img_p), "country": country, "num_boxes": len(lines)}
        if len(by_country[country]) < args.max_per_group:
            by_country[country].append(rec)
        if len(lines) == 0 and len(negatives) < args.max_per_group:
            negatives.append(rec)
        if len(lines) >= 2 and len(multi_defect) < args.max_per_group:
            multi_defect.append(rec)

    report = {"weights": str(args.weights), "conf": args.conf, "by_country": {}, "special_scenarios": {}, "stress_conditions": {}}
    for c, items in sorted(by_country.items()):
        report["by_country"][c] = eval_image_set(model, items, test_lbl_dir, args.conf, device)

    report["special_scenarios"]["zero_defect_negatives"] = eval_image_set(model, negatives, test_lbl_dir, args.conf, device)
    report["special_scenarios"]["multi_defect_dense"] = eval_image_set(model, multi_defect, test_lbl_dir, args.conf, device)

    stress_pool = by_country.get("India", [])[:50] or next(iter(by_country.values()), [])[:50]
    if stress_pool:
        report["stress_conditions"]["clear_baseline"] = eval_image_set(model, stress_pool, test_lbl_dir, args.conf, device, None)
        report["stress_conditions"]["low_light_35pct"] = eval_image_set(model, stress_pool, test_lbl_dir, args.conf, device, "low_light")
        report["stress_conditions"]["gaussian_blur"] = eval_image_set(model, stress_pool, test_lbl_dir, args.conf, device, "motion_blur")
        report["stress_conditions"]["low_resolution_3x_down"] = eval_image_set(model, stress_pool, test_lbl_dir, args.conf, device, "low_res")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
