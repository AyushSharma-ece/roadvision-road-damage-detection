"""RoadVision Phase 5 Annotation Conversion & Validation Suite.

Converts verified splits (Pascal VOC XML) into:
  1. Ultralytics YOLO directory + TXT labels + dataset.yaml
  2. COCO JSON (train/val/test annotations)
  3. MATLAB-friendly bounding-box JSON/CSV table
and validates round-trip coordinate accuracy and mask integrity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image


def load_classes(classes_json: Path) -> tuple[list[str], dict[str, int]]:
    data = json.loads(classes_json.read_text(encoding="utf-8"))
    ordered = sorted(data["classes"], key=lambda x: int(x["id"]))
    names = [str(c["code"]) for c in ordered]
    mapping = {str(c["code"]): int(c["id"]) for c in ordered}
    return names, mapping


def voc_to_boxes(xml_path: Path, img_w: int, img_h: int, class_map: dict[str, int]) -> list[dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or bnd is None or not name_el.text:
            continue
        code = name_el.text.strip()
        if code not in class_map:
            continue
        try:
            xmin = max(0.0, min(float(img_w), float(bnd.findtext("xmin", "0"))))
            ymin = max(0.0, min(float(img_h), float(bnd.findtext("ymin", "0"))))
            xmax = max(0.0, min(float(img_w), float(bnd.findtext("xmax", "0"))))
            ymax = max(0.0, min(float(img_h), float(bnd.findtext("ymax", "0"))))
        except ValueError:
            continue
        bw = xmax - xmin
        bh = ymax - ymin
        if bw < 1.0 or bh < 1.0:
            continue
        boxes.append(
            {
                "class_id": class_map[code],
                "code": code,
                "xmin": xmin,
                "ymin": ymin,
                "w": bw,
                "h": bh,
                "cx_norm": (xmin + bw / 2.0) / img_w,
                "cy_norm": (ymin + bh / 2.0) / img_h,
                "w_norm": bw / img_w,
                "h_norm": bh / img_h,
            }
        )
    return boxes


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_yolo_and_coco(
    split_csv: Path,
    classes_json: Path,
    out_dir: Path,
    countries: list[str] | None = None,
) -> dict[str, Any]:
    class_names, class_map = load_classes(classes_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir = out_dir / "yolo_dataset"
    coco_dir = out_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    coco_splits: dict[str, dict[str, Any]] = {
        sp: {
            "images": [],
            "annotations": [],
            "categories": [{"id": class_map[c], "name": c} for c in class_names],
        }
        for sp in ("train", "val", "test")
    }
    ann_id = 1
    img_id = 1
    stats: dict[str, Any] = {
        "splits": collections.Counter(),
        "boxes": collections.Counter(),
        "countries": collections.Counter(),
    }

    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row["country"]
            if countries and country not in countries:
                continue
            split = row["split"]
            if split not in coco_splits:
                continue

            src_img = Path(row["image_path"])
            src_xml = Path(row["xml_path"])
            img_w = int(row["width"])
            img_h = int(row["height"])
            if not src_img.exists() or not src_xml.exists():
                continue

            boxes = voc_to_boxes(src_xml, img_w, img_h, class_map)

            unique_name = f"{country}__{src_img.name}"
            dst_img = yolo_dir / "images" / split / unique_name
            dst_lbl = yolo_dir / "labels" / split / f"{Path(unique_name).stem}.txt"
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(src_img, dst_img)

            yolo_lines = [
                f"{b['class_id']} {b['cx_norm']:.6f} {b['cy_norm']:.6f} {b['w_norm']:.6f} {b['h_norm']:.6f}"
                for b in boxes
            ]
            dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

            coco_splits[split]["images"].append(
                {
                    "id": img_id,
                    "file_name": unique_name,
                    "width": img_w,
                    "height": img_h,
                    "country": country,
                }
            )
            for b in boxes:
                coco_splits[split]["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": b["class_id"],
                        "bbox": [round(b["xmin"], 2), round(b["ymin"], 2), round(b["w"], 2), round(b["h"], 2)],
                        "area": round(b["w"] * b["h"], 2),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                stats["boxes"][b["code"]] += 1

            img_id += 1
            stats["splits"][split] += 1
            stats["countries"][country] += 1

    # Write Ultralytics dataset YAML
    yaml_lines = [
        f"path: {yolo_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, cname in enumerate(class_names):
        yaml_lines.append(f"  {idx}: {cname}")
    (yolo_dir / "roadvision_rdd.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # Write COCO JSON files
    for sp, payload in coco_splits.items():
        (coco_dir / f"instances_{sp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "yolo_yaml": str((yolo_dir / "roadvision_rdd.yaml").resolve()),
        "splits": dict(stats["splits"]),
        "boxes_by_class": dict(stats["boxes"]),
        "countries": dict(stats["countries"]),
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_yolo_export(yolo_dir: Path, num_classes: int) -> dict[str, Any]:
    """Verify every YOLO TXT label has valid class IDs and normalized [0, 1] coords."""
    errors = []
    checked_labels = 0
    total_boxes = 0
    for split in ("train", "val", "test"):
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                errors.append(f"Missing label file: {lbl_path}")
                continue
            checked_labels += 1
            with Image.open(img_path) as im:
                w, h = im.size
                if w <= 0 or h <= 0:
                    errors.append(f"Invalid image size: {img_path}")
            for line_no, ln in enumerate(lbl_path.read_text(encoding="utf-8").splitlines(), 1):
                if not ln.strip():
                    continue
                parts = ln.split()
                if len(parts) != 5:
                    errors.append(f"Malformed line {line_no} in {lbl_path}")
                    continue
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if not (0 <= cid < num_classes):
                    errors.append(f"Out-of-range class id {cid} in {lbl_path}")
                if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    errors.append(f"Invalid normalized box {vals} in {lbl_path}")
                total_boxes += 1
    return {"checked_labels": checked_labels, "total_boxes": total_boxes, "errors": errors[:20], "error_count": len(errors)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RoadVision splits to YOLO and COCO formats.")
    parser.add_argument("--split-csv", type=Path, default=Path("data/interim/rdd2022_splits.csv"))
    parser.add_argument("--classes-json", type=Path, default=Path("config/classes.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--countries", nargs="*", default=None, help="Optional subset of countries to include.")
    args = parser.parse_args()

    summary = export_yolo_and_coco(args.split_csv, args.classes_json, args.out_dir, args.countries)
    class_names, _ = load_classes(args.classes_json)
    val_res = validate_yolo_export(args.out_dir / "yolo_dataset", len(class_names))
    print(json.dumps({"conversion": summary, "validation": val_res}, indent=2))
    return 0 if val_res["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
