"""Pytest suite for RoadVision Phase 2 audit & Phase 5 annotation conversion."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_datasets  # noqa: E402
import convert_annotations  # noqa: E402


def _make_sample_voc_xml(path: Path, filename: str, w: int, h: int, objects: list[tuple[str, int, int, int, int]]) -> None:
    objs_xml = []
    for name, xmin, ymin, xmax, ymax in objects:
        objs_xml.append(
            f"<object><name>{name}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"
        )
    xml_text = (
        f"<annotation><filename>{filename}</filename>"
        f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
        + "".join(objs_xml)
        + "</annotation>"
    )
    path.write_text(xml_text, encoding="utf-8")


def test_audit_detects_corrupt_duplicate_and_prevents_leakage(tmp_path: Path) -> None:
    country_dir = tmp_path / "India" / "train"
    img_dir = country_dir / "images"
    xml_dir = country_dir / "annotations" / "xmls"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    # Image 0 & Image 1: distinct valid images
    arr0 = np.full((200, 300, 3), 80, dtype=np.uint8)
    arr0[50:80, 100:150] = 200
    Image.fromarray(arr0).save(img_dir / "India_000001.jpg")
    _make_sample_voc_xml(xml_dir / "India_000001.xml", "India_000001.jpg", 300, 200, [("D40", 100, 50, 150, 80)])

    arr1 = np.full((200, 300, 3), 120, dtype=np.uint8)
    arr1[10:190, 20:28] = 20
    Image.fromarray(arr1).save(img_dir / "India_000020.jpg")
    _make_sample_voc_xml(xml_dir / "India_000020.xml", "India_000020.jpg", 300, 200, [("D00", 20, 10, 28, 190)])

    # Image 2: exact byte duplicate of Image 0
    (img_dir / "India_000030.jpg").write_bytes((img_dir / "India_000001.jpg").read_bytes())
    _make_sample_voc_xml(xml_dir / "India_000030.xml", "India_000030.jpg", 300, 200, [("D40", 100, 50, 150, 80)])

    # Image 3: corrupt image
    (img_dir / "India_000040.jpg").write_bytes(b"not-a-valid-jpeg")

    res = audit_datasets.audit_rdd2022(tmp_path)
    summary = res["summary"]
    assert summary["corrupt_images"] == 1
    assert summary["exact_duplicates"] == 1
    assert summary["total_valid_labeled_images"] == 3

    splits = audit_datasets.make_leakage_free_splits(res["records"], seed=42)
    # Exact duplicate excluded from split output
    assert len(splits) == 2


def test_voc_to_yolo_and_coco_roundtrip(tmp_path: Path) -> None:
    img_path = tmp_path / "India_000001.jpg"
    xml_path = tmp_path / "India_000001.xml"
    Image.new("RGB", (640, 480), color=(90, 90, 90)).save(img_path)
    _make_sample_voc_xml(
        xml_path,
        "India_000001.jpg",
        640,
        480,
        [("D00", 10, 20, 110, 220), ("D40", 300, 200, 460, 360), ("D99_UNKNOWN", 1, 1, 10, 10)],
    )

    split_csv = tmp_path / "splits.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "country", "image_path", "xml_path", "width", "height"],
        )
        w.writeheader()
        w.writerow(
            {
                "split": "train",
                "country": "India",
                "image_path": str(img_path),
                "xml_path": str(xml_path),
                "width": "640",
                "height": "480",
            }
        )

    out_dir = tmp_path / "processed"
    summary = convert_annotations.export_yolo_and_coco(split_csv, ROOT / "config" / "classes.json", out_dir)
    assert summary["boxes_by_class"] == {"D00": 1, "D40": 1}

    val_res = convert_annotations.validate_yolo_export(out_dir / "yolo_dataset", 4)
    assert val_res["error_count"] == 0
    assert val_res["total_boxes"] == 2

    coco_train = json.loads((out_dir / "coco" / "instances_train.json").read_text(encoding="utf-8"))
    assert len(coco_train["images"]) == 1
    assert len(coco_train["annotations"]) == 2
