"""Pytest edge-case & integration suite for RoadVision Python workers (Phase 15)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_pdf_report  # noqa: E402
import infer_yolov8  # noqa: E402


def test_infer_missing_weights_raises(tmp_path: Path) -> None:
    img = tmp_path / "test.jpg"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(FileNotFoundError):
        infer_yolov8.run_image_inference(tmp_path / "missing.pt", img, 0.3, 0.45, 640)


def test_infer_missing_image_raises(tmp_path: Path) -> None:
    fake_pt = tmp_path / "dummy.pt"
    fake_pt.write_bytes(b"not-a-model")
    with pytest.raises(FileNotFoundError):
        infer_yolov8.run_image_inference(fake_pt, tmp_path / "missing.jpg", 0.3, 0.45, 640)


def test_pdf_generator_handles_zero_defects_and_missing_gps(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    ann = tmp_path / "ann.png"
    out_pdf = tmp_path / "zero_defects.pdf"
    cv2.imwrite(str(orig), np.full((120, 160, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(ann), np.full((120, 160, 3), 128, dtype=np.uint8))

    payload = {
        "created_at": "2026-11-25 17:15:00",
        "source_path": str(orig),
        "source_type": "image",
        "model_backend": "YOLOv8s",
        "model_version": "1.0",
        "overall_severity": "None",
        "damage_count": 0,
        "has_gps": False,
        "latitude": None,
        "longitude": None,
        "inference_ms": 8.4,
        "original_image": str(orig),
        "annotated_image": str(ann),
        "defects": [],
    }
    payload_json = tmp_path / "payload.json"
    payload_json.write_text(json.dumps(payload), encoding="utf-8")

    pdf_out = generate_pdf_report.build_pdf_report(payload_json, out_pdf)
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 1500
    assert pdf_out.read_bytes().startswith(b"%PDF-")


def test_pdf_generator_handles_multiple_defects(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    ann = tmp_path / "ann.png"
    out_pdf = tmp_path / "multi_defects.pdf"
    cv2.imwrite(str(orig), np.full((240, 320, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(ann), np.full((240, 320, 3), 100, dtype=np.uint8))

    payload = {
        "created_at": "2026-11-25 17:15:00",
        "source_path": str(orig),
        "source_type": "image",
        "model_backend": "YOLOv8s",
        "model_version": "1.0",
        "overall_severity": "High",
        "damage_count": 2,
        "has_gps": True,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "inference_ms": 14.2,
        "original_image": str(orig),
        "annotated_image": str(ann),
        "defects": [
            {
                "detection_id": 1,
                "class_code": "D00",
                "confidence": 0.81,
                "relative_area": 0.03,
                "severity_score": 0.42,
                "severity_level": "Medium",
                "explanation": "Longitudinal crack spanning 180 px.",
            },
            {
                "detection_id": 2,
                "class_code": "D40",
                "confidence": 0.93,
                "relative_area": 0.11,
                "severity_score": 0.89,
                "severity_level": "High",
                "explanation": "Large pothole detected; depth unmeasured from 2D image.",
            },
        ],
    }
    payload_json = tmp_path / "payload_multi.json"
    payload_json.write_text(json.dumps(payload), encoding="utf-8")

    pdf_out = generate_pdf_report.build_pdf_report(payload_json, out_pdf)
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 2000


def test_pdf_generator_handles_scalar_defect_dict_and_empty_paths(tmp_path: Path) -> None:
    """MATLAB jsonencode emits scalar structs as dicts rather than 1-element lists."""
    out_pdf = tmp_path / "scalar_dict.pdf"
    payload = {
        "created_at": "2026-11-25 17:25:00",
        "source_path": "",
        "source_type": "image",
        "model_backend": "matlab_squeezenet",
        "model_version": "1.0",
        "overall_severity": "Medium",
        "damage_count": 1,
        "has_gps": False,
        "latitude": None,
        "longitude": None,
        "inference_ms": 18.5,
        "original_image": "",
        "annotated_image": "",
        "defects": {
            "detection_id": 1,
            "class_code": "D40",
            "confidence": 0.91,
            "relative_area": 0.05,
            "severity_score": 0.56,
            "severity_level": "Medium",
            "explanation": "Single-defect MATLAB struct serialization test.",
        },
    }
    payload_json = tmp_path / "payload_scalar.json"
    payload_json.write_text(json.dumps(payload), encoding="utf-8")

    pdf_out = generate_pdf_report.build_pdf_report(payload_json, out_pdf)
    assert pdf_out.exists()
    assert pdf_out.read_bytes().startswith(b"%PDF-")
