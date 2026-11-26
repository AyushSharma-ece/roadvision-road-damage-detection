"""RoadVision Phase 2 Dataset Integrity Audit, Deduplication & Leakage-Free Splitter.

Validates raw images, Pascal VOC XML bounding-box annotations, and pixel
segmentation masks; detects exact (MD5) and near-duplicate (64-bit dHash)
images; and generates grouped train/val/test splits without data leakage.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

VALID_DETECTION_CLASSES = ("D00", "D10", "D20", "D40")


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(gray_img: np.ndarray, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash) for near-duplicate detection."""
    resized = cv2.resize(gray_img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return f"{bits:016x}"


def validate_image_file(path: Path) -> tuple[bool, dict[str, Any]]:
    """Verify full image decode integrity and return dimensions/hashes."""
    try:
        with Image.open(path) as pil_img:
            pil_img.verify()
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            return False, {"error": "cv2_decode_failed"}
        h, w = img.shape[:2]
        c = 1 if img.ndim == 2 else img.shape[2]
        if w <= 0 or h <= 0:
            return False, {"error": "invalid_dimensions"}
        gray = img if c == 1 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        return True, {
            "width": int(w),
            "height": int(h),
            "channels": int(c),
            "md5": compute_md5(path),
            "dhash": compute_dhash(gray),
        }
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def parse_pascal_voc_xml(
    xml_path: Path,
    img_w: int,
    img_h: int,
    allowed_classes: tuple[str, ...] = VALID_DETECTION_CLASSES,
) -> dict[str, Any]:
    """Parse and validate Pascal VOC XML bounding boxes for RoadVision classes."""
    result: dict[str, Any] = {
        "valid": True,
        "boxes": [],
        "ignored_classes": collections.Counter(),
        "clipped_boxes": 0,
        "invalid_boxes": 0,
        "error": None,
    }
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:  # noqa: BLE001
        result["valid"] = False
        result["error"] = f"xml_parse_error: {exc}"
        return result

    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            result["invalid_boxes"] += 1
            continue
        cls_name = name_el.text.strip()
        if cls_name not in allowed_classes:
            result["ignored_classes"][cls_name] += 1
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            result["invalid_boxes"] += 1
            continue
        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            result["invalid_boxes"] += 1
            continue

        orig = (xmin, ymin, xmax, ymax)
        xmin = max(0.0, min(float(img_w), xmin))
        ymin = max(0.0, min(float(img_h), ymin))
        xmax = max(0.0, min(float(img_w), xmax))
        ymax = max(0.0, min(float(img_h), ymax))
        if (xmin, ymin, xmax, ymax) != orig:
            result["clipped_boxes"] += 1

        if xmax - xmin < 1.0 or ymax - ymin < 1.0:
            result["invalid_boxes"] += 1
            continue

        result["boxes"].append(
            {
                "class": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return result


def audit_rdd2022(rdd_root: Path) -> dict[str, Any]:
    """Audit all country subsets of RDD2022 (train labeled & test unlabelled)."""
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset": "RDD2022",
        "countries": {},
        "corrupt_images": 0,
        "missing_xml": 0,
        "xml_errors": 0,
        "invalid_boxes": 0,
        "clipped_boxes": 0,
        "ignored_classes": collections.Counter(),
        "class_counts": collections.Counter(),
        "exact_duplicates": 0,
        "near_duplicates_dhash": 0,
    }

    seen_md5: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}

    if not rdd_root.exists():
        summary["status"] = "not_extracted"
        return {"summary": summary, "records": records}

    country_dirs = sorted(
        d for d in rdd_root.iterdir()
        if d.is_dir() and (d / "train").exists()
    )
    # Handle nested RDD2022/RDD2022/<Country> if present
    if not country_dirs and (rdd_root / "RDD2022").is_dir():
        rdd_root = rdd_root / "RDD2022"
        country_dirs = sorted(d for d in rdd_root.iterdir() if d.is_dir() and (d / "train").exists())

    for cdir in country_dirs:
        country = cdir.name
        c_stats = {
            "train_images": 0,
            "train_xmls": 0,
            "negative_images": 0,
            "corrupt_images": 0,
            "missing_xml": 0,
            "test_images_unlabelled": 0,
            "class_counts": collections.Counter(),
        }

        img_dir = cdir / "train" / "images"
        xml_dir = cdir / "train" / "annotations" / "xmls"
        img_files = sorted(img_dir.glob("*"))
        print(f"[audit] Auditing RDD2022/{country} ({len(img_files):,} files)...", flush=True)
        for img_path in img_files:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            ok, meta = validate_image_file(img_path)
            if not ok:
                c_stats["corrupt_images"] += 1
                summary["corrupt_images"] += 1
                continue

            c_stats["train_images"] += 1
            xml_path = xml_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                c_stats["missing_xml"] += 1
                summary["missing_xml"] += 1
                continue

            c_stats["train_xmls"] += 1
            ann = parse_pascal_voc_xml(xml_path, meta["width"], meta["height"])
            if not ann["valid"]:
                summary["xml_errors"] += 1
                continue

            summary["invalid_boxes"] += ann["invalid_boxes"]
            summary["clipped_boxes"] += ann["clipped_boxes"]
            summary["ignored_classes"].update(ann["ignored_classes"])

            if not ann["boxes"]:
                c_stats["negative_images"] += 1
            for b in ann["boxes"]:
                c_stats["class_counts"][b["class"]] += 1
                summary["class_counts"][b["class"]] += 1

            is_exact_dup = meta["md5"] in seen_md5
            is_dhash_dup = (not is_exact_dup) and (meta["dhash"] in seen_dhash)
            if is_exact_dup:
                summary["exact_duplicates"] += 1
            else:
                seen_md5[meta["md5"]] = str(img_path)
            if is_dhash_dup:
                summary["near_duplicates_dhash"] += 1
            else:
                seen_dhash[meta["dhash"]] = str(img_path)

            # Sequence / numeric grouping key prevents consecutive frame leakage
            stem_parts = img_path.stem.rsplit("_", 1)
            seq_bucket = int(stem_parts[1]) // 10 if len(stem_parts) == 2 and stem_parts[1].isdigit() else 0
            group_id = f"{country}_{meta['dhash'][:8]}_{seq_bucket}"

            records.append(
                {
                    "country": country,
                    "image_path": str(img_path),
                    "xml_path": str(xml_path),
                    "width": meta["width"],
                    "height": meta["height"],
                    "md5": meta["md5"],
                    "dhash": meta["dhash"],
                    "is_exact_dup": is_exact_dup,
                    "is_near_dup": is_dhash_dup,
                    "group_id": group_id,
                    "num_boxes": len(ann["boxes"]),
                    "classes_present": sorted({b["class"] for b in ann["boxes"]}),
                }
            )

        test_img_dir = cdir / "test" / "images"
        if test_img_dir.exists():
            c_stats["test_images_unlabelled"] = sum(
                1 for p in test_img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

        c_stats["class_counts"] = dict(c_stats["class_counts"])
        summary["countries"][country] = c_stats

    summary["status"] = "audited"
    summary["ignored_classes"] = dict(summary["ignored_classes"])
    summary["class_counts"] = dict(summary["class_counts"])
    summary["total_valid_labeled_images"] = len(records)
    return {"summary": summary, "records": records}


def audit_segmentation_dir(
    name: str,
    image_dir: Path,
    mask_dir: Path,
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    mask_exts: tuple[str, ...] = (".png", ".jpg", ".bmp", ".mat"),
) -> dict[str, Any]:
    """Audit paired image + segmentation mask directories."""
    if not image_dir.exists():
        return {"dataset": name, "status": "missing", "valid_pairs": 0}

    imgs = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)
    masks_by_stem: dict[str, Path] = {}
    if mask_dir.exists():
        for m in mask_dir.rglob("*"):
            if m.is_file() and m.suffix.lower() in mask_exts:
                masks_by_stem[m.stem] = m

    valid_pairs = 0
    corrupt = 0
    missing_mask = 0
    md5_seen: set[str] = set()
    dup_count = 0
    resolutions: collections.Counter = collections.Counter()

    for img_path in imgs:
        ok, meta = validate_image_file(img_path)
        if not ok:
            corrupt += 1
            continue
        mask_path = masks_by_stem.get(img_path.stem)
        if mask_path is None:
            missing_mask += 1
            continue
        if meta["md5"] in md5_seen:
            dup_count += 1
        md5_seen.add(meta["md5"])
        resolutions[f"{meta['width']}x{meta['height']}"] += 1
        valid_pairs += 1

    return {
        "dataset": name,
        "status": "audited",
        "total_images": len(imgs),
        "valid_pairs": valid_pairs,
        "corrupt_images": corrupt,
        "missing_masks": missing_mask,
        "exact_duplicates": dup_count,
        "top_resolutions": dict(resolutions.most_common(5)),
    }


def make_leakage_free_splits(
    records: list[dict[str, Any]],
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign train/val/test splits grouped by country & sequence/dHash bucket."""
    # Deduplicate exact copies first so identical images never cross splits.
    unique_records = [r for r in records if not r["is_exact_dup"]]

    # Group by (country, dhash) and sequence bucket to keep related frames together.
    by_country: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for rec in unique_records:
        by_country[rec["country"]][rec["group_id"]].append(rec)

    rng = random.Random(seed)
    split_records: list[dict[str, Any]] = []
    for country in sorted(by_country):
        groups = list(by_country[country].values())
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        acc = 0
        for g in groups:
            if acc < n_train:
                split = "train"
            elif acc < n_train + n_val:
                split = "val"
            else:
                split = "test"
            acc += len(g)
            for item in g:
                row = dict(item)
                row["split"] = split
                split_records.append(row)

    return split_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit RoadVision datasets and create leakage-free splits.")
    parser.add_argument("--data-raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"segmentation_datasets": []}

    # 1. CFD
    cfd_dir = args.data_raw / "cfd" / "CrackForest-dataset-master"
    report["segmentation_datasets"].append(
        audit_segmentation_dir("CFD (CrackForest)", cfd_dir / "image", cfd_dir / "groundTruth")
    )

    # 2. DeepCrack
    dc_base = args.data_raw / "deepcrack" / "extracted"
    dc_train = audit_segmentation_dir("DeepCrack-train", dc_base / "train_img", dc_base / "train_lab")
    dc_test = audit_segmentation_dir("DeepCrack-test", dc_base / "test_img", dc_base / "test_lab")
    report["segmentation_datasets"].extend([dc_train, dc_test])

    # 3. Pothole-600 (detect directory layout dynamically if extracted)
    p600_dir = args.data_raw / "pothole600"
    if p600_dir.exists() and any(p600_dir.rglob("*.png")):
        # Try common layouts: rgb / label or testing/training/validation
        img_dirs = [d for d in p600_dir.rglob("rgb") if d.is_dir()]
        if img_dirs:
            for idir in sorted(img_dirs):
                lbl_dir = idir.parent / "label"
                report["segmentation_datasets"].append(
                    audit_segmentation_dir(f"Pothole-600 ({idir.parent.name})", idir, lbl_dir)
                )

    # 4. Crack500 extracted splits (testdata / valdata / traincrop / testcrop / valcrop)
    c500_ext = args.data_raw / "crack500" / "extracted"
    if c500_ext.exists():
        for sub_name, img_sub, mask_sub in [
            ("Crack500-testdata", "testdata/testdata", "testdata/testdata"),
            ("Crack500-valdata", "valdata/valdata", "valdata/valdata"),
            ("Crack500-traincrop", "traincrop/traincrop", "traincrop/traincrop"),
            ("Crack500-testcrop", "testcrop/testcrop", "testcrop/testcrop"),
            ("Crack500-valcrop", "valcrop/valcrop", "valcrop/valcrop"),
        ]:
            idir = c500_ext / img_sub
            if not idir.exists():
                continue
            imgs = sorted(p for p in idir.glob("*.jpg"))
            masks_by_stem = {m.stem.replace("_mask", ""): m for m in idir.glob("*.png")}
            valid_pairs = 0
            corrupt = 0
            missing = 0
            md5s: set[str] = set()
            dups = 0
            res_cnt: collections.Counter = collections.Counter()
            for img_p in imgs:
                ok, meta = validate_image_file(img_p)
                if not ok:
                    corrupt += 1
                    continue
                if img_p.stem not in masks_by_stem:
                    missing += 1
                    continue
                if meta["md5"] in md5s:
                    dups += 1
                md5s.add(meta["md5"])
                res_cnt[f"{meta['width']}x{meta['height']}"] += 1
                valid_pairs += 1
            report["segmentation_datasets"].append(
                {
                    "dataset": sub_name,
                    "status": "audited",
                    "total_images": len(imgs),
                    "valid_pairs": valid_pairs,
                    "corrupt_images": corrupt,
                    "missing_masks": missing,
                    "exact_duplicates": dups,
                    "top_resolutions": dict(res_cnt.most_common(4)),
                }
            )

    # 5. RDD2022
    rdd_res = audit_rdd2022(args.data_raw / "rdd2022")
    report["rdd2022"] = rdd_res["summary"]

    if rdd_res["records"]:
        split_rows = make_leakage_free_splits(rdd_res["records"], seed=args.seed)
        manifest_csv = args.output_dir / "rdd2022_splits.csv"
        fieldnames = [
            "split",
            "country",
            "image_path",
            "xml_path",
            "width",
            "height",
            "md5",
            "dhash",
            "is_near_dup",
            "group_id",
            "num_boxes",
            "classes_present",
        ]
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in split_rows:
                row = {k: r[k] for k in fieldnames}
                row["classes_present"] = ";".join(r["classes_present"])
                writer.writerow(row)
        split_counts = collections.Counter(r["split"] for r in split_rows)
        report["rdd2022"]["split_counts"] = dict(split_counts)
        report["rdd2022"]["split_manifest"] = str(manifest_csv)

    out_json = args.output_dir / "dataset_audit_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
