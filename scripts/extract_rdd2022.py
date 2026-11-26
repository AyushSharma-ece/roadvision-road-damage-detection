"""RoadVision Phase 2 Reproducible RDD2022 Archive Extractor."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_rdd2022(archive_path: Path, dest_dir: Path) -> int:
    if not archive_path.exists():
        raise FileNotFoundError(f"RDD2022 archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        print(f"Extracting {total:,} entries from {archive_path.name} -> {dest_dir} ...")
        for idx, info in enumerate(members, start=1):
            out_item = dest_dir / info.filename
            if not (out_item.exists() and out_item.is_file() and out_item.stat().st_size == info.file_size):
                zf.extract(info, dest_dir)
            if idx % 5000 == 0 or idx == total:
                print(f"  Stage 1: {idx:,} / {total:,} entries ({100.0 * idx / total:.1f}%)")

    # Handle nested country zips (e.g., India.zip, Japan.zip, Norway.zip, ...) inside RDD2022 master archive
    inner_zips = sorted(dest_dir.rglob("*.zip"))
    if inner_zips:
        print(f"Found {len(inner_zips)} nested country archives; extracting cleanly...")
        for zp in inner_zips:
            with zipfile.ZipFile(zp, "r") as zf2:
                mems = zf2.infolist()
                tot2 = len(mems)
                print(f"  [{zp.name}] Extracting {tot2:,} files...")
                for idx2, info2 in enumerate(mems, start=1):
                    zf2.extract(info2, zp.parent)
                    if idx2 % 5000 == 0 or idx2 == tot2:
                        print(f"    [{zp.name}] {idx2:,}/{tot2:,}")
            zp.unlink()

    # Report discovered country directories
    search_root = dest_dir / "RDD2022" if (dest_dir / "RDD2022").is_dir() else dest_dir
    countries = sorted(d.name for d in search_root.iterdir() if d.is_dir() and (d / "train").exists())
    print(f"Discovered country subsets in {search_root}: {countries}")
    return len(countries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RDD2022 zip cleanly.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/archives/RDD2022_released_through_CRDDC2022.zip"),
    )
    parser.add_argument("--dest", type=Path, default=Path("data/raw/rdd2022"))
    args = parser.parse_args()
    n = extract_rdd2022(args.archive, args.dest)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
