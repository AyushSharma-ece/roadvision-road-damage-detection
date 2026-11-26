"""RoadVision Phase 2 Reproducible Dataset Downloader.

Downloads official/verified road damage and pavement crack datasets into
data/raw/archives/ with integrity verification and SHA-256 logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_SPECS = {
    "rdd2022": {
        "name": "RDD2022 (CRDDC 2022 Multi-National)",
        "url": "https://ndownloader.figshare.com/files/38030910",
        "filename": "RDD2022_released_through_CRDDC2022.zip",
        "license": "CC BY-SA 4.0",
        "expected_min_bytes": 13_000_000_000,
    },
    "cfd": {
        "name": "CrackForest Dataset (CFD)",
        "url": "https://github.com/cuilimeng/CrackForest-dataset/archive/refs/heads/master.zip",
        "filename": "CrackForest-dataset.zip",
        "license": "Non-commercial research purposes only",
        "expected_min_bytes": 9_000_000,
    },
    "deepcrack": {
        "name": "DeepCrack",
        "url": "https://github.com/yhlleo/DeepCrack/archive/refs/heads/master.zip",
        "filename": "DeepCrack.zip",
        "license": "Non-commercial research and educational purposes only",
        "expected_min_bytes": 70_000_000,
    },
    "pothole600": {
        "name": "Pothole-600",
        "url": "https://drive.usercontent.google.com/download?id=1c5fl0ktFnF4LBP0CDZD76_4ec_bg78Cz&export=download&confirm=t",
        "filename": "Pothole-600.zip",
        "license": "Academic research (see official project site)",
        "expected_min_bytes": 200_000_000,
    },
    "crack500_bundle": {
        "name": "Pavement Crack Datasets Bundle (Crack500, GAPs384, AEL, CrackTree200, CFD)",
        "url": "https://drive.google.com/uc?export=download&id=13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD",
        "filename": "pavement_crack_datasets_bundle.zip",
        "license": "Academic / non-commercial research (citation required)",
        "expected_min_bytes": 1_700_000_000,
    },
}


def resolve_gdrive_confirm_url(url: str) -> str:
    """Resolve Google Drive large-file virus-scan confirmation URL if an HTML interstitial is returned."""
    if "drive.google.com" not in url:
        return url
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read(65536)
    text = data.decode("utf-8", errors="ignore")
    if "download-form" not in text:
        return url
    id_match = re.search(r'name="id"\s+value="([^"]+)"', text)
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', text)
    confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', text)
    file_id = id_match.group(1) if id_match else ""
    confirm = confirm_match.group(1) if confirm_match else "t"
    resolved = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"
    if uuid_match:
        resolved += f"&uuid={uuid_match.group(1)}"
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_curl(url: str, dest: Path, original_url: str | None = None, max_attempts: int = 8) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    current_url = url
    for attempt in range(1, max_attempts + 1):
        cmd = [
            "curl.exe",
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--speed-limit",
            "10240",
            "--speed-time",
            "30",
            "-C",
            "-",
            "-o",
            str(dest),
            current_url,
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            return 0
        print(f"Attempt {attempt}/{max_attempts} failed with curl exit={proc.returncode}; refreshing URL and resuming...")
        if original_url:
            current_url = resolve_gdrive_confirm_url(original_url)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official RoadVision datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SPECS.keys()),
        default=["cfd", "deepcrack", "pothole600"],
        help="Dataset keys to download.",
    )
    parser.add_argument("--archives-dir", type=Path, default=Path("data/raw/archives"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/archives/download_manifest.json"))
    args = parser.parse_args()

    manifest = {}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    for key in args.datasets:
        spec = DATASET_SPECS[key]
        dest = args.archives_dir / spec["filename"]
        url = resolve_gdrive_confirm_url(spec["url"])
        print(f"[{key}] Downloading {spec['name']} -> {dest}")
        rc = download_with_curl(url, dest, original_url=spec["url"])
        if rc != 0 or not dest.exists():
            print(f"[{key}] ERROR: curl exited with {rc}", file=sys.stderr)
            return rc or 1

        head = dest.read_bytes()[:4]
        is_zip = head.startswith(b"PK")
        size_bytes = dest.stat().st_size
        print(f"[{key}] Done: {size_bytes:,} bytes | is_zip={is_zip}")
        if not is_zip or size_bytes < spec["expected_min_bytes"]:
            print(f"[{key}] WARNING: Archive failed basic size/ZIP header check.", file=sys.stderr)
            return 2

        manifest[key] = {
            "name": spec["name"],
            "url": spec["url"],
            "license": spec["license"],
            "filename": spec["filename"],
            "size_bytes": size_bytes,
            "sha256": sha256_file(dest),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
