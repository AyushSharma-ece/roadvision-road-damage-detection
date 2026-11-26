"""RoadVision Publication Chart Generator (Matplotlib Agg).

Reads verified JSON audit, segmentation benchmark, and model evaluation outputs
to generate publication-grade PNG charts for reports/ and docs/assets/.
Never fabricates metrics: only plots files/fields actually present on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dataset_distribution(audit_json: Path, out_png: Path) -> bool:
    if not audit_json.exists():
        return False
    data = json.loads(audit_json.read_text(encoding="utf-8"))
    rdd = data.get("rdd2022", {})
    countries = rdd.get("countries", {})
    if not countries:
        return False

    c_names = sorted(countries.keys())
    classes = ["D00", "D10", "D20", "D40"]
    colors = ["#2E86C1", "#28B463", "#F39C12", "#E74C3C"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Stacked per-country instance counts by class
    x = np.arange(len(c_names))
    bottom = np.zeros(len(c_names))
    for cls, col in zip(classes, colors):
        vals = np.array([countries[c].get("class_counts", {}).get(cls, 0) for c in c_names])
        ax1.bar(x, vals, bottom=bottom, label=cls, color=col, alpha=0.9)
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(c_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Annotated Bounding Boxes")
    ax1.set_title("RDD2022 Class Distribution by Country Subset")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Overall class distribution & train/val/test split counts
    cls_totals = [rdd.get("class_counts", {}).get(c, 0) for c in classes]
    bars = ax2.bar(classes, cls_totals, color=colors, alpha=0.9)
    for b, v in zip(bars, cls_totals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Total Verified Boxes")
    splits = rdd.get("split_counts", {})
    split_txt = ", ".join(f"{k}={v:,}" for k, v in sorted(splits.items()))
    ax2.set_title(f"RDD2022 Combined Classes ({split_txt})")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_segmentation_benchmarks(reports_dir: Path, out_png: Path) -> bool:
    files = [
        ("DeepCrack-test", reports_dir / "seg_bench_deepcrack.json"),
        ("Crack500-test", reports_dir / "seg_bench_crack500.json"),
        ("CFD (Cross-Domain)", reports_dir / "seg_bench_cfd.json"),
    ]
    loaded = []
    for label, p in files:
        if p.exists():
            loaded.append((label, json.loads(p.read_text(encoding="utf-8"))))
    if not loaded:
        return False

    methods = ["otsu_morph", "canny_morph", "hybrid_morphology", "dl_unet"]
    present_methods = [m for m in methods if any(row["Method"] == m for _, rows in loaded for row in rows)]
    colors = {"otsu_morph": "#5DADE2", "canny_morph": "#AF7AC5", "hybrid_morphology": "#F5B041", "dl_unet": "#1E8449"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    x = np.arange(len(loaded))
    width = 0.75 / max(1, len(present_methods))

    for i, m in enumerate(present_methods):
        dices = []
        ious = []
        for _, rows in loaded:
            hit = next((r for r in rows if r["Method"] == m), None)
            dices.append(hit["MeanDice"] if hit else 0.0)
            ious.append(hit["MeanIoU"] if hit else 0.0)
        offset = (i - (len(present_methods) - 1) / 2) * width
        b1 = ax1.bar(x + offset, dices, width=width, label=m, color=colors.get(m, "#7F8C8D"))
        ax2.bar(x + offset, ious, width=width, label=m, color=colors.get(m, "#7F8C8D"))
        for bar, val in zip(b1, dices):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    labels = [lbl for lbl, _ in loaded]
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(0, 1.0); ax1.set_ylabel("Mean Dice / F1"); ax1.set_title("Crack Segmentation Mean Dice Across Datasets"); ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylim(0, 1.0); ax2.set_ylabel("Mean IoU"); ax2.set_title("Crack Segmentation Mean IoU Across Datasets"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_model_comparison(eval_json: Path, out_png: Path) -> bool:
    if not eval_json.exists():
        return False
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not models:
        return False

    names = [m["name"] for m in models]
    metrics = ["mAP50", "precision", "recall", "f1"]
    m_colors = ["#1F618D", "#239B56", "#D68910", "#8E44AD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    x = np.arange(len(names))
    w = 0.18
    for idx, (met, col) in enumerate(zip(metrics, m_colors)):
        vals = [float(m.get(met) or 0.0) for m in models]
        bars = ax1.bar(x + (idx - 1.5) * w, vals, width=w, label=met, color=col)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha="right"); ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score"); ax1.set_title("Object Detection Accuracy on Held-Out RDD2022 Test Split")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    fps = [float(m.get("fps") or 0.0) for m in models]
    sizes = [float(m.get("size_mb") or 0.0) for m in models]
    ax2_t = ax2.twinx()
    b_fps = ax2.bar(x - 0.18, fps, width=0.34, color="#2E86C1", label="FPS (GPU/CPU)")
    b_sz = ax2_t.bar(x + 0.18, sizes, width=0.34, color="#E67E22", label="Model Size (MB)")
    for b, v in zip(b_fps, fps):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b_sz, sizes):
        ax2_t.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}MB", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("Throughput (FPS)", color="#2E86C1")
    ax2_t.set_ylabel("Checkpoint Size (MB)", color="#E67E22")
    ax2.set_title("Inference Throughput vs. Model Size")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return True


def main() -> int:
    reports = Path("reports")
    done = {
        "dataset_distribution": plot_dataset_distribution(Path("data/interim/dataset_audit_report.json"), reports / "dataset_distribution.png"),
        "segmentation_benchmark": plot_segmentation_benchmarks(reports, reports / "segmentation_benchmark_chart.png"),
        "model_comparison": plot_model_comparison(reports / "model_evaluation_summary.json", reports / "model_comparison_chart.png"),
    }
    print(json.dumps(done, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
