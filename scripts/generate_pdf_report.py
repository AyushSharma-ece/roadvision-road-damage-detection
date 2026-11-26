"""RoadVision PDF Inspection Report Generator (ReportLab).

Generates multi-page engineering PDF inspection reports including original vs.
annotated imagery, damage statistics, severity explanations, performance metrics,
and explicit 2D measurement / model limitations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _scaled_image(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    return Image(str(path), width=w * scale, height=h * scale)


def build_pdf_report(payload_json: Path, output_pdf: Path) -> Path:
    data = json.loads(payload_json.read_text(encoding="utf-8"))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RVTitle", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#123456"), spaceAfter=4
    )
    h2_style = ParagraphStyle(
        "RVH2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#1B4F72"), spaceBefore=10, spaceAfter=4
    )
    body_style = ParagraphStyle("RVBody", parent=styles["BodyText"], fontSize=9.5, leading=13)

    story = []
    story.append(Paragraph("ROADVISION – INTELLIGENT ROAD INSPECTION REPORT", title_style))
    story.append(
        Paragraph(
            f"Generated: {data.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | "
            f"Model: {data.get('model_backend', 'N/A')} ({data.get('model_version', '1.0')})",
            body_style,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Metadata summary table
    meta_rows = [
        ["Input Source", str(data.get("source_path", "N/A")), "Source Type", str(data.get("source_type", "image"))],
        ["Overall Severity", str(data.get("overall_severity", "N/A")), "Total Defects", str(data.get("damage_count", 0))],
        [
            "GPS Coordinates",
            f"{data.get('latitude')}, {data.get('longitude')}" if data.get("has_gps") else "Unavailable (Not Recorded)",
            "Inference Time",
            f"{data.get('inference_ms', 0):.1f} ms",
        ],
    ]
    meta_tbl = Table(meta_rows, colWidths=[3.3 * cm, 6.2 * cm, 3.3 * cm, 5.0 * cm])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAF2F8")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0C4DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_tbl)

    # Original and Annotated Images
    orig_str = str(data.get("original_image", "") or "").strip()
    ann_str = str(data.get("annotated_image", "") or "").strip()
    orig_path = Path(orig_str) if orig_str else None
    ann_path = Path(ann_str) if ann_str else None
    if (orig_path and orig_path.is_file()) or (ann_path and ann_path.is_file()):
        story.append(Paragraph("Visual Inspection Comparison", h2_style))
        img_row = []
        lbl_row = []
        if orig_path and orig_path.is_file():
            img_row.append(_scaled_image(orig_path, 8.4 * cm, 6.2 * cm))
            lbl_row.append(Paragraph("<b>Original Input Image</b>", body_style))
        if ann_path and ann_path.is_file():
            img_row.append(_scaled_image(ann_path, 8.4 * cm, 6.2 * cm))
            lbl_row.append(Paragraph("<b>RoadVision Annotated Detection Output</b>", body_style))
        img_tbl = Table([lbl_row, img_row], colWidths=[8.8 * cm] * len(img_row))
        img_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(img_tbl)

    # Per-Defect Details Table
    story.append(Paragraph("Detected Defects & Severity Analysis", h2_style))
    defects = data.get("defects", [])
    if isinstance(defects, dict):
        defects = [defects]
    if not defects:
        story.append(Paragraph("No road damage detections exceeded the active confidence threshold.", body_style))
    else:
        det_rows = [["ID", "Class", "Conf", "Area %", "Score", "Severity", "Engineering Explanation"]]
        for d in defects:
            det_rows.append(
                [
                    str(d.get("detection_id", "")),
                    str(d.get("class_code", "")),
                    f"{float(d.get('confidence', 0)):.2f}",
                    f"{float(d.get('relative_area', 0)) * 100:.2f}%",
                    f"{float(d.get('severity_score', 0)):.2f}",
                    str(d.get("severity_level", "")),
                    Paragraph(str(d.get("explanation", "")), ParagraphStyle("Cell", fontSize=7.5, leading=9.5)),
                ]
            )
        det_tbl = Table(det_rows, colWidths=[1.0 * cm, 1.5 * cm, 1.3 * cm, 1.6 * cm, 1.3 * cm, 1.8 * cm, 9.3 * cm])
        det_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4F72")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(det_tbl)

    # Recommendations & Model Limitations
    story.append(Paragraph("Inspection Recommendations", h2_style))
    sev = str(data.get("overall_severity", "None"))
    if sev == "High":
        rec = "Priority field maintenance inspection recommended: high-severity pavement distress or dense fatigue cracking detected."
    elif sev == "Medium":
        rec = "Schedule preventative crack sealing / surface patching and monitor progression in subsequent surveys."
    elif sev == "Low":
        rec = "Routine monitoring recommended; minor surface distress detected."
    else:
        rec = "No immediate defect action flagged at current threshold; retain baseline survey record."
    story.append(Paragraph(rec, body_style))

    story.append(Paragraph("Model & Measurement Limitations", h2_style))
    story.append(
        Paragraph(
            "1. Severity scores use calibrated image-space relative area, defect risk weighting, and detector confidence; "
            "they are not official IRC/ASTM pavement condition indices without physical scale calibration.<br/>"
            "2. Single 2D RGB photographs cannot measure true pothole depth or subsurface structural failure.<br/>"
            "3. Low-light, wet reflection glare, shadows, and severe motion blur may reduce detection recall.",
            body_style,
        )
    )

    doc.build(story)
    return output_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RoadVision PDF inspection report.")
    parser.add_argument("--payload-json", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    out = build_pdf_report(args.payload_json, args.output_pdf)
    print(f"PDF generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
