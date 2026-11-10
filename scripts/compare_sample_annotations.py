"""Reproducible descriptive comparison to the WORKSHOP sample annotations.

This is not a model benchmark: the annotations may be incomplete or overlap
training data. Uses confidence-ordered one-to-one matching at IoU >= 0.50.
Does not touch or alter the scientific model or prediction records.

Run from project root:
    python scripts/compare_sample_annotations.py
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_images"
VALIDATION = ROOT / "validation"
SAMPLE = "OSBS_029"
IOU_THRESHOLD = 0.5
BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")


def box(record):
    return tuple(int(record[key]) for key in BOX_KEYS)


def intersection_over_union(box_a, box_b):
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def match_predictions(predictions, references, *, threshold=IOU_THRESHOLD):
    """Greedy, descending-score, one-to-one matching for a single class."""
    unmatched = set(range(len(references)))
    matches = []
    unmatched_prediction_ids = []
    for p in sorted(predictions, key=lambda item: (-item["score"], item["id"])):
        candidates = (
            (intersection_over_union(box(p), box(references[i])), i)
            for i in sorted(unmatched)
            if p["label"] == references[i]["label"]
        )
        iou, ref_idx = max(candidates, default=(0.0, None))
        if ref_idx is not None and iou >= threshold:
            unmatched.remove(ref_idx)
            matches.append({"prediction_id": p["id"], "reference_index": ref_idx,
                            "iou": round(iou, 6)})
        else:
            unmatched_prediction_ids.append(p["id"])
    return matches, unmatched_prediction_ids, sorted(unmatched)


def plot_comparison(predictions, references, output_path: Path):
    """Separate sources visually rather than suggest annotations are predictions."""
    with Image.open(SAMPLES / f"{SAMPLE}.png") as source:
        rgb = source.convert("RGB")
    width, height = rgb.size
    header = 42
    display = Image.new("RGB", (width * 2, height + header), (17, 23, 32))
    display.paste(rgb, (0, header))
    display.paste(rgb, (width, header))
    canvas = ImageDraw.Draw(display)
    canvas.text((9, 13), f"WORKSHOP ANNOTATIONS: {len(references)}", fill=(100, 230, 234))
    canvas.text((width + 9, 13), f"MODEL PREDICTIONS: {len(predictions)}", fill=(255, 208, 65))
    for ref in references:
        b = box(ref)
        canvas.rectangle((b[0], b[1] + header, b[2], b[3] + header),
                         outline=(100, 230, 234), width=2)
    for p in predictions:
        b = box(p)
        canvas.rectangle((b[0] + width, b[1] + header, b[2] + width, b[3] + header),
                         outline=(255, 208, 65), width=2)
    display.save(output_path)


def evaluate():
    with (SAMPLES / f"{SAMPLE}.csv").open(newline="", encoding="utf-8") as f:
        references = list(csv.DictReader(f))
    document = ET.parse(SAMPLES / f"{SAMPLE}.xml").getroot()
    annotated = [
        {"label": o.findtext("name"), **{k: o.findtext(f"bndbox/{k}") for k in BOX_KEYS}}
        for o in document.findall("object")
    ]
    if [(*box(r), r["label"]) for r in references] != [(*box(r), r["label"]) for r in annotated]:
        raise ValueError("CSV and XML annotations differ; cannot compare safely")
    with Image.open(SAMPLES / f"{SAMPLE}.png") as png:
        with Image.open(SAMPLES / f"{SAMPLE}.tif") as tif:
            png_rgb, tif_rgb = png.convert("RGB"), tif.convert("RGB")
            same_rgb = png_rgb.size == tif_rgb.size and ImageChops.difference(png_rgb, tif_rgb).getbbox() is None
    if not same_rgb:
        raise ValueError("PNG pixels are not identical to the annotated TIFF; comparison invalid")

    baseline = json.loads((VALIDATION / f"baseline_{SAMPLE}_cpu.json").read_text(encoding="utf-8"))
    if baseline["image_width"] != png_rgb.width or baseline["image_height"] != png_rgb.height:
        raise ValueError("Prediction and annotation image dimensions differ")
    predictions = baseline["predictions"]
    matches, unmatched_predictions, unmatched_references = match_predictions(predictions, references)
    report = {
        "purpose": "descriptive comparison only, not an independent performance benchmark",
        "image": f"{SAMPLE}.png",
        "annotation_source": f"{SAMPLE}.csv and .xml (refer to pixel-identical {SAMPLE}.tif)",
        "rgb_pixel_identical_png_tif": same_rgb,
        "annotation_labels": dict(Counter(r["label"] for r in references)),
        "model_output_labels": dict(Counter(p["label"] for p in predictions)),
        "matching": "confidence-descending, greedy, one-to-one; same-class",
        "iou_threshold": IOU_THRESHOLD,
        "n_reference_annotations": len(references),
        "n_predictions": len(predictions),
        "n_matched": len(matches),
        "unmatched_prediction_ids": unmatched_predictions,
        "unmatched_reference_indices_zero_based": unmatched_references,
        "annotation_coverage": round(len(matches) / len(references), 6) if references else None,
        "prediction_match_fraction": round(len(matches) / len(predictions), 6) if predictions else None,
        "matches": matches,
        "important_limitation": "No independent hold-out provenance or annotation exhaustiveness established",
    }
    # The SOAP_061 sample has class labels incompatible with the trained model.
    soap = ET.parse(SAMPLES / "SOAP_061.xml").getroot()
    report["soap_061_reference_labels"] = dict(Counter(o.findtext("name") for o in soap.findall("object")))
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "OSBS_029_annotation_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_comparison(predictions, references, VALIDATION / "OSBS_029_reference_vs_predictions.png")
    print(f"Supplied references={len(references)} predictions={len(predictions)} "
          f"matches@{IOU_THRESHOLD}={len(matches)} "
          f"unmatched predictions={len(unmatched_predictions)} "
          f"unmatched references={len(unmatched_references)}")
    print("SOAP_061 reference labels:", report["soap_061_reference_labels"])
    return report


def evaluate_yell_multitile():
    """Optional second descriptive comparison; no new model inference."""
    name = "2019_YELL_2_541000_4977000_image_crop"
    baseline_path = VALIDATION / "baseline_YELL_multitile_cpu.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Run the original detector on {name}.png first")
    result = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_xml = ET.parse(SAMPLES / f"{name}.xml").getroot()
    references = [
        {"label": o.findtext("name"), **{k: int(o.findtext(f"bndbox/{k}")) for k in BOX_KEYS}}
        for o in reference_xml.findall("object")
    ]
    if (int(reference_xml.findtext("size/width")), int(reference_xml.findtext("size/height"))) != (
        result["image_width"], result["image_height"]
    ):
        raise ValueError("YELL annotation size does not match the prediction image size")
    if result["settings"].get("n_windows") != 12:
        raise ValueError("Expected a 12-window original CLI baseline")
    matched, unmatched_predictions, unmatched_references = match_predictions(result["predictions"], references)
    report = {
        "purpose": "descriptive comparison only, not an independent performance benchmark",
        "sample": f"{name}.png",
        "reference_xml": f"{name}.xml",
        "annotation_labels": dict(Counter(r["label"] for r in references)),
        "model_output_labels": dict(Counter(p["label"] for p in result["predictions"])),
        "number_of_original_tile_windows": result["settings"]["n_windows"],
        "n_reference_annotations": len(references),
        "n_predictions": len(result["predictions"]),
        "matching": "confidence-descending greedy one-to-one IoU >= 0.5, same class",
        "n_matched": len(matched),
        "unmatched_prediction_count": len(unmatched_predictions),
        "unmatched_reference_count": len(unmatched_references),
        "important_limitation": "Annotation completeness and training/test independence not established",
    }
    (VALIDATION / "YELL_multitile_annotation_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"YELL reference annotations={len(references)}, predictions={len(result['predictions'])}, "
          f"matches@0.5={len(matched)}, unmatched predictions={len(unmatched_predictions)}, "
          f"unmatched references={len(unmatched_references)}")
    return report


if __name__ == "__main__":
    evaluate()
    evaluate_yell_multitile()
