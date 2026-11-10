"""Opt-in real-model baseline regression; RUN_MODEL_TESTS=1 pytest..."""

import json
import os
from pathlib import Path

import pytest

from app_adapter import process_image, write_downloads
from tree_crown_detector import load_model


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="real model opt-in")
def test_real_sample_matches_original_cli(tmp_path):
    src = Path("sample_images/OSBS_029.png")
    baseline = json.loads(Path("validation/baseline_OSBS_029_cpu.json").read_text())
    assert src.is_file()
    model = load_model("weights", device="cpu")
    result, overlay, rows, _ = process_image(src, model=model)
    assert result == baseline
    assert overlay.size == (baseline["image_width"], baseline["image_height"])
    assert len(rows) == result["n_predictions"]
    for row, prediction in zip(rows, result["predictions"]):
        assert row == [prediction[k] for k in ("id", "label", "score", "xmin", "ymin", "xmax", "ymax")]
    json_path, png_path = write_downloads(result, overlay, tmp_path)
    assert json.loads(json_path.read_text()) == baseline
    assert png_path.is_file()


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MULTI_TILE_TESTS") != "1", reason="optional 12-window real-model check")
def test_multitile_yellowstone_matches_original_cli(tmp_path):
    """Catch accidental changes to tile overlap, offsets or merge NMS.

    Unlike OSBS_029 (one window), the original 1249 x 1035 PNG uses 12
    overlapping windows and cross-tile box merging.
    """
    src = Path("sample_images/2019_YELL_2_541000_4977000_image_crop.png")
    baseline = json.loads(Path("validation/baseline_YELL_multitile_cpu.json").read_text())
    assert src.is_file()
    assert baseline["settings"]["n_windows"] == 12
    model = load_model("weights", device="cpu")
    result, overlay, rows, summary = process_image(src, model=model)
    assert result == baseline
    assert result["n_predictions"] == len(rows)
    assert overlay.size == (1249, 1035)
    assert "**Tile windows:** 12" in summary
    raw_json, rendered_png = write_downloads(result, overlay, tmp_path)
    assert json.loads(raw_json.read_text()) == baseline
    assert rendered_png.exists()
