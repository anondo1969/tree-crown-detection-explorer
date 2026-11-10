
"""Real-model regression tests against the original detector.

Compare coordinates, labels, counts, settings and metadata exactly.
Allow small CPU floating-point differences in confidence scores only.
"""

import copy
import json
import os
from pathlib import Path

import pytest

from app_adapter import process_image, write_downloads
from tree_crown_detector import load_model


# Absolute tolerance for CPU confidence-score differences.
# This does not change the model or its scientific thresholds.
SCORE_ABS_TOL = 2e-5


def assert_baseline_equivalent(result: dict, baseline: dict) -> None:
    """Compare scientific outputs, tolerating only tiny score differences."""

    assert result.keys() == baseline.keys(), "Output fields changed"

    for key in baseline:
        if key != "predictions":
            assert result[key] == baseline[key], (
                f"Output metadata mismatch: {key}"
            )

    actual = result["predictions"]
    expected = baseline["predictions"]

    assert len(actual) == len(expected), "Prediction count differs"

    for i, (observed, reference) in enumerate(zip(actual, expected)):
        assert observed.keys() == reference.keys(), (
            f"Prediction {i}: output fields changed"
        )

        for field in reference:
            if field == "score":
                assert observed[field] == pytest.approx(
                    reference[field],
                    abs=SCORE_ABS_TOL,
                    rel=0,
                ), (
                    f"Prediction {i}: score differs by more "
                    f"than {SCORE_ABS_TOL}: "
                    f"observed={observed[field]}, "
                    f"baseline={reference[field]}"
                )
            else:
                assert observed[field] == reference[field], (
                    f"Prediction {i}: {field} differs: "
                    f"observed={observed[field]}, "
                    f"baseline={reference[field]}"
                )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="real model opt-in",
)
def test_real_sample_matches_original_cli(tmp_path):
    src = Path("sample_images/OSBS_029.png")
    baseline = json.loads(
        Path("validation/baseline_OSBS_029_cpu.json").read_text()
    )

    assert src.is_file()

    model = load_model("weights", device="cpu")
    result, overlay, rows, _ = process_image(src, model=model)

    assert_baseline_equivalent(result, baseline)

    assert overlay.size == (
        baseline["image_width"],
        baseline["image_height"],
    )
    assert len(rows) == result["n_predictions"]

    for row, prediction in zip(rows, result["predictions"]):
        assert row == [
            prediction[k]
            for k in (
                "id", "label", "score",
                "xmin", "ymin", "xmax", "ymax",
            )
        ]

    json_path, png_path = write_downloads(
        result, overlay, tmp_path
    )

    # The download must preserve the CURRENT model output exactly.
    assert json.loads(json_path.read_text()) == result
    assert png_path.is_file()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_MULTI_TILE_TESTS") != "1",
    reason="optional 12-window real-model check",
)
def test_multitile_yellowstone_matches_original_cli(tmp_path):
    """Check overlapping tiles and cross-tile merging."""

    src = Path(
        "sample_images/"
        "2019_YELL_2_541000_4977000_image_crop.png"
    )
    baseline = json.loads(
        Path("validation/baseline_YELL_multitile_cpu.json").read_text()
    )

    assert src.is_file()
    assert baseline["settings"]["n_windows"] == 12

    model = load_model("weights", device="cpu")
    result, overlay, rows, summary = process_image(
        src, model=model
    )

    assert_baseline_equivalent(result, baseline)

    assert result["n_predictions"] == len(rows)
    assert overlay.size == (1249, 1035)
    assert "**Tile windows:** 12" in summary

    raw_json, rendered_png = write_downloads(
        result, overlay, tmp_path
    )

    assert json.loads(raw_json.read_text()) == result
    assert rendered_png.exists()


def _tiny_baseline():
    return {
        "image_path": "sample_images/test.png",
        "image_width": 400,
        "image_height": 400,
        "model": "model-name",
        "device": "cpu",
        "settings": {
            "mode": "tile",
            "patch_size": 400,
            "n_windows": 1,
        },
        "coordinate_system": "pixels",
        "n_predictions": 1,
        "predictions": [{
            "id": 0,
            "label": "Tree",
            "label_id": 0,
            "score": 0.685865,
            "xmin": 2,
            "ymin": 3,
            "xmax": 15,
            "ymax": 20,
            "width": 13,
            "height": 17,
            "area": 221,
            "center_x": 8.5,
            "center_y": 11.5,
        }],
    }


def test_regression_accepts_small_score_roundoff():
    baseline = _tiny_baseline()
    result = copy.deepcopy(baseline)

    result["predictions"][0]["score"] += 0.000001

    assert_baseline_equivalent(result, baseline)


@pytest.mark.parametrize("field,value", [
    ("xmin", 3),
    ("label", "Not tree"),
    ("id", 1),
    ("center_x", 9.0),
    ("score", 0.686),
])
def test_regression_rejects_changed_prediction(field, value):
    baseline = _tiny_baseline()
    result = copy.deepcopy(baseline)

    result["predictions"][0][field] = value

    with pytest.raises(
        AssertionError,
        match=f"Prediction 0: {field}",
    ):
        assert_baseline_equivalent(result, baseline)


def test_regression_rejects_changed_settings():
    baseline = _tiny_baseline()
    result = copy.deepcopy(baseline)

    result["settings"]["n_windows"] = 2

    with pytest.raises(AssertionError, match="settings"):
        assert_baseline_equivalent(result, baseline)


def test_regression_rejects_removed_detections():
    baseline = _tiny_baseline()
    result = copy.deepcopy(baseline)

    result["predictions"] = []
    result["n_predictions"] = 0

    with pytest.raises(AssertionError, match="n_predictions"):
        assert_baseline_equivalent(result, baseline)
