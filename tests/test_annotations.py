"""Annotation matching is audit-only and never changes inference."""

from scripts.compare_sample_annotations import (
    box, intersection_over_union, match_predictions, evaluate, evaluate_yell_multitile
)


def test_iou_uses_pixel_box_area_convention():
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert abs(intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-12


def test_matching_is_one_to_one_and_score_ordered():
    reference = [{"label": "Tree", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}]
    predictions = [
        {"id": 0, "label": "Tree", "score": 0.8, **{k: v for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (0, 0, 10, 10))}},
        {"id": 1, "label": "Tree", "score": 0.7, **{k: v for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (0, 0, 10, 10))}},
    ]
    matched, unmatched_pred, unmatched_ref = match_predictions(predictions, reference)
    assert len(matched) == 1
    assert matched[0]["prediction_id"] == 0
    assert unmatched_pred == [1]
    assert unmatched_ref == []


def test_osbs_comparison_reproduces_counts():
    result = evaluate()
    assert result["rgb_pixel_identical_png_tif"] is True
    assert (result["n_reference_annotations"], result["n_predictions"], result["n_matched"]) == (61, 55, 39)
    assert result["soap_061_reference_labels"] == {"Dead": 28, "Alive": 9}


def test_multitile_yell_comparison_reproduces_counts():
    result = evaluate_yell_multitile()
    assert (result["number_of_original_tile_windows"], result["n_reference_annotations"],
            result["n_predictions"], result["n_matched"]) == (12, 279, 397, 194)
