"""Fast isolated adapter tests; no real model weights needed."""

import json
from pathlib import Path

import pytest
from PIL import Image

from app_adapter import (
    MAX_FILE_BYTES, MAX_PIXELS, annotate_image, process_image,
    rows_from_predictions, summary_markdown, validate_upload, write_downloads,
)


def result_for(predictions, width=20, height=20):
    return {
        "image_path": "original.png", "image_width": width, "image_height": height,
        "model": "fixture", "device": "cpu",
        "settings": {"mode": "tile", "score_thresh": 0.1},
        "coordinate_system": "pixels, origin top-left", "n_predictions": len(predictions),
        "predictions": predictions,
    }


def sample_rgb(tmp_path):
    filename = tmp_path / "test.png"
    Image.new("RGB", (20, 20), (7, 8, 9)).save(filename)
    return filename


def test_valid_png_and_original_file_unchanged(tmp_path):
    image = sample_rgb(tmp_path)
    before = image.read_bytes()
    assert validate_upload(image) == image
    assert image.read_bytes() == before


def test_valid_tiff(tmp_path):
    image = tmp_path / "test.tif"
    Image.new("RGB", (20, 20)).save(image)
    assert validate_upload(image) == image


@pytest.mark.parametrize("kind", ["gray", "bad_extension", "corrupt", "missing"])
def test_bad_uploads(tmp_path, kind):
    if kind == "gray":
        path = tmp_path / "gray.png"
        Image.new("L", (20, 20)).save(path)
    elif kind == "bad_extension":
        path = tmp_path / "test.pdf"
        path.write_bytes(b"not an image")
    elif kind == "corrupt":
        path = tmp_path / "bad.jpg"
        path.write_bytes(b"not a jpeg")
    else:
        path = None
    with pytest.raises(ValueError):
        validate_upload(path)


def test_pixel_limit(tmp_path, monkeypatch):
    from app_adapter import MAX_PIXELS
    path = tmp_path / "large.png"
    Image.new("RGB", (101, 100)).save(path)
    monkeypatch.setattr("app_adapter.MAX_PIXELS", 10_000)
    with pytest.raises(ValueError, match="does not downsample"):
        validate_upload(path)


def test_byte_limit(tmp_path, monkeypatch):
    path = sample_rgb(tmp_path)
    monkeypatch.setattr("app_adapter.MAX_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="over 20 MiB"):
        validate_upload(path)


def test_exact_overlay_coordinates_and_no_predictions(tmp_path):
    path = sample_rgb(tmp_path)
    empty = result_for([])
    assert annotate_image(path, empty).tobytes() == Image.open(path).tobytes()
    p = {"id": 0, "label": "Tree", "score": 0.561234,
         "xmin": 4, "ymin": 5, "xmax": 9, "ymax": 10}
    result = result_for([p])
    overlay = annotate_image(path, result)
    assert overlay.getpixel((4, 5)) == (255, 202, 37)
    assert overlay.getpixel((9, 10)) == (255, 202, 37)
    assert overlay.getpixel((0, 0)) == (7, 8, 9)
    assert rows_from_predictions(result) == [[0, "Tree", 0.561234, 4, 5, 9, 10]]
    assert "1 candidate tree crowns" in summary_markdown(result)
    with pytest.raises(ValueError, match="dimensions"):
        annotate_image(path, result_for([], 40, 20))


def test_round_trip_json_and_png(tmp_path):
    image = sample_rgb(tmp_path)
    result = result_for([])
    json_path, png_path = write_downloads(result, annotate_image(image, result), tmp_path / "output")
    assert json.loads(json_path.read_text()) == result
    with Image.open(png_path) as rendered:
        assert rendered.size == (20, 20)
        assert rendered.tobytes() == Image.open(image).tobytes()


def test_process_delegates_directly_to_predict(tmp_path, monkeypatch):
    from app_adapter import process_image
    image = sample_rgb(tmp_path)
    result = result_for([])
    called = []

    def fake_predict(path, model):
        called.append((path, model))
        return result

    sentinel = object()
    monkeypatch.setattr("app_adapter.predict", fake_predict)
    raw, annotated, rows, summary = process_image(image, model=sentinel)
    assert called == [(image, sentinel)]
    assert raw is result
    assert rows == []
    assert annotated.size == (20, 20)
    assert "0 candidate tree crowns" in summary


def test_multipage_tiff_is_not_silently_truncated(tmp_path):
    path = tmp_path / "pages.tif"
    img1 = Image.new("RGB", (20, 20), (50, 60, 70))
    img2 = Image.new("RGB", (20, 20), (10, 20, 30))
    img1.save(path, save_all=True, append_images=[img2])
    with pytest.raises(ValueError, match="Multi-page"):
        validate_upload(path)


def test_exif_orientation_is_not_silently_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 30), (50, 60, 70))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match="Normalize image orientation"):
        validate_upload(path)


def test_summary_reports_actual_tile_count():
    result = result_for([])
    result["settings"]["n_windows"] = 12
    assert "**Tile windows:** 12" in summary_markdown(result)


def test_clear_all_results_on_new_input_or_retry():
    from main import clear_outputs

    overlay, summary, rows, json_path, png_path = clear_outputs()
    assert overlay is None
    assert "No analysis for the current image" in summary
    assert rows == [] and json_path is None and png_path is None
