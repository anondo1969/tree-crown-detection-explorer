"""Optional live-API review checks. Run after starting main.py on the specified URL.

Requires gradio_client and Pillow from the app's installed dependencies.
All comparisons are with actual original-CLI baselines, not simulated outputs.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
from PIL import Image, ImageChops

PROJECT = Path(__file__).resolve().parents[1]
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
CLIENT = Client(ENDPOINT, verbose=False)


def live(file_path):
    overlay, summary, table, json_file, png_file = CLIENT.predict(
        handle_file(str(file_path)), api_name="/detect"
    )
    result = json.loads(Path(json_file).read_text(encoding="utf-8"))
    assert result["n_predictions"] == len(table["data"])
    assert table["headers"] == ["ID", "Label", "Score", "xmin", "ymin", "xmax", "ymax"]
    assert all(row == [p[k] for k in ("id", "label", "score", "xmin", "ymin", "xmax", "ymax")]
               for row, p in zip(table["data"], result["predictions"]))
    with Image.open(overlay) as rendered:
        assert rendered.size == (result["image_width"], result["image_height"])
    with Image.open(png_file) as rendered:
        assert rendered.size == (result["image_width"], result["image_height"])
        if not result["predictions"]:
            with Image.open(file_path) as original:
                # True zero-detection images must have no overlay markings.
                assert ImageChops.difference(rendered.convert("RGB"), original.convert("RGB")).getbbox() is None
    assert f"{result['n_predictions']:,} candidate tree crowns" in summary
    assert f"**Tile windows:** {result['settings']['n_windows']}" in summary
    return result


def assert_matches_baseline(result, name):
    original = json.loads((PROJECT / "validation" / name).read_text(encoding="utf-8"))
    # Gradio's original upload path is deliberately different from the local CLI.
    assert {k: v for k, v in result.items() if k != "image_path"} == {
        k: v for k, v in original.items() if k != "image_path"
    }


def assert_rejected(path, description):
    try:
        live(path)
    except AppError as exc:
        # Other errors (e.g. network or assertion failures) do NOT count as rejection.
        print("PASS rejected", description, "->", type(exc).__name__)
    else:
        raise AssertionError(f"Unexpectedly accepted {description}: {path}")


def main():
    osbs_tiff = live(PROJECT / "sample_images/OSBS_029.tif")
    assert_matches_baseline(osbs_tiff, "baseline_OSBS_029_cpu.json")
    assert osbs_tiff["n_predictions"] == 55
    print("PASS OSBS TIFF: 55 boxes identical to pixel-equivalent PNG CLI baseline")

    yell = live(PROJECT / "sample_images/2019_YELL_2_541000_4977000_image_crop.png")
    assert_matches_baseline(yell, "baseline_YELL_multitile_cpu.json")
    assert yell["settings"]["n_windows"] == 12
    assert yell["n_predictions"] == 397
    print("PASS Yellowstone upload: 397 boxes identical to original CLI, 12 tiles")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gray = root / "gray.png"
        Image.new("L", (30, 30), 20).save(gray)
        assert_rejected(gray, "grayscale image")

        pages = root / "pages.tif"
        Image.new("RGB", (30, 30), (5, 20, 30)).save(
            pages, save_all=True, append_images=[Image.new("RGB", (30, 30), (5, 30, 10))]
        )
        assert_rejected(pages, "multipage TIFF")

        orient = root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (30, 30), (50, 20, 40)).save(orient, exif=exif)
        assert_rejected(orient, "EXIF-rotated JPEG")

        too_big = PROJECT / "sample_images/2019_YELL_2_528000_4978000_image_crop2.png"
        assert_rejected(too_big, "5.68 million pixel example")

        flat = root / "flat.png"
        Image.new("RGB", (400, 400), (0, 0, 0)).save(flat)
        empty = live(flat)
        print("Uniform black image returned", empty["n_predictions"], "boxes (real inference, not mocked)")
        if empty["n_predictions"] == 0:
            print("PASS real zero-detection path: no entries, valid JSON and 400x400 image")
        else:
            print("NOTE zero detections remain covered by unit fixtures, not by this uniform sample")

    print("PASS all completed live Gradio API regression / validation checks")


if __name__ == "__main__":
    main()
