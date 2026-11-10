"""Non-scientific input checks and presentation for the unmodified detector.

No thresholds, network operations, tile merges, transforms, or predictions
are implemented here; the original tree_crown_detector is the only detector.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

from tree_crown_detector import WEIGHTS_DIR, load_model, predict

MAX_FILE_BYTES = 20 * 1024 * 1024  # workshop upload, not a resampling trigger
MAX_PIXELS = 2_000_000              # bound the number of costly 400-px tiles
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}
TABLE_HEADERS = ["ID", "Label", "Score", "xmin", "ymin", "xmax", "ymax"]


@lru_cache(maxsize=1)
def get_model():
    """Lazily load model weights exactly once per running app process.

    Default 'auto' retains the scientific script's CPU/GPU selection.
    Set TREE_DETECTOR_DEVICE=cpu for reproducible, CPU-only operation.
    """
    return load_model(WEIGHTS_DIR, device=os.getenv("TREE_DETECTOR_DEVICE", "auto"))


def validate_upload(image_path: str | Path | None) -> Path:
    """Check upload without resizing, changing image color or writing bytes."""
    if not image_path:
        raise ValueError("Upload a PNG, JPEG, or RGB TIFF image first.")
    path = Path(image_path)
    if not path.is_file():
        raise ValueError("The uploaded file could not be found. Please upload it again.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Unsupported extension. Please use PNG, JPEG, or TIFF.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File is over 20 MiB. This demo does not resize uploads.")
    try:
        # Inspect image headers and pixel count before decompressing large inputs.
        with Image.open(path) as img:
            width, height = img.size
            fmt, mode = img.format, img.mode
            if fmt not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format. Use PNG, JPEG, or TIFF.")
            if mode != "RGB":
                raise ValueError(
                    f"Expected 3-channel RGB imagery; this file is {mode!r}. "
                    "Convert it to RGB explicitly before uploading."
                )
            if getattr(img, "n_frames", 1) != 1:
                raise ValueError(
                    "Multi-page or animated images are not supported. "
                    "Export the intended RGB frame explicitly before uploading."
                )
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise ValueError(
                    f"Image contains {width} x {height} pixels. "
                    "This workshop demo accepts at most 2,000,000 pixels "
                    "and does not downsample."
                )
            img.verify()
        # verify() deliberately invalidates the decoder: reopen and decode
        # to catch truncated/corrupt image data before loading model weights.
        with Image.open(path) as img:
            # EXIF access may decode the file, so do it only AFTER the initial
            # header checks and verify() in the first open above.
            orientation = img.getexif().get(274, 1)
            if orientation not in (None, 1):
                raise ValueError(
                    f"EXIF orientation {orientation} is not supported. "
                    "Normalize image orientation explicitly before uploading."
                )
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError) as exc:
        raise ValueError("Cannot decode this image. Try a valid RGB PNG, JPEG, or TIFF.") from exc
    return path


def rows_from_predictions(result: dict[str, Any]) -> list[list[Any]]:
    """Expose each original prediction one-to-one; no score rounding/filtering."""
    return [
        [p["id"], p["label"], p["score"], p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
        for p in result["predictions"]
    ]


def annotate_image(image_path: str | Path, result: dict[str, Any]) -> Image.Image:
    """Draw exactly the original pixel-coordinate boxes; do not modify data."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != (result["image_width"], result["image_height"]):
        raise ValueError("Prediction dimensions do not match the original upload.")
    if not result["predictions"]:
        return image  # zero detections: pixel-identical original image
    drawer = ImageDraw.Draw(image)
    width = max(2, min(image.size) // 200)
    for prediction in result["predictions"]:
        drawer.rectangle(
            (
                prediction["xmin"], prediction["ymin"],
                prediction["xmax"], prediction["ymax"],
            ),
            outline=(255, 202, 37),
            width=width,
        )
    return image


def summary_markdown(result: dict[str, Any]) -> str:
    """Build only from the scientific result, never estimated or simulated."""
    settings = result["settings"]
    return (
        f"### {result['n_predictions']:,} candidate tree crowns\n\n"
        f"**Image:** {result['image_width']:,} x {result['image_height']:,} pixels  \n"
        f"**Model:** {result['model']}  \n"
        f"**Mode:** {settings['mode']}  \n"
        f"**Tile windows:** {settings.get('n_windows', 'not applicable')}  \n"
        f"**Device:** {result['device']}  \n"
        f"**Score threshold:** {settings['score_thresh']}  \n"
        f"**Coordinates:** pixels, top-left origin."
    )


def process_image(image_path: str | Path, *, model=None):
    """One call to the original predict() using unchanged scientific defaults.

    The explicit model parameter is for integration tests and application reuse.
    Returns a raw result, display-only image, and unaltered table values.
    """
    validated_path = validate_upload(image_path)
    selected_model = model if model is not None else get_model()
    result = predict(validated_path, model=selected_model)
    annotated = annotate_image(validated_path, result)
    return result, annotated, rows_from_predictions(result), summary_markdown(result)


def write_downloads(result: dict[str, Any], annotated: Image.Image, out_dir: Path):
    """Save faithful JSON metadata and a lossless annotated PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tree_crown_predictions.json"
    png_path = out_dir / "tree_crown_annotated.png"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    annotated.save(png_path, format="PNG")
    return json_path, png_path
