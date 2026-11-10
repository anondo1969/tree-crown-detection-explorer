"""Tree Crown Detection Explorer: a thin Gradio UI over the original detector."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

# Avoid external analytics requests for this local workshop demonstration.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from app_adapter import (
    MAX_FILE_BYTES, TABLE_HEADERS, process_image, write_downloads,
)
from tree_crown_detector import WEIGHTS_DIR

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.getenv("TREE_APP_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "tree-crown-explorer-outputs"))
)
OUTPUT_MAX_AGE_SECONDS = 60 * 60  # cached output data is not permanent
_THIS_PROCESS_OUTPUTS: list[Path] = []


def cleanup_stale_outputs() -> None:
    """Only delete this application's old result directories, never uploads."""
    if not OUTPUT_ROOT.is_dir():
        return
    cutoff = time.time() - OUTPUT_MAX_AGE_SECONDS
    for entry in OUTPUT_ROOT.iterdir():
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith("result-"):
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry)
            except OSError:
                LOGGER.warning("Unable to remove an old result directory")


def cleanup_own_outputs() -> None:
    for path in _THIS_PROCESS_OUTPUTS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(cleanup_own_outputs)


def clear_outputs():
    """Remove results whenever the input changes or a new run begins.

    This prevents a previous image's boxes/count from appearing to describe
    the currently selected file after an upload change or a failed run.
    """
    return None, "No analysis for the current image yet. Select **Detect Trees**.", [], None, None


def detect_trees(image_path: str | None):
    """Gradio event handler; all scientific inference occurs in process_image."""
    try:
        result, overlay, rows, summary = process_image(image_path)
        OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        cleanup_stale_outputs()
        out_dir = Path(tempfile.mkdtemp(prefix="result-", dir=OUTPUT_ROOT))
        _THIS_PROCESS_OUTPUTS.append(out_dir)
        json_file, annotated_file = write_downloads(result, overlay, out_dir)
        return overlay, summary, rows, str(json_file), str(annotated_file)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        # The traceback contains code locations only; do not log image bytes.
        LOGGER.exception("Prediction could not be completed")
        raise gr.Error(
            f"Tree detection failed ({type(exc).__name__}). "
            "Verify the model files and consult the app log."
        ) from exc


def build_app() -> gr.Blocks:
    samples = [
        PROJECT / "sample_images" / name
        for name in ("OSBS_029.png", "SOAP_031.png", "SOAP_061.png", "OSBS_029.tif")
    ]
    examples = [[str(p)] for p in samples if p.is_file()]

    with gr.Blocks(title="Tree Crown Detection Explorer", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# Tree Crown Detection Explorer\n"
            "Detect candidate individual tree crowns in high-resolution RGB aerial images "
            "using the existing **weecology/deepforest-tree** model. "
            "All scientific inference settings are fixed to the original script's defaults."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                image_input = gr.File(
                    label="Upload RGB aerial image (PNG, JPEG, TIFF)",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    type="filepath", file_count="single",
                )
                gr.Markdown(
                    "**Demo limits:** 20 MiB and 2 million pixels. "
                    "Oversized images are rejected, never resized or cropped."
                )
                detect_button = gr.Button("Detect Trees", variant="primary")
                if examples:
                    gr.Examples(
                        examples=examples, inputs=[image_input],
                        label="Try a supplied sample image",
                        cache_examples=False,
                    )
            with gr.Column(scale=2, min_width=340):
                overlay_output = gr.Image(
                    label="MODEL PREDICTIONS ONLY — yellow bounding boxes (not reference annotations)",
                    interactive=False, format="png", height=430,
                )
                summary_output = gr.Markdown(
                    "Upload an image and select **Detect Trees** to see model results."
                )
        result_table = gr.Dataframe(
            label="Model-predicted boxes (unfiltered; scores are not calibrated probabilities)",
            headers=TABLE_HEADERS,
            datatype=["number", "str", "number", "number", "number", "number", "number"],
            interactive=False, wrap=False, max_height=380,
        )
        with gr.Row():
            json_download = gr.File(label="Download complete predictions (JSON)")
            png_download = gr.File(label="Download annotated image (PNG)")

        gr.Markdown(
            "**Reading this display:** The preview may be visually scaled by the browser; "
            "download the PNG for full-resolution pixels. The boxes are model predictions "
            "only, not polygons, reference annotations, or geographic map features. "
            "The annotated PNG is **not a georeferenced GeoTIFF**. In the JSON, "
            "`image_path` is the temporary server upload path, **not** a stable dataset identifier."
        )
        gr.Markdown(
            "### Scientific interpretation and limitations\n"
            "These boxes are **model-predicted candidate tree crowns**, not ground-truth "
            "tree counts. The model can miss trees or return false detections. "
            "Performance depends on resolution, landscape, lighting, and similarity "
            "to the training domain (roughly **10 cm/pixel** imagery). "
            "Coordinates are **image pixels**, not georeferenced map positions. "
            "Confidence scores have **not** been established here as calibrated "
            "probabilities. The model does not determine species, health, biomass, "
            "or ecological impact.\n\n"
            "Method: pretrained RetinaNet ResNet-50 FPN, 400 x 400 px tiling, "
            "5% overlap, detector score cutoff 0.1, internal NMS IoU 0.05, "
            "tile merge NMS IoU 0.15, all unchanged from the supplied code."
        )
        outputs = [overlay_output, summary_output, result_table, json_download, png_download]
        image_input.change(
            fn=clear_outputs, outputs=outputs, api_name=False, show_progress="hidden"
        )
        # Clear previous outputs before each new prediction too; an error must
        # not leave old boxes or an old count visible for a different upload.
        detect_button.click(
            fn=clear_outputs, outputs=outputs, api_name=False, show_progress="hidden"
        ).then(
            fn=detect_trees, inputs=image_input, outputs=outputs,
            api_name="detect", concurrency_limit=1, show_progress="full",
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    if not (WEIGHTS_DIR / "config.json").is_file():
        raise SystemExit("Missing weights/config.json; restore it from the workshop materials.")
    if not (WEIGHTS_DIR / "model.safetensors").is_file():
        raise SystemExit(
            "Missing weights/model.safetensors. Download it from "
            "https://huggingface.co/weecology/deepforest-tree/resolve/main/model.safetensors"
        )
    cleanup_stale_outputs()
    logging.basicConfig(level=logging.INFO)
    demo = build_app()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        max_file_size=MAX_FILE_BYTES,
        show_error=False,
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
