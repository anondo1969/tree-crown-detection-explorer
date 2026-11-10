"""Test a *running* local Gradio app with a real file upload and model inference.

Run `python main.py` in another terminal, then:
    python scripts/smoke_gradio.py http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gradio_client import Client, handle_file
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"

client = Client(ENDPOINT, verbose=False)
outputs = client.predict(handle_file(str(PROJECT / "sample_images/OSBS_029.png")), api_name="/detect")
assert len(outputs) == 5
image_path, summary, table, json_download, png_download = outputs
result = json.loads(Path(json_download).read_text(encoding="utf-8"))
baseline = json.loads((PROJECT / "validation/baseline_OSBS_029_cpu.json").read_text(encoding="utf-8"))
# Original CLI uses a local relative input path; Gradio uses a temporary upload
# path. The original scientific prediction records/settings must be identical.
assert {k: v for k, v in result.items() if k != "image_path"} == {
    k: v for k, v in baseline.items() if k != "image_path"
}
assert result["n_predictions"] == len(table["data"])
assert table["headers"] == ["ID", "Label", "Score", "xmin", "ymin", "xmax", "ymax"]
for row, prediction in zip(table["data"], result["predictions"]):
    assert row == [prediction[key] for key in ("id", "label", "score", "xmin", "ymin", "xmax", "ymax")]
with Image.open(image_path) as img:
    assert img.size == (result["image_width"], result["image_height"])
with Image.open(png_download) as img:
    assert img.size == (result["image_width"], result["image_height"])
assert f"{result['n_predictions']} candidate tree crowns" in summary
print(f"PASS: Gradio upload, {result['n_predictions']} real predictions, overlay, table and downloads")
