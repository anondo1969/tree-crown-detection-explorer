# Tree Crown Detection Explorer

A small **Gradio Blocks** interface for candidate tree-crown bounding-box detection in high-resolution RGB aerial images. It wraps the original `tree_crown_detector.py` from the [DDLS 2026 workshop](https://github.com/ScilifelabDataCentre/ddls-2026-workshop-materials/tree/main/tree_crown_detection) **without modifying that scientific file**. Detection is performed with the actual [weecology/deepforest-tree](https://huggingface.co/weecology/deepforest-tree) RetinaNet ResNet-50 FPN model.

This is a single-page local research demo, not a trained replacement model or a tree-census tool.

## Contents

- `tree_crown_detector.py` — unmodified original scientific model and inference code.
- `app_adapter.py` — validation, exact pixel-coordinate rectangle drawing, table/export formatting, and once-per-process model loading.
- `main.py` — Gradio upload, Detect Trees button, examples, summary, table, PNG and JSON downloads.
- `weights/config.json` — original class/architecture configuration.
- `weights/model.safetensors` — **must be placed here**, 129,049,980 bytes; intentionally excluded from Git commits.
- `sample_images/` — examples supplied in the original workshop zip.
- `validation/baseline_OSBS_029_cpu.json` — original-CLI baseline for the single-window OSBS example.
- `validation/baseline_YELL_multitile_cpu.json` — original-CLI baseline for the 12-window Yellowstone example.
- `docs/scientific-audit-2026-11-06.md` — research audit with annotation comparisons, actual tests, corrections and remaining risks.
- `tests/` — fast test suite and opt-in real-model regression test.
- `Dockerfile` and `.github/workflows/validate-and-build.yml` — Serve-compatible packaging and optional, manually triggered CI.
- `docs/original-exercise-README.md` and `docs/original-requirements.txt` — original upstream instructions and requirements for provenance.

## 1. Model weights

Use the original pre-trained model, not an alternative. Save it as `weights/model.safetensors` next to `weights/config.json`:

```sh
curl -fL --retry 3 https://huggingface.co/weecology/deepforest-tree/resolve/main/model.safetensors \
  -o weights/model.safetensors
```

If you already have the uploaded `model.safetensors`, copy it there instead. Reference checksum (SHA-256) for the upstream `main` weights used in this project:

```
d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8
```

Check with `sha256sum weights/model.safetensors` on Linux or `shasum -a 256 weights/model.safetensors` on macOS. The actual file size is **129,049,980 bytes** (about **123 MiB**). The Hugging Face model repository declares the **MIT** license; check the original repository and source-data licenses before public redistribution. The Docker build **copies a locally provided model file into the image**; it does not download it at startup. The optional GitHub Actions workflow downloads and verifies it at *CI build time*.

## 2. Local CPU installation and launch

Recommended: **Python 3.11**; other recent versions may work, but were not tested for this Dockerfile. Commands assume the project directory is the current directory.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements.txt
TREE_DETECTOR_DEVICE=cpu python main.py
```

Open **http://localhost:7860**. Local launches bind to **127.0.0.1** by default, not the whole network; the Dockerfile explicitly uses `GRADIO_SERVER_NAME=0.0.0.0` inside the container. Upload RGB PNG/JPEG/TIFF or select a provided example, then click **Detect Trees**. The original 400-pixel sample images are appropriate for a short CPU demo. On Windows use the equivalent venv activation and environment-variable commands. On Apple Silicon, install the appropriate official PyTorch builds instead of the Linux-oriented CPU index; the app's default device mode is `auto` if `TREE_DETECTOR_DEVICE` is unset.

### Input and output behavior

- Accepted: decodable **3-channel RGB** PNG, JPEG, and TIFF, **at most 20 MiB and 2,000,000 pixels**. These *application* limits are intended to protect a small workshop process; no image is silently resampled, recompressed, or cropped for inference. Multi-page TIFFs and images with non-upright EXIF orientation are rejected, rather than silently interpreting the wrong frame or orientation. The large workshop image `2019_YELL_2_528000_4978000_image_crop2.png` (2299 × 2472) exceeds this limit and is intentionally rejected.
- The uploaded **file path and original bytes** are handed to the original `predict(...)` function. The model's own original RGB conversion happens inside that unchanged function.
- One model instance is lazily loaded and cached per process. Requests are queued; only one inference runs concurrently.
- The overlay draws all returned boxes at the returned pixel coordinates, without filtering or score changes. The data table preserves returned numeric values. The UI clears stale results when input changes or a new run starts. A zero-detection image has zero rows, an unchanged RGB preview, and valid JSON. The preview can be visually rescaled by the browser: the PNG download has the original pixel dimensions.
- Downloads: annotated PNG and original-style JSON with full model metadata, settings, and every prediction. **Neither the PNG nor JSON is georeferenced**; the PNG strips GeoTIFF spatial metadata. The `image_path` field retains a **transient Gradio upload path**, not the original source file identity. A `prediction.area` value is **bounding-box pixel area**, not actual canopy area.
- Files are staged in temporary result directories and cleaned on process exit or after approximately one hour via subsequent requests; Gradio's file-cache cleanup also runs hourly. This is not archival storage. Avoid uploading sensitive imagery to a public service.

### Scientific defaults (not user-adjustable)

| Original parameter | Value |
| --- | --- |
| Inference mode | `tile` |
| Patch | `400 x 400` pixels |
| Patch overlap | `0.05` |
| Detector confidence cutoff | `0.1` |
| Detector internal NMS IoU | `0.05` |
| Tile merge NMS IoU | `0.15` |
| Device | `auto` / CPU default for documented Docker run |

These defaults live in the original `tree_crown_detector.py`; this application does **not** alter them. See `docs/original-exercise-README.md` and `tree_crown_detector.py` for the precise scientific implementation.

## 3. Baseline and tests

Replicate the baseline with the original command-line program:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tree_crown_detector.py \
  sample_images/OSBS_029.png --device cpu -o /tmp/tree-baseline-recheck.json
```

The supplied baseline was generated on CPU with PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu, and contains **55 predictions** for the **400 x 400** sample; importantly, this tests **only one tile**. The original command-line detector was also run on the 1249×1035 Yellowstone example: **12 overlapping tile windows and 397 candidate boxes**. Both baseline JSON files are retained for comparison with the adapter. See the [scientific audit](docs/scientific-audit-2026-11-06.md) for a comparison against supplied annotation files; this is **not an independent performance benchmark**.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_adapter.py tests/test_annotations.py
python scripts/compare_sample_annotations.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RUN_MODEL_TESTS=1 RUN_MULTI_TILE_TESTS=1 python -m pytest -q tests/test_integration.py
```

The model regression tests use real weights and run actual detector inferences, not mocks. With `python main.py` running in another terminal, also run `python scripts/smoke_gradio.py http://127.0.0.1:7860` for a genuine API upload regression. See `docs/verification-report.md` for the tests actually run (Python 3.13.5 on the authoring host; the Python 3.11 Docker image was not executed). If another platform exhibits numerical differences, compare with documented tolerances and report them; do not assume exact reproducibility across heterogeneous GPUs/versions. The fast unit tests do *not* load the model.

## 4. Build and run with Docker

Docker is not required for local use. The Dockerfile follows the [current Serve app template](https://github.com/ScilifelabDataCentre/serve-app-template) and [Gradio hosting guidance](https://serve.scilifelab.se/docs/application-hosting/gradio/): Python 3.11, non-root UID 1000, Gradio on 0.0.0.0:7860, writable temporary directories, and a CPU-only PyTorch dependency build.

Place model weights in `weights/` *before* building:

```sh
docker build --platform linux/amd64 -t tree-crown-explorer:local .
docker run --rm -p 7860:7860 tree-crown-explorer:local
```

Open **http://localhost:7860** and run a real sample inference. The Docker image includes the weights and is consequently large. Do not assume it meets a given registry, memory, storage, or CPU time budget before testing it. The current SciLifeLab Serve default profile is reported as 2 vCPU / 4 GB RAM; a real resource benchmark on the intended host is advisable. **A Docker image build and Serve deployment were not run in the authoring environment.**

## 5. Optional GitHub / Serve deployment (approval required)

Nothing is pushed or published automatically from this deliverable. The included GitHub workflow is **manual only** (`workflow_dispatch`), runs the unit and model regressions, and builds a local Docker image. It has **no publication or push logic**. The workflow fetches and checksum-verifies upstream weights before integration testing and building. Keep the weights out of Git (`.gitignore` already does this). Make a GHCR package public only after checking the project's licensing and approvals.

After a public image and code repository exist, the SciLifeLab Serve operator can register the Docker image URL, set **port 7860**, and link to the source code. Follow the [current Serve documentation](https://serve.scilifelab.se/docs/application-hosting/gradio/) for permission limitations, supported image architectures, and resource requirements. For research-app publication, ensure rights to all included sample imagery, code and model weights. The included Docker image contains the workshop `LICENSE` text for attribution.

## 6. Research limitations

The boxes identify **candidate visible tree crowns**, not verified individual trees. False positives and missed trees occur, and results are sensitive to landscape, image resolution, lighting, and training-data differences. The pretrained reference model is intended for imagery around **10 cm/pixel**. Bounding boxes are in **original-image pixels**, not geographic coordinates. Confidence scores should not be described as calibrated probabilities without separate calibration evidence. The sample annotations include 61 OSBS boxes and 279 boxes on the supplied Yellowstone crop; one-to-one IoU ≥ 0.5 matching found 39 and 194 matched boxes, respectively, but training/test independence is **unknown**. The `SOAP_061.xml` reference annotations say `Dead`/`Alive`; the model **does not output those classes**. The model does not estimate species, tree condition, biomass, ecological impacts, or other attributes. Validate its appropriateness for your research setting before relying on any outputs.

## License and attribution

The original scientific source and workshop exercise are retained as provided in the uploaded materials. The model repository reports MIT licensing. This project includes the workshop repository's `LICENSE`; consult [DeepForest](https://github.com/weecology/DeepForest) for original implementation provenance and applicable requirements. Sample imagery attribution/licensing should be confirmed before any public distribution or service deployment.
