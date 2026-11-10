# Coding-agent prompt 2: Tree Crown Detection Interface (Gradio)

> Copy this entire prompt into your coding assistant. It is self-contained, and can also be used alongside the `I already have some code` prompt in the SciLifeLab Serve app template README. Give the agent access to the workshop materials `tree_crown_detection/` folder. The pretrained model weights must also be available locally or downloadable.

## Role and task

Act as a careful research software engineer. Create a **small, elegant Gradio application** named **Tree Crown Detection Explorer** around an **existing and scientifically defined inference implementation**. This is primarily a **web interface task, not a machine-learning research task**.

Read these inputs before editing anything:

- Workshop materials: https://github.com/ScilifelabDataCentre/ddls-2026-workshop-materials
- Existing project: `tree_crown_detection/`
- Exercise instructions: `tree_crown_detection/README.md`
- Scientific code: `tree_crown_detection/tree_crown_detector.py`
- Dependencies: `tree_crown_detection/requirements.txt`
- Model configuration: `tree_crown_detection/weights/config.json`
- Sample images and annotations: `tree_crown_detection/sample_images/`
- Pretrained model: https://huggingface.co/weecology/deepforest-tree
- Web app template: https://github.com/ScilifelabDataCentre/serve-app-template

Use the **current** Serve template's `I already have some code` workflow and its Gradio guidance. The template should help build a clean, simple, reproducible app whether or not it is eventually published on Serve. Preserve the scientific code and defaults. Create the web app in a separate project directory or an isolated copy of the exercise; do not alter the original materials.

**If you cannot access the repository or scientific files, say exactly what is missing. Do not claim you read or tested unavailable files, invent a replacement model, or generate simulated detections and label them as real.**

## 1. Scientific specification: already implemented

**Research task:** Detect individual apparent tree crowns in high-resolution RGB aerial imagery using the supplied pretrained tree detector.

**Input:** A user-provided RGB image, ideally of comparable resolution and characteristics to the model's training imagery.

**Scientific implementation:** The existing `tree_crown_detector.py`, which uses the pretrained `weecology/deepforest-tree` model (RetinaNet with ResNet-50 FPN) and exports bounding boxes and confidence scores.

**Output:** The detector's prediction records, plus a user-friendly image overlay and tabular/downloadable representation. Bounding boxes use pixel coordinates in the original image.

**Non-goals:** Do not train or fine-tune a model, change the model architecture, create a new predictor, add object tracking, estimate tree species/health/biomass, or change detection thresholds to make the output look better.

**Research responsibility:** The user decides whether the model is scientifically appropriate for their use case. Preserve the provided method, explain limitations, and ask before making an important scientific change.

## 2. Prerequisites and baseline inference, BEFORE any Gradio UI

First read the actual README and Python script. In particular, verify the existing APIs:

```python
from tree_crown_detector import load_model, predict
model = load_model("weights", device="auto")
result = predict("path/to/image.tif", model=model)
```

Inspect the exact function signatures in the checked-out code before using them. The README specifies that `weights/model.safetensors` must be downloaded separately from:

https://huggingface.co/weecology/deepforest-tree/resolve/main/model.safetensors

It should be placed alongside the supplied `weights/config.json`. Check that both exist and that the weights can actually be loaded. If downloading is necessary and network access is available, follow the README. Otherwise, report the missing prerequisite and provide the exact download/placement steps. Do not fabricate a successful model load.

Inspect `requirements.txt` and select a compatible Python/PyTorch/torchvision environment, including platform-specific GPU requirements when relevant. Prefer a **CPU-compatible default** so the app can be demonstrated without a GPU. Do not promise that heavyweight dependencies will install or run without testing them.

Run the original command-line prediction on a supplied sample image, for example by following the README's invocation shape:

```bash
python tree_crown_detector.py PATH_TO_SAMPLE_IMAGE --device cpu -o baseline.json
```

Use a **real** sample image path discovered in `sample_images/`. Check the resulting JSON for model metadata, image size, settings, number of predictions, and prediction coordinates. Save this baseline for later comparison. If running inference fails, diagnose and report the failure before claiming the interface is valid.

## 3. Preserve original inference behavior

The existing source defines, among other settings:

- Default inference mode: `tile`.
- Tile size: 400 x 400 pixels.
- Overlap: 0.05 (5%).
- Tile-level box merge NMS IoU: 0.15.
- Detector score threshold: 0.1.
- Detector internal NMS threshold: 0.05.
- Automatic device selection where supported.

**Do not change these defaults.** Do not add user-adjustable scientific settings in the first version. Use the existing `load_model()` and `predict()` functions rather than duplicating the network, tiling, image transformations, filtering, or post-processing inside the UI.

The existing result dictionary includes `image_width`, `image_height`, `model`, `device`, `settings`, `coordinate_system`, `n_predictions`, and `predictions`. Each prediction has fields including `label`, `score`, `xmin`, `ymin`, `xmax`, `ymax`.

Treat the result dictionary as the source of truth. An annotation overlay must faithfully represent these exact returned boxes and must not modify or filter the predictions. Preserve result precision and settings in the JSON output. Do not invent detections when the model returns none.

## 4. Separate scientific code from the interface code

Keep the original scientific implementation unchanged if possible:

- `tree_crown_detector.py`: original model loading and inference.
- Small independent integration/formatting helper (optional): convert output into display-safe table and annotated image.
- Gradio entry-point module: handle interactions and display results.

Load the model **once per application process** and reuse it between requests. Do not reload model weights on every button click. A small amount of adapter code is fine; scientific algorithm changes are not.

Implement and test any adapter/annotation functions **before** creating the complete UI. Keep them testable without Gradio.

## 5. Create the Gradio interface

**App title:** Tree Crown Detection Explorer

Use Gradio Blocks to create a focused single-page interface.

### Inputs

- Accept an RGB image via a file-upload component. Support common image formats (PNG/JPEG) and TIFF where the installed image-reading libraries support it.
- Provide a few examples from the repository's `sample_images/` folder where reasonable.
- A prominent **Detect Trees** button.
- Do not require users to know Python, Docker, or the command line.

Prefer an upload component that retains the **original file path and bytes**, especially for TIFF. Avoid UI operations that silently resize or recompress an input before passing it to the detector. Validate that the uploaded file is a decodable RGB image. Set reasonable documented size/pixel limits for this workshop demo and present a helpful error when a file exceeds them. Never silently downsample, crop, or change scientific inputs to meet a limit.

### Outputs

1. **Annotated image** showing the original RGB image with visible boxes at the model's exact returned pixel coordinates. Overlay numbers or labels only if they do not obscure the interpretation.
2. **Detection summary** showing `n_predictions`, image dimensions, model identifier, inference mode and device, all derived from the actual result.
3. **Detection table** with ID, class label, confidence score, `xmin`, `ymin`, `xmax`, and `ymax`.
4. **Downloads** for the complete original-style prediction JSON and annotated PNG.
5. **Short scientific explanation** of what the model predicts and limitations of the resulting boxes.

A simple layout is enough: upload and action on the left, annotated image on the right, results table and downloads below. Use clean defaults and avoid excess CSS, dashboards, databases, complicated navigation, or authentication systems.

### Errors and resource handling

- An empty upload, unsupported file, corrupt image, failed inference, or too-large image must produce a clear user-facing message rather than a silent exception.
- Handle an image with zero detections correctly: empty table, count zero, original image unchanged (apart from any informational caption), valid JSON.
- Use per-request temporary paths for downloads and clean up stale temporary files where possible.
- Do not log uploaded image content or retain user images/results beyond what the app requires for the active session.
- Since loading a large model or tiled imagery can be resource-intensive, use sensible queue/concurrency limits without changing inference behavior.

## 6. Scientific limitations displayed in the app

Clearly communicate that:

- These are **model-predicted candidate tree crowns**, not verified ground-truth counts.
- The model can miss trees and can produce false detections.
- Image resolution, landscape, lighting and the training domain affect performance.
- The reference model is intended for imagery around **10 cm per pixel**; results on different imagery require caution.
- Bounding box coordinates are in **image pixels**, not automatically in geographic map coordinates.
- The model does **not** infer tree species, health, biomass, or ecological consequences.

Do not present a confidence score as a calibrated probability unless such calibration is established by the supplied scientific documentation.

## 7. Verification must compare the new app with the existing script

Using an identical sample image and the same model weights, device mode, and inference settings:

1. Run the original `tree_crown_detector.py` to produce the baseline JSON.
2. Call the inference through the new adapter/Gradio path.
3. Compare `n_predictions`, labels, bounding boxes, confidence scores, and settings. Use exact comparisons where deterministic and documented numerical tolerances only where necessary.
4. Check that the annotations align with the original image dimensions.
5. Verify every result table row corresponds to exactly one prediction in JSON.
6. Verify the downloadable JSON preserves the complete prediction data and metadata.
7. Test zero-detection behavior, invalid inputs, and input-size limits.
8. If hardware or numerical nondeterminism creates legitimate differences, record them; do not silently assert the outputs are identical.

Add a small number of automated tests for formatting, input validation, and overlays that do not require running the full pretrained model. Perform at least one actual end-to-end inference test if the weights and runtime are available. Never describe a mocked test as an end-to-end model verification.

## 8. Docker, template compatibility, and optional Serve deployment

Follow the **current** SciLifeLab Serve template README and Gradio-specific deployment requirements. Keep the application reproducible and the setup as simple as possible:

- Minimal, compatible dependencies and a documented Python version.
- An explicit Gradio launch that listens on the host/port required by the selected profile.
- Non-root container execution where the template requires it.
- Correct file and weight placement for local and container environments.
- No automatic remote download of model weights on every application startup.
- No persistent storage of uploaded images.
- Dockerfile, optional GitHub Actions build, and README with exact build/run/test instructions.

Check the real sizes and license/distribution conditions of model weights and dependencies before deciding how to include them in a Docker image. If an image cannot reasonably be built or run under available CPU, memory or registry constraints, say what is unverified and provide a reasonable documented path forward. Never silently replace the trained model with a lightweight substitute.

**A functioning local app is the primary deliverable.** Publishing on SciLifeLab Serve is optional. Do not push, publish or deploy it without my approval.

## 9. Execution order and required deliverables

Work sequentially:

1. Inspect source README, Python implementation, dependencies, sample images and model assets.
2. Confirm prerequisites and perform baseline inference on one actual sample image.
3. Make a thin, independently testable adapter around `load_model()` and `predict()` while preserving the original source.
4. Verify the adapter output against the baseline.
5. Build the Gradio Blocks interface.
6. Run the app, inspect a real upload and compare the web output with the baseline.
7. Add tests, Docker files, template-aligned configuration and an exact-command README.
8. Summarize **what actually ran**, any open issues, and what the user should verify locally.

Deliver all source files needed to run the app, dependency information, the original model code or clear reference to it, a small test suite, Docker configuration, model-weight setup instructions, and a short scientific limitations section. Keep the first working version minimal and honest about any untested behavior.
