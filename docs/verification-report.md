# Verification report — 6 November 2026

This is a record of **executed checks**, not an assertion that untested platforms were verified.

## Actual runtime

- Test host: Python **3.13.5**.
- PyTorch **2.10.0+cpu**, torchvision **0.25.0+cpu**, NumPy **2.3.5**, Pillow **12.3.0**, safetensors **0.7.0**, Gradio **6.5.1**, pandas **2.2.3**.
- The uploaded pretrained model SHA-256 was checked against Hugging Face `main` and matched: `d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8`.
- Model state dict loaded with no missing keys.
- Baseline command:

  ```sh
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tree_crown_detector.py \
    sample_images/OSBS_029.png --device cpu -o baseline.json
  ```

- Baseline: **55** detections on a **400 x 400 RGB** image using default tile mode. Saved as `validation/baseline_OSBS_029_cpu.json`.
- `tree_crown_detector.py` was verified **byte-for-byte identical** to the version inside the uploaded original workshop ZIP.

## Tests actually completed

1. **11 fast adapter tests passed**: valid PNG/TIFF, corrupt files, empty input, unsupported extension, grayscale mode rejection, pixel/file-size limits, exact overlay corners, zero-detection image unchanged, JSON round trip, table values, and delegation to the scientific function.
2. **1 real-model integration test passed**. The entire adapter result was exactly equal to the baseline JSON, including settings, metadata, scores, box coordinates and prediction ordering (same CPU runtime / sample input path).
3. **The Gradio server started locally and served the frontend**, HTTP 200 at `http://127.0.0.1:7866/` and API info.
4. **Real upload via Gradio Client passed**: 55 predictions, 55 table rows, downloadable prediction JSON, annotated PNG of 400 x 400, and an image preview. The JSON matched baseline in all fields except the expected transient `image_path` for the Gradio-uploaded copy.
5. **User-facing errors passed over the real Gradio API** for no file and grayscale PNG upload.

## Not tested / follow-up

- **Docker build and run not tested**: Docker executable was unavailable in the authoring container. Build the image on a Docker-capable host and test with a sample image before Serve use.
- **Python 3.11 deployment image not executed**: 3.11 was selected to match the Serve template; functional tests ran on the provided Python 3.13.5 environment.
- **Serve resources not benchmarked**: default CPU/RAM limits may require limiting image dimensions further. The demo has a 2-million-pixel limit, but worst-case memory/latency is not measured on the Serve 2-vCPU/4-GB profile.
- **No deployment/push/publication attempted**. Verify image licensing and hardware resource use before publishing.
- Exact predictions on different CPU/GPU architectures and dependency versions may not always be bitwise identical; record any differences and inspect them scientifically.

---

## Skeptical review addendum — 6 November 2026

The original report above records the state of the first delivery; it is retained for provenance. The **reviewed** application fixes presentation, upload validation, stale-output handling, default local binding, and CI publication complexity while leaving the research detector byte-identical. A dedicated, more detailed report is in [scientific-audit-2026-11-06.md](scientific-audit-2026-11-06.md).

- **19 fast tests passed** on Python 3.13.5, including new annotation-matching and input-orientation tests.
- **2 real-model CPU integration regressions passed** in 43.11 seconds (OSBS 55/1 window and Yellowstone 397/12 windows). Exact output equality to original-CLI JSON in the same environment.
- Original Yellowstone CLI run completed in 30.35s on 1249×1035 with peak child process RSS ~727 MiB (one specific host, not Serve).
- **Live Gradio API**: OSBS TIFF and Yellowstone real uploads matched the original CLI except for the expected transient upload path; grayscale, multi-frame TIFF, EXIF-rotated JPEG and oversized images were rejected; a black RGB sample returned **zero** actual detector boxes.
- **Browser UI interaction test NOT executed:** Chromium refused a connection to local Gradio with `net::ERR_BLOCKED_BY_ADMINISTRATOR`, despite HTTP/API access working. Upload-change resets are wired and unit-tested but not visually validated in a browser.
- **Docker/Serve NOT tested** (Docker unavailable); no deploy, push or publication. The CI workflow in the reviewed version no longer includes a publish path.
- Original and reference annotation comparisons are reproducible via `python scripts/compare_sample_annotations.py`; one-to-one IoU ≥ 0.50 matches: **OSBS 39 of 55 model boxes vs 61 references**, **YELL 194 of 397 model boxes vs 279 references**. Not an independent test set, not evidence of generalization.

- The final reviewed **complete ZIP** was extracted separately; supplied weights were a real file, hash-verified. **20 passed / 1 skipped** on that extracted copy (multi-tile test intentionally opt-in for this package smoke check). Zero-detection PNG pixels were confirmed identical to input through the live API.
