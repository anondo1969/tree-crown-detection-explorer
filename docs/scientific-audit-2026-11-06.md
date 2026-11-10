# Skeptical research audit — Tree Crown Detection Explorer

**Date:** 6 November 2026. **Scope:** reviewed the delivered Gradio wrapper, the **actual** original detector from the user-uploaded DDLS 2026 workshop ZIP, the supplied pretrained weights, the workshop reference annotations and image files, and the corrected wrapper. No scientific inference code or defaults were changed.

## Verdict and scientific interpretation

**Research demonstration only; not a validated tree census or ecological analysis tool.** The app calls the original research implementation and does not invent detections, but matching the original code alone does not establish accuracy against reference data or generalization to new sensors and sites. The displayed rectangles are **predicted pixel-coordinate bounding boxes**, not segmented crown boundaries, precise tree positions, tree health classes, or geographically referenced map features.

### Reproducibility / methods preserved

- `tree_crown_detector.py` SHA-256 in both the original ZIP and the reviewed app: `5dc4f0058fdbda04e181914079f4543749c11961ad36a92be3290cbddd1533e8`.
- Uploaded `model.safetensors` SHA-256: `d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8`.
- `weights/config.json`: **one class**, `Tree: 0`. The supplied app does not predict health status, species, or biomass.
- Original methods remain: `tile` mode, 400-pixel square patches, 5% overlap, score cutoff **0.1**, internal detector NMS IoU **0.05**, tile-merge NMS IoU **0.15**. No threshold, preprocessing, architecture, or tile-merging changes were made.
- The original `predict` path is used for both 400×400 and 1249×1035 images with identical weight file and CPU settings. JSON fields, settings, prediction coordinates, labels, and scores are preserved by the app.
- The workshop OSBS GeoTIFF includes map/geotransform tags (`ModelPixelScaleTag` contains `0.1, 0.1, 0.0`), **but the app's annotated PNG and JSON do not carry its CRS or geotransform**. Do not use the PNG as a georeferenced raster or infer a physical crown area from a box's pixel area.

## Comparison with supplied reference annotations

**Matching protocol (descriptive, NOT an independent benchmark):** read supplied XML/CSV bounding boxes; sort original model predictions descending by their returned score, assign each prediction to the best remaining annotation **of the same class** with pixel-coordinate IoU ≥ 0.50, at most one prediction per annotation; count unpaired boxes. This is a conventional diagnostic with explicitly specified matching, not a published accuracy statistic. An unpaired prediction is **not necessarily a false positive**, because supplied annotations may not be exhaustive; an unmatched reference is **not automatically a model defect** without independent annotation-quality assessment. These supplied images may overlap the model's training data; split provenance was not established.

| Sample | Original image px | Original tile windows | Supplied reference annotations | Predicted boxes | 1:1 matches (IoU ≥ 0.5) | Unpaired predictions | Unpaired reference annotations |
|---|---:|---:|---:|---:|---:|---:|---:|
| `OSBS_029.png` | 400 × 400 | 1 | 61 | 55 | 39 | 16 | 22 |
| `2019_YELL_2_541000_4977000_image_crop.png` | 1249 × 1035 | 12 | 279 | 397 | 194 | 203 | 85 |

The supplied OSBS PNG and TIFF were confirmed **pixel-for-pixel identical RGB images**. `OSBS_029.csv` and `OSBS_029.xml` contained exactly the same **61** boxes and labels; the XML and CSV refer to the TIFF while the fixed baseline uses the equivalent PNG. Yellowstone XML size and image dimensions agree. Full machine-readable tallies: `validation/OSBS_029_annotation_comparison.json` and `validation/YELL_multitile_annotation_comparison.json`. An honest side-by-side view of the OSBS annotations and predictions, deliberately **not fused into one alleged ground-truth image**, is at `validation/OSBS_029_reference_vs_predictions.png`. Recompute using `python scripts/compare_sample_annotations.py`.

**Incompatible annotations:** `SOAP_061.xml` has **28 `Dead`** and **9 `Alive`** objects. These annotation classes are **not** model outputs; the model's config supports only **`Tree`**. No conclusion about mortality/health can be drawn from model boxes or scores. Other supplied images require site-specific validation before any scientific claim.

### Problems and corrections in the interface (scientific method unchanged)

| Finding in delivered app | Why it matters | Correction |
|---|---|---|
| One-image exact regression used only 400×400 with **one** patch | It never tested overlap, offset, multi-tile merge/NMS | Added actual original CLI reference on supplied 1249×1035 Yellowstone file (12 overlapping windows) and an opt-in exact adapter regression on CPU. |
| Old count/image/table/downloads persisted when a new upload was selected or a later run failed | Predicted boxes can be falsely attributed to another image | Clear all outputs on upload changes and before starting new inference; no stale download buttons. |
| A multipage TIFF would silently use the first page; EXIF-rotated JPEGs could display inconsistent orientation | Wrong source interpretation, invisible data discard, coordinate confusion | Reject multi-page/animated images and non-upright EXIF orientation with actionable errors. No automatic cropping, rotation, or pixel conversion added. |
| Single yellow overlay and count could be confused with ground truth | The app is not a reference-annotation comparison | Relabeled image/table and added explicit explanation: model predictions only, score not a probability, boxes not polygons or geographic geometry. |
| Summary said “tile” but gave no clue how much tiling occurred | A 400×400 single window looks like a multi-patch method | Added original `settings.n_windows` to display. |
| Downloaded PNG / JSON may appear spatially authoritative | PNG drops original GeoTIFF georeferencing; JSON stores temporary server `image_path` | Added explicit UI/documentation caveats. JSON numeric metadata **not changed**. Bounding-box `area` is pixels squared, not canopy area. |
| Optional CI was able to publish a container image | Unnecessary publication path for a research-demo review | Removed GHCR push/login/publish inputs. Kept **manual, local-build-only** validation workflow. Docker image now includes the workshop MIT `LICENSE`. |
| Local server defaulted to listening on all network interfaces without authentication | Potential unintended exposure of uploaded research imagery on a local network | Local launch now binds to **127.0.0.1** by default; Docker explicitly overrides to **0.0.0.0** within the container. External hosting still requires access-control and privacy review. |

The low 20 MiB / 2M-pixel upload cap is an **application resource limit**, not a model specification. One supplied Yellowstone example (`2299×2472`) exceeds it and cannot be submitted unchanged; this is made explicit rather than silently downsampling. Gradio file caching and temporary output retention (approximately hourly cleanup, subject to runtime behavior) are **not a suitable privacy or retention guarantee** for sensitive or public-service imagery. The app does not attempt geographic registration or independent evaluation, to avoid expanding beyond the defined scope.

## What was tested

### Tested by original delivery (rechecked against source)

- The archived scientific file hash and supplied weight hash; model configuration and sample labels.
- Baseline `OSBS_029.png` original CLI: 55 predictions on one window, 400×400, CPU. Existing baseline JSON saved in `validation/baseline_OSBS_029_cpu.json`.

### Tested during this review

- Original CLI on **real full Yellowstone sample** `2019_YELL_2_541000_4977000_image_crop.png` (1249×1035, 12 windows, **397** predictions), CPU, original defaults, baseline in `validation/baseline_YELL_multitile_cpu.json`. Measured ~30.35 seconds, peak child process RSS ~727 MB **on this test host**, not a Serve benchmark.
- **2 of 2 real-model CPU regression tests passed** (`RUN_MODEL_TESTS=1 RUN_MULTI_TILE_TESTS=1 pytest -q tests/test_integration.py`, 43.11 s test duration, 47.64 s including process startup, peak ~788 MB). Both adapter outputs exactly matched the respective original CLI JSON; comparisons use the same physical input image paths and CPU library versions.
- **19 fast unit/annotation tests passed** (`pytest -q tests/test_adapter.py tests/test_annotations.py`): pixel/byte limits, bad formats, file unchanged, empty input, zero-detections fixture, accurate table/PNG/JSON roundtrip, non-RGB rejection, multi-page TIFF and EXIF orientation rejection, model-delegation check, IoU matching, annotation tallies, and UI clear-output function.
- Annotation comparison reproduces the specified counts with independent CSV/XML parsing, checks exact PNG/TIFF RGB equality and annotation-label mismatch.
- Gradio served the interface over HTTP 200. The **real Gradio `/detect` upload API** matched both original CPU baselines: OSBS TIFF **55 boxes** (same pixels as reference PNG), and multi-tile Yellowstone PNG **397 boxes** with **12 windows**, downloaded JSON/PNG, exact rows/scores, and full-size image previews. Expected API errors occurred for grayscale PNG, multipage TIFF, non-upright EXIF JPEG and 5.68M-pixel input. A real uniform black RGB image returned **zero** predictions (not a mocked result), with an empty table and an annotated PNG whose decoded RGB pixels were **identical** to the input.
- **Independent package extraction check:** the reviewed **complete ZIP** was extracted to a fresh directory; `weights/model.safetensors` was a real 129,049,980-byte file (not a symlink) and both its checksum and the original detector's checksum were verified. **20 tests passed, 1 long multi-tile test skipped** in that extracted directory (19 fast tests plus the OSBS real-model regression). The long multi-tile regression passed previously in the development directory.
- UI event wiring was inspected and includes an upload-change reset and a button-click reset. **A real interactive Chromium browser test could not be executed in this environment** (`net::ERR_BLOCKED_BY_ADMINISTRATOR` on localhost), so the visual reset/race behavior remains unverified; an API test is not equivalent to a browser test.

## Verification still required before research use or deployment

1. **Independent evaluation** on annotated, representative, geographically and temporally held-out imagery; establish annotation completeness, acquisition resolution, and agreed IoU/score evaluation protocol. The above descriptive counts must not be represented as reported model precision/recall, ecological error rates, or generalization evidence.
2. **Ground sampling distance, sensor properties, radiometry, site conditions**, and image quality. The model was trained for approximately 10 cm/pixel RGB imagery; this app intentionally does not resample, tune thresholds, infer CRS, or compute map areas.
3. **Georeferencing**: For TIFF inputs the original geotransform/CRS is not exported by the app; annotate GIS data using an independently verified geospatial workflow. JSON `image_path` is an ephemeral Gradio upload path.
4. **CPU/GPU and package variants**: numerical equality is demonstrated only for the tested CPU environment; retest with target GPU/MPS and deployment dependencies.
5. **Deployment**: Python 3.11 Docker image build/run, current Serve host resource limits, full-size 2M-pixel worst-case runtime, file-cache cleanup under concurrency, network access controls, and distribution rights for samples/weights must be checked on the target host. SciLifeLab Serve currently documents **2 vCPU/4 GB RAM by default**; the observed CPU runtime/RSS on this authoring host is not a guarantee in that profile. No deployment, push, or publication has been performed.
6. **Full source review of ported DeepForest method** vs upstream DeepForest 2.1.0 was outside this UI audit; we confirmed the supplied implementation was unchanged, **not** equivalence to arbitrary new upstream versions.

## Commands to reproduce locally

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tree_crown_detector.py \
  sample_images/OSBS_029.png --device cpu -o /tmp/osbs_recheck.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tree_crown_detector.py \
  sample_images/2019_YELL_2_541000_4977000_image_crop.png --device cpu -o /tmp/yell_recheck.json
python scripts/compare_sample_annotations.py
python -m pytest -q tests/test_adapter.py tests/test_annotations.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RUN_MODEL_TESTS=1 RUN_MULTI_TILE_TESTS=1 \
  python -m pytest -q tests/test_integration.py
# Start server in another terminal, then use real uploaded files:
TREE_DETECTOR_DEVICE=cpu python main.py
python scripts/smoke_gradio.py http://127.0.0.1:7860
```

**External scientific background:** [DeepForest prebuilt-model documentation](https://deepforest.readthedocs.io/en/stable/user_guide/02_prebuilt.html), [weecology model card](https://huggingface.co/weecology/deepforest-tree) and [DeepForest source repository](https://github.com/weecology/DeepForest). These describe the intended model context, not a validation of the app on new research imagery.
