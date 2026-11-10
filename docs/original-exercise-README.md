*Before you start working with this example, download the `model.safetensors` file from here: https://huggingface.co/weecology/deepforest-tree/resolve/main/model.safetensors. Place it in the `weights/` folder.*

This folder contains a prediction script based on the pretrained model [weecology/deepforest-tree](https://huggingface.co/weecology/deepforest-tree). The model weights are in the folder `weights`. The prediction script file is `tree_crown_detector.py`. The file `requirements.txt` contains the required scripts to run this model. The folder `sample_images` contains example images that can be used to make an inference using this model as well as some annotation of these images that was provided by the model authors.

The model detects individual trees in high resolution RGB remote sensing imagery using deep learning. For reference see https://doi.org/10.1111/2041-210X.13472 (Weinstein et al 20219, Remote Sensing) and https://doi.org/10.3390/rs11111309 (Weinstein et al 2020, Methods in Ecology and Evolution). While the original authors implemented wrapping around their model, this folder contains only one model and a simplified prediction script so that it is more similar to a real world scenario where you would have your own model. The inference code is ported from DeepForest 2.1.0 (model definition, `predict_image`, `predict_tile`, sliding windows and box merging), with the same default thresholds.

## Setup

The lines below (to be run in Terminal) should work for most. Tested using Python 3.14.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # for GPU: install the CUDA build of torch from pytorch.org first
```

## Use for a single image

From command line:

```bash
python tree_crown_detector.py image.tif  # auto: GPU if available, else CPU
python tree_crown_detector.py image.tif --device cpu -o out.json
python tree_crown_detector.py image.tif --device cuda:0 --mode image
```

This will write a JSON file with the result in the same folder as the input image, with `_trees.json` added to the name.

From Python:

```python
from tree_crown_detector import load_model, predict

model = load_model("weights", device="cuda")   # "cpu", "cuda", "cuda:1", "mps" or "auto"
result = predict("image.tif", model=model, output_json="image_trees.json")
print(result["n_predictions"])
```

## Use for many images

Load the model once and call `predict` repeatedly when processing many images.

## CPU vs GPU

Either set `DEVICE` at the top of `tree_crown_detector.py`, pass `--device`, or pass `device=` to `load_model()`.

## Modes

- `tile` (default, = `deepforest.predict_tile`): splits the image into 400×400 px windows with 5 % overlap, predicts each and merges with NMS (IoU 0.15). Use this for orthomosaics / anything larger than ~400 px.
- `image` (= `deepforest.predict_image`): the whole image in one pass. Good for small crops.

The model expects RGB imagery at roughly 10 cm/pixel. At very different resolutions, resample the image or change `--patch-size`.

## Output JSON

```json
{
  "image_path": "image.tif", "image_width": 1210, "image_height": 950,
  "model": "...", "device": "cuda:0",
  "settings": {"mode": "tile", "score_thresh": 0.1, "nms_thresh": 0.05, "patch_size": 400, ...},
  "coordinate_system": "pixels, origin top-left; boxes are [xmin, ymin, xmax, ymax]",
  "n_predictions": 2,
  "predictions": [
    {"id": 0, "label": "Tree", "label_id": 0, "score": 0.83,
     "xmin": 760, "ymin": 10, "xmax": 790, "ymax": 44,
     "width": 30, "height": 34, "center_x": 775.0, "center_y": 27.0, "area": 1020}
  ]
}
```

Coordinates are in pixels. For GeoTIFFs, convert to map coordinates with the raster's geotransform (e.g. with `rasterio`: `src.transform * (x, y)`).

## License

The weights and original code are by the Weecology lab (DeepForest, MIT license); see
https://github.com/weecology/DeepForest.
