"""
This script makes inference using the "weecology/deepforest-tree" model (RetinaNet, ResNet-50 FPN backbone):

Dependencies: torch, torchvision, numpy, pillow, safetensors.

Usage from Python:
    from tree_crown_detector import load_model, predict
    model = load_model("weights", device="cuda")      # or "cpu"
    result = predict("my_image.tif", model=model, output_json="my_image.json")

Usage from the command line:
    python tree_crown_detector.py my_image.png --device cpu --output result.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops import nms
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

# ---------------------------------------------------------------------------
# SETTINGS - edit these to change default behaviour
# ---------------------------------------------------------------------------
# "cpu", "cuda", "cuda:0", "cuda:1", "mps" (Apple silicon), or "auto"
# ("auto" = use the GPU if one is available, otherwise the CPU).
DEVICE = "auto"

# Folder holding config.json + model.safetensors
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# DeepForest defaults (conf/config.yaml). These override the values stored in
# the checkpoint's config.json.
SCORE_THRESH = 0.1     # minimum confidence for a box to be returned
NMS_THRESH = 0.05      # RetinaNet's internal per-class NMS IoU threshold

# Tiling (predict_tile) defaults. The model was trained on 400x400 px crops of
# ~10 cm resolution imagery, so large images should be tiled.
MODE = "tile"          # "tile" (= predict_tile) or "image" (= predict_image)
PATCH_SIZE = 400
PATCH_OVERLAP = 0.05
TILE_IOU_THRESH = 0.15  # NMS IoU used to merge boxes from overlapping tiles
# ---------------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None  # allow large orthomosaics


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    device = DEVICE if device is None else device
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_retinanet(num_classes: int = 1, score_thresh: float = SCORE_THRESH,
                    nms_thresh: float = NMS_THRESH) -> RetinaNet:
    """Architecture: torchvision's retinanet_resnet50_fpn
    backbone (ResNet-50 with frozen batch norm, FPN on C3-C5 + P6/P7) and the
    default RetinaNet anchors/heads/transform. No pretrained weights are
    downloaded here; the DeepForest weights are loaded afterwards."""
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        body, trainable_layers=3, returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    return RetinaNet(backbone=backbone, num_classes=num_classes,
                     score_thresh=score_thresh, nms_thresh=nms_thresh)


def _read_state_dict(weights_file: Path) -> dict:
    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(weights_file), device="cpu")
    else:  # .pt / .pth / .ckpt
        state = torch.load(str(weights_file), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:  # lightning checkpoint
            state = state["state_dict"]
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}


def load_model(weights_dir: str | Path = WEIGHTS_DIR, device: str | torch.device | None = None,
               score_thresh: float = SCORE_THRESH, nms_thresh: float = NMS_THRESH) -> RetinaNet:
    """Load the DeepForest tree model onto `device` ("cpu", "cuda", "auto", ...)."""
    weights_dir = Path(weights_dir)
    if weights_dir.is_file():
        weights_file, cfg_file = weights_dir, weights_dir.parent / "config.json"
    else:
        candidates = [weights_dir / n for n in ("model.safetensors", "pytorch_model.bin", "model.pt")]
        weights_file = next((p for p in candidates if p.exists()), None)
        cfg_file = weights_dir / "config.json"
        if weights_file is None:
            raise FileNotFoundError(
                f"No weights found in {weights_dir}. Run `python download_weights.py` first.")

    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    label_dict = cfg.get("label_dict") or {"Tree": 0}
    num_classes = int(cfg.get("num_classes") or len(label_dict))

    model = build_retinanet(num_classes, score_thresh=score_thresh, nms_thresh=nms_thresh)
    missing, unexpected = model.load_state_dict(_read_state_dict(weights_file), strict=False)
    if missing:
        raise RuntimeError(f"Weights do not match the architecture; missing keys: {missing[:10]}...")
    if unexpected:
        print(f"[tree_crown_detector] ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    dev = resolve_device(device)
    model.to(dev).eval()
    model.label_dict = {str(k): int(v) for k, v in label_dict.items()}
    model.numeric_to_label = {v: k for k, v in model.label_dict.items()}
    model.device = dev
    model.weights_file = str(weights_file)
    return model


# ---------------------------------------------------------------------------
# Image loading / tiling helpers
# ---------------------------------------------------------------------------
def _load_image(image) -> np.ndarray:
    """Return an HxWx3 float32 array in 0-255, like deepforest does (RGB)."""
    if isinstance(image, (str, Path)):
        arr = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-channel image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:  # CHW -> HWC
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected a 3-channel RGB image, got shape {arr.shape}")
    arr = arr.astype("float32")
    if arr.max() <= 1.0 and arr.min() >= 0:  # already 0-1 -> back to 0-255 convention
        arr = arr * 255.0
    return arr


def compute_windows(width: int, height: int, patch_size: int, patch_overlap: float):
    """Port of slidingwindow.generateForSize (used by deepforest.preprocess.compute_windows).
    Returns (x, y, w, h) tuples in the same x-major order."""
    if not 0 <= patch_overlap <= 1:
        raise ValueError("patch_overlap must be between 0 and 1")
    wx, wy = min(patch_size, width), min(patch_size, height)
    step_x = wx - int(math.floor(wx * patch_overlap))
    step_y = wy - int(math.floor(wy * patch_overlap))
    last_x, last_y = width - wx, height - wy
    xs = list(range(0, last_x + 1, step_x))
    ys = list(range(0, last_y + 1, step_y))
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    return [(x, y, wx, wy) for x in xs for y in ys]


def _to_rows(pred: dict, offset_x: int = 0, offset_y: int = 0):
    boxes = pred["boxes"].detach().cpu().numpy()
    return (boxes, pred["scores"].detach().cpu().numpy(),
            pred["labels"].detach().cpu().numpy(), offset_x, offset_y)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run(model: RetinaNet, chw: torch.Tensor):
    return model([chw.to(model.device)])[0]


def predict(image, model: RetinaNet | None = None, output_json: str | Path | None = None,
            mode: str = MODE, patch_size: int = PATCH_SIZE, patch_overlap: float = PATCH_OVERLAP,
            iou_threshold: float = TILE_IOU_THRESH, device: str | None = None,
            weights_dir: str | Path = WEIGHTS_DIR) -> dict:
    """Detect tree crowns in an image.

    Args:
        image: path to an image (png/jpg/tif...), a PIL image, or an RGB numpy array.
        model: a model from load_model(). Loaded on the fly if None (slower for many images).
        output_json: if given, the result is also written to this file.
        mode: "tile" (sliding window, like deepforest predict_tile) or
              "image" (whole image in one pass, like deepforest predict_image).
        patch_size, patch_overlap, iou_threshold: tiling parameters (mode="tile").
        device: only used when model is None.

    Returns:
        A JSON-serialisable dict with image info, settings and one entry per detected crown.
    """
    if model is None:
        model = load_model(weights_dir, device=device)

    img = _load_image(image)
    height, width = img.shape[:2]
    chw = torch.from_numpy(np.ascontiguousarray(np.moveaxis(img, -1, 0))) / 255.0

    if mode == "image":
        pred = _run(model, chw)
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()
        # deepforest predict_image: across-class NMS only when >1 label predicted
        if len(boxes) > 1 and labels.unique().numel() > 1:
            keep = nms(boxes.float(), scores.float(), float(model.nms_thresh))
            boxes = boxes[keep].to(torch.int).float()
            scores, labels = scores[keep], labels[keep]
        windows_used = 1
    elif mode == "tile":
        windows = compute_windows(width, height, patch_size, patch_overlap)
        all_boxes, all_scores, all_labels = [], [], []
        for (x, y, w, h) in windows:
            pred = _run(model, chw[:, y:y + h, x:x + w])
            if len(pred["boxes"]) == 0:
                continue
            b = pred["boxes"].cpu().clone()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            all_boxes.append(b)
            all_scores.append(pred["scores"].cpu())
            all_labels.append(pred["labels"].cpu())
        if all_boxes:
            # deepforest predict.mosiac: shift to image coords, cast to int, class-agnostic NMS
            boxes = torch.cat(all_boxes).to(torch.int).float()
            scores = torch.cat(all_scores).float()
            labels = torch.cat(all_labels)
            if len(boxes) > 1:
                keep = nms(boxes, scores, iou_threshold)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        else:
            boxes, scores, labels = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        windows_used = len(windows)
    else:
        raise ValueError("mode must be 'tile' or 'image'")

    predictions = []
    for i, (b, s, l) in enumerate(zip(boxes.tolist(), scores.tolist(), labels.tolist())):
        xmin, ymin, xmax, ymax = (round(v, 2) for v in b)
        if mode == "tile":
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        predictions.append({
            "id": i,
            "label": model.numeric_to_label.get(int(l), str(int(l))),
            "label_id": int(l),
            "score": round(float(s), 6),
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "width": round(xmax - xmin, 2), "height": round(ymax - ymin, 2),
            "center_x": round((xmin + xmax) / 2, 2), "center_y": round((ymin + ymax) / 2, 2),
            "area": round((xmax - xmin) * (ymax - ymin), 2),
        })

    result = {
        "image_path": str(image) if isinstance(image, (str, Path)) else None,
        "image_width": width,
        "image_height": height,
        "model": "weecology/deepforest-tree (RetinaNet ResNet-50 FPN)",
        "device": str(model.device),
        "settings": {
            "mode": mode,
            "score_thresh": float(model.score_thresh),
            "nms_thresh": float(model.nms_thresh),
            **({"patch_size": patch_size, "patch_overlap": patch_overlap,
                "tile_iou_threshold": iou_threshold, "n_windows": windows_used}
               if mode == "tile" else {}),
        },
        "coordinate_system": "pixels, origin top-left; boxes are [xmin, ymin, xmax, ymax]",
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    if output_json is not None:
        Path(output_json).write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description="DeepForest tree crown detection -> JSON")
    p.add_argument("image", help="path to an RGB image")
    p.add_argument("--output", "-o", help="output JSON path (default: <image>.json)")
    p.add_argument("--device", default=DEVICE, help='"cpu", "cuda", "cuda:1", "mps" or "auto"')
    p.add_argument("--weights", default=str(WEIGHTS_DIR), help="folder with model.safetensors + config.json")
    p.add_argument("--mode", default=MODE, choices=["tile", "image"])
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--patch-overlap", type=float, default=PATCH_OVERLAP)
    p.add_argument("--iou-threshold", type=float, default=TILE_IOU_THRESH)
    p.add_argument("--score-thresh", type=float, default=SCORE_THRESH)
    args = p.parse_args()

    model = load_model(args.weights, device=args.device, score_thresh=args.score_thresh)
    out = args.output or os.path.splitext(args.image)[0] + "_trees.json"
    res = predict(args.image, model=model, output_json=out, mode=args.mode,
                  patch_size=args.patch_size, patch_overlap=args.patch_overlap,
                  iou_threshold=args.iou_threshold)
    print(f"{res['n_predictions']} tree crowns detected on {res['device']} -> {out}")


if __name__ == "__main__":
    main()
