"""Predict boxes and shared query IDs for one synchronized group."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from training.data import image_tensor
from training.engine import build_model
from training.metrics import decode, xyxy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="runs/inference")
    p.add_argument("--confidence", type=float)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["config"]
    if len(args.images) != cfg["model"]["num_views"]:
        p.error("Number of images must equal the model's configured views")
    threshold = args.confidence if args.confidence is not None else cfg["evaluation"]["confidence_threshold"]
    if not 0 <= threshold <= 1:
        p.error("--confidence must be between zero and one")
    device = torch.device(args.device)
    originals = []
    for path in args.images:
        with Image.open(path) as image:
            originals.append(image.convert("RGB"))
    size = (cfg["data"]["height"], cfg["data"]["width"])
    images = [image_tensor(im, size).unsqueeze(0).to(device) for im in originals]
    net = build_model(cfg).to(device).eval()
    net.load_state_dict(state["model"])
    with torch.no_grad():
        pred = decode(net(images), threshold)[0]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for v, im in enumerate(originals):
        w, h = im.size
        boxes = xyxy(pred["boxes"][v]) * torch.tensor([w, h, w, h])
        draw = ImageDraw.Draw(im)
        detections = []
        for query in pred["present"][v].nonzero().flatten().tolist():
            corners = boxes[query].tolist()
            color = (query * 73 % 256, query * 151 % 256, query * 181 % 256)
            draw.rectangle(corners, outline=color, width=3)
            draw.text(corners[:2], str(query), fill=color)
            detections.append(dict(query_id=query, label=int(pred["labels"][query]),
                                   score=float(pred["scores"][v, query]), xyxy=corners))
        im.save(output / f"view_{v+1}.jpg")
        result.append(dict(image=args.images[v], detections=detections))
    (output / "predictions.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
