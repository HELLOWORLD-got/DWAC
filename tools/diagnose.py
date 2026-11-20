"""Measure box coverage, cross-view query alignment, and filtering on a saved model."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from training.data import MultiViewDataset, collate
from training.engine import build_model
from training.metrics import AssociationMetrics, decode, xyxy


def measure(prediction, target, foreground, visible_prediction, iou_threshold):
    visible = target["visible"]
    overlaps = torch.stack([
        box_iou(xyxy(predicted), xyxy(truth))
        for predicted, truth in zip(prediction["boxes"], target["boxes"])
    ]).nan_to_num(0.0)
    raw = (overlaps >= iou_threshold) & visible[:, None, :]
    kept = raw & prediction["present"][:, :, None]
    raw_same_query = raw.all(dim=0)
    shared = visible.all(dim=0)
    counts = {
        "visible_gt_boxes": int(visible.sum()),
        "raw_box_hits": int(raw.any(dim=1).sum()),
        "kept_box_hits": int(kept.any(dim=1).sum()),
        "shared_gt": int(shared.sum()),
        "shared_gt_with_raw_hit_in_every_view": int(
            (shared & raw.any(dim=1).all(dim=0)).sum()
        ),
        "raw_same_query_hits": int(raw_same_query.any(dim=0).sum()),
        "same_query_hits_with_foreground_filter": int(
            (raw_same_query & foreground[:, None]).any(dim=0).sum()
        ),
        "same_query_hits_with_visibility_filter": int(
            (raw_same_query & visible_prediction.all(dim=0)[:, None]).any(dim=0).sum()
        ),
        "kept_same_query_hits": int(kept.all(dim=0).any(dim=0).sum()),
    }
    return counts, overlaps.amax(dim=1)[visible]


@torch.inference_mode()
def scan(net, config, split, limit, batch_size, device):
    data, model, settings = config["data"], config["model"], config["evaluation"]
    dataset = MultiViewDataset(
        data[split], size=(data["height"], data["width"]),
        num_views=model["num_views"], num_classes=model["num_classes"],
        num_queries=model["num_queries"], augment=False,
    )
    count = min(len(dataset), limit)
    indices = [i * (len(dataset) - 1) // max(1, count - 1) for i in range(count)]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, collate_fn=collate)
    counts = Counter()
    best_ious = []
    association = AssociationMetrics(settings["iou_threshold"], settings["ucr_threshold"])
    for images, targets in loader:
        outputs = net([image.to(device) for image in images])
        predictions = decode(outputs, settings["confidence_threshold"])
        classes = outputs["class_logits"].float().argmax(dim=-1).cpu()
        foreground = classes != outputs["class_logits"].shape[-1] - 1
        visible_prediction = outputs["confidences"].float().cpu() >= settings["confidence_threshold"]
        for pred, target, fg, vis in zip(predictions, targets, foreground, visible_prediction):
            hits, ious = measure(pred, target, fg, vis, settings["iou_threshold"])
            counts.update(hits)
            best_ious.append(ious)
            association.update(pred, target)
    ious = torch.cat(best_ious)
    metrics = association.compute()
    return dict(
        samples=count, split_size=len(dataset),
        median_best_raw_iou=ious.median().item() if ious.numel() else None,
        **counts,
        MDA=metrics["MDA"], true_associations=metrics["true_associations"],
        predicted_associations=metrics["predicted_associations"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", default=48, type=int, help="Maximum samples per split")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.limit, args.batch_size) < 1:
        parser.error("--limit and --batch-size must be positive")
    if args.output and args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must be different from --checkpoint")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    net = build_model(config)
    net.load_state_dict(checkpoint["model"])
    net.to(device).eval()
    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "num_queries": config["model"]["num_queries"],
        "background_weight": config["training"].get("background_weight", 1.0),
        "iou_threshold": config["evaluation"]["iou_threshold"],
        "confidence_threshold": config["evaluation"]["confidence_threshold"],
    }
    del checkpoint

    for split in args.splits:
        print(f"Diagnosing {split} (up to {args.limit} samples, no augmentation) ...", file=sys.stderr)
        report[split] = scan(net, config, split, args.limit, args.batch_size, device)
    summary = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
