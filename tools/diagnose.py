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


def geometry_counts(predicted, truth, visible, threshold):
    """GT-assisted center/size counterfactuals, not valid detection scores."""
    truth = truth[visible]
    if not len(truth):
        return dict(visible_gt_boxes=0, raw_box_hits=0,
                    oracle_center_box_hits=0, oracle_size_box_hits=0)

    def paired_iou(a, b):
        a, b = xyxy(a), xyxy(b)
        intersection = (torch.minimum(a[..., 2:], b[..., 2:]) -
                        torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
        area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
        area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
        return intersection / (area_a + area_b - intersection).clamp_min(1e-8)

    pred = predicted[:, None, :].expand(-1, len(truth), -1)
    gt = truth[None, :, :].expand(len(predicted), -1, -1)
    center_corrected = torch.cat([gt[..., :2], pred[..., 2:]], -1)
    size_corrected = torch.cat([pred[..., :2], gt[..., 2:]], -1)
    return dict(
        visible_gt_boxes=len(truth),
        raw_box_hits=int((paired_iou(pred, gt) >= threshold).any(0).sum()),
        oracle_center_box_hits=int((paired_iou(center_corrected, gt) >= threshold).any(0).sum()),
        oracle_size_box_hits=int((paired_iou(size_corrected, gt) >= threshold).any(0).sum()),
    )


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
    geometry = Counter()
    for view in range(len(visible)):
        geometry.update(geometry_counts(prediction["boxes"][view], target["boxes"][view],
                                        visible[view], iou_threshold))
    for key in ("oracle_center_box_hits", "oracle_size_box_hits"):
        counts[key] = geometry[key]
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
    per_sequence, per_view = {}, {}
    amplitude = [dict(samples=0, raw_feature_rms_sum=0., feature_rms_sum=0., position_rms_sum=0.)
                 for _ in range(model["num_views"])]
    best_ious = []
    association = AssociationMetrics(settings["iou_threshold"], settings["ucr_threshold"])
    for images, targets in loader:
        outputs = net([image.to(device) for image in images])
        if model.get("mvoq_position_encoding", "none") != "none":
            for view, feature in enumerate(outputs["view_features"]):
                tokens, pe = net.mvoq.prepare_view_tokens(feature)
                row = amplitude[view]
                row["samples"] += len(feature)
                row["raw_feature_rms_sum"] += float(feature.float().square().mean((1, 2, 3)).sqrt().sum())
                row["feature_rms_sum"] += float(tokens.float().square().mean((1, 2)).sqrt().sum())
                row["position_rms_sum"] += float(pe.float().square().mean().sqrt()) * len(feature)
        predictions = decode(outputs, settings["confidence_threshold"])
        classes = outputs["class_logits"].float().argmax(dim=-1).cpu()
        foreground = classes != outputs["class_logits"].shape[-1] - 1
        visible_prediction = outputs["confidences"].float().cpu() >= settings["confidence_threshold"]
        for pred, target, fg, vis in zip(predictions, targets, foreground, visible_prediction):
            hits, ious = measure(pred, target, fg, vis, settings["iou_threshold"])
            counts.update(hits)
            sequence = per_sequence.setdefault(str(target["sequence"]), Counter())
            sequence.update(hits)
            sequence["samples"] += 1
            for view in range(model["num_views"]):
                row = per_view.setdefault(str(view + 1), Counter())
                row.update(geometry_counts(pred["boxes"][view], target["boxes"][view],
                                           target["visible"][view], settings["iou_threshold"]))
            best_ious.append(ious)
            association.update(pred, target)
    ious = torch.cat(best_ious)
    metrics = association.compute()
    position_amplitude = [dict(
        view=view+1, feature_rms=row["feature_rms_sum"] / row["samples"],
        raw_feature_rms=row["raw_feature_rms_sum"] / row["samples"],
        position_rms=row["position_rms_sum"] / row["samples"],
        position_to_feature_rms=row["position_rms_sum"] / max(row["feature_rms_sum"], 1e-12),
    ) for view, row in enumerate(amplitude) if row["samples"]]
    return dict(
        samples=count, split_size=len(dataset),
        median_best_raw_iou=ious.median().item() if ious.numel() else None,
        **counts,
        MDA=metrics["MDA"], true_associations=metrics["true_associations"],
        predicted_associations=metrics["predicted_associations"],
        per_sequence={key: dict(value) for key, value in per_sequence.items()},
        per_view={key: dict(value) for key, value in per_view.items()},
        position_amplitude=position_amplitude,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", default=48, type=int, help="Maximum samples per split")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
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
        "backbone": config["model"]["backbone"],
        "mvoq_position_encoding": config["model"].get("mvoq_position_encoding", "none"),
        "diagnostic_note": "Oracle center/size uses GT for each pair; not achievable metrics or an exclusive error decomposition.",
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
