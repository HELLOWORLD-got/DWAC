"""Check learning on a fixed batch of real MDMT pairs before a full training run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.ops import box_iou

from training.data import MultiViewDataset, collate
from training.engine import build_criterion, build_model, forward_loss, freeze_training_bn, load_config
from training.engine import seed_all
from training.metrics import AssociationMetrics, DetectionMetrics, decode, xyxy


def use_resnet_stride16(backbone):
    """Mirror TorchVision's stride replacement in layer4 while retaining its weights."""
    first, *remaining = backbone.stem[-1]
    first.conv2.stride = (1, 1)
    first.downsample[0].stride = (1, 1)
    for block in remaining:
        block.conv2.dilation = (2, 2)
        block.conv2.padding = (2, 2)
    backbone.stride = 16


class MatchingProbe(torch.nn.Module):
    """Track which query supervises each real instance in a fixed batch."""

    def __init__(self, matcher, *, fixed):
        super().__init__()
        self.matcher = matcher
        self.fixed = fixed
        self.first_assignment = None
        self.previous_queries = None
        self.changed = 0
        self.compared = 0

    def forward(self, logits, boxes, targets):
        training_step = torch.is_grad_enabled()
        if self.fixed and self.first_assignment is not None:
            return self.first_assignment

        assignment = self.matcher(logits, boxes, targets)
        if training_step:
            if self.fixed:
                self.first_assignment = assignment
            # The padded slots are interchangeable. Compare real GT only.
            queries = []
            for pairs, count in zip(assignment, targets["num_instances"].tolist()):
                rows, columns = pairs.unbind(1)
                queries.append(rows[columns.argsort()][:count])
            queries = torch.cat(queries)
            if self.previous_queries is not None:
                self.changed += int((queries != self.previous_queries).sum())
                self.compared += queries.numel()
            self.previous_queries = queries
        return assignment

    def sample_change_rate(self):
        if self.fixed:
            return 0.0 if self.first_assignment is not None else None
        rate = self.changed / self.compared if self.compared else None
        self.changed = self.compared = 0
        return rate


@torch.no_grad()
def measure(net, criterion, images, target, device, config):
    net.eval()
    outputs, loss, _ = forward_loss(net, criterion, images, [target], device)
    settings = config["evaluation"]
    pred = decode(outputs, settings["confidence_threshold"])[0]
    assoc = AssociationMetrics(settings["iou_threshold"], settings["ucr_threshold"])
    detect = DetectionMetrics(config["model"]["num_classes"])
    assoc.update(pred, target)
    detect.update(pred, target)
    overlaps = torch.stack([
        box_iou(xyxy(pred["boxes"][v]), xyxy(target["boxes"][v]))
        for v in range(len(images))
    ])
    visible = target["visible"]
    valid = (overlaps >= settings["iou_threshold"]) & visible[:, None, :]
    best_iou = overlaps.amax(dim=1)[visible].nan_to_num(0.0)
    common = visible.all(dim=0)
    filtered = valid & pred["present"][:, :, None]
    association = assoc.compute()
    class_probs = outputs["class_logits"][0].softmax(-1)
    foreground_probs = 1 - class_probs[:, -1]
    confidences = outputs["confidences"][0]
    return {
        "loss": {key: value.item() for key, value in loss.items()},
        "visible_gt_boxes": int(visible.sum()),
        "raw_box_hits": int(valid.any(dim=1).sum()),
        "median_best_raw_iou": best_iou.median().item() if best_iou.numel() else None,
        "kept_box_hits": int(filtered.any(dim=1).sum()),
        "shared_gt": int(common.sum()),
        "raw_same_query_hits": int(valid.all(dim=0).any(dim=0).sum()),
        "kept_same_query_hits": int(filtered.all(dim=0).any(dim=0).sum()),
        "foreground_queries": int((class_probs.argmax(-1) != class_probs.shape[-1] - 1).sum()),
        "mean_foreground_probability": foreground_probs.mean().item(),
        "max_foreground_probability": foreground_probs.max().item(),
        "mean_visibility_probability": confidences.mean().item(),
        "max_visibility_probability": confidences.max().item(),
        **detect.compute(),
        "MDA": association["MDA"],
        "true_associations": association["true_associations"],
        "predicted_associations": association["predicted_associations"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/mdmt_resnet50.json")
    p.add_argument("--device", default="cpu")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int, default=334, help="Zero-based training manifest index")
    selection.add_argument("--indices", nargs="+", type=int,
                           help="Train repeatedly on this fixed batch of manifest indices")
    p.add_argument("--num-queries", type=int, help="Override query count for this fixed-batch diagnostic")
    p.add_argument("--background-weight", type=float, help="Loss weight of padded slots, in (0, 1]")
    p.add_argument("--resnet-stride16", action="store_true",
                   help="Test dilated ResNet-50 layer4 at output stride 16")
    p.add_argument("--fixed-assignment", action="store_true",
                   help="Reuse the first training-step Hungarian assignments on this fixed batch")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--output")
    args = p.parse_args()
    indices = args.indices if args.indices is not None else [args.index]
    if min(indices) < 0 or min(args.steps, args.report_every) < 1:
        p.error("Sample indices must be nonnegative, and reporting interval and steps must be positive")
    if args.num_queries is not None and args.num_queries < 1:
        p.error("--num-queries must be positive")
    cfg = load_config(args.config)
    cfg["data"]["augment"] = False
    if args.num_queries is not None:
        cfg["model"]["num_queries"] = args.num_queries
    device = torch.device(args.device)
    settings = cfg["training"]
    if args.background_weight is not None:
        settings["background_weight"] = args.background_weight
    if not 0 < settings["background_weight"] <= 1:
        p.error("--background-weight must be in (0, 1]")
    seed_all(settings["seed"])
    data, model = cfg["data"], cfg["model"]
    if args.resnet_stride16 and model["backbone"] != "resnet50":
        p.error("--resnet-stride16 requires the ResNet-50 backbone")
    # Check capacity on the selected pair only. This makes the paper's N=25
    # testable without admitting a 25-query model to the full MDMT loader.
    dataset = MultiViewDataset(data["train"], size=(data["height"], data["width"]),
                               num_views=model["num_views"], num_classes=model["num_classes"])
    if max(indices) >= len(dataset):
        p.error(f"Sample indices must be below {len(dataset)}")
    images, targets = collate([dataset[index] for index in indices])
    maximum = max(len(target["labels"]) for target in targets)
    if maximum > model["num_queries"]:
        p.error(f"Sample has {maximum} instances but only {model['num_queries']} queries")
    images = [image.to(device) for image in images]
    net = build_model(cfg, initialize_weights=True)
    if args.resnet_stride16:
        use_resnet_stride16(net.backbone)
    net.to(device)
    criterion = build_criterion(net, cfg)
    author_criterion = getattr(criterion, "author", criterion)
    probe = MatchingProbe(author_criterion.matcher, fixed=args.fixed_assignment)
    author_criterion.matcher = probe
    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=settings["lr"], weight_decay=settings["weight_decay"],
    )
    samples = [
        dict(sequence=target["sequence"], frame=target["frame"], index=index)
        for index, target in zip(indices, targets)
    ]
    print(f"Fixed batch of {len(targets)} image pairs, samples={samples}, "
          f"stride={net.backbone.stride}, device={device}", flush=True)
    default_run = "author_overfit_stride16" if args.resnet_stride16 else "author_overfit"
    if args.fixed_assignment:
        default_run += "_fixed"
    if len(targets) > 1:
        default_run += "_batch"
    args.output = args.output or f"runs/{default_run}/report.json"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(config=cfg, arguments=vars(args), backbone_stride=net.backbone.stride,
                  samples=samples, steps=[])
    if len(targets) == 1:
        report["sample"] = samples[0]
    for step in range(args.steps + 1):
        if step % args.report_every == 0 or step == args.steps:
            if len(targets) == 1:
                observation = dict(step=step, **measure(net, criterion, images, targets[0], device, cfg))
            else:
                net.eval()
                # Measure the same fixed batch used for training, including when
                # the optional fixed matcher stores a batch-sized assignment.
                with torch.no_grad():
                    _, batch_loss, _ = forward_loss(net, criterion, images, targets, device)
                observations = []
                individual = build_criterion(net, cfg)
                for index, target in enumerate(targets):
                    # Geometry and metrics need no fixed correspondence. Use the
                    # normal matcher for each sample's diagnostic loss.
                    observations.append(dict(
                        **samples[index],
                        **measure(net, individual, [image[index:index+1] for image in images],
                                  target, device, cfg),
                    ))
                observation = dict(
                    step=step, per_sample=observations,
                    loss={name: value.item() for name, value in batch_loss.items()},
                )
                for name in ("visible_gt_boxes", "raw_box_hits", "kept_box_hits",
                             "shared_gt", "raw_same_query_hits", "kept_same_query_hits",
                             "true_associations", "predicted_associations"):
                    observation[name] = sum(sample[name] for sample in observations)
            observation["match_change_rate"] = probe.sample_change_rate()
            report["steps"].append(observation)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(observation, allow_nan=False), flush=True)
        if step == args.steps:
            break
        net.train()
        freeze_training_bn(net, cfg)
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = forward_loss(net, criterion, images, targets, device)
        loss["loss"].backward()
        cap = settings.get("clip_grad")
        torch.nn.utils.clip_grad_norm_(net.parameters(), cap if cap is not None else float("inf"),
                                      error_if_nonfinite=True)
        optimizer.step()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
