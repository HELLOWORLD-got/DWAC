"""Check learning on one real MDMT pair before a full training run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.ops import box_iou

from training.engine import build_model, forward_loss, freeze_training_bn, load_config
from training.engine import make_loader, seed_all
from training.metrics import AssociationMetrics, DetectionMetrics, decode, xyxy


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
    common = visible.all(dim=0)
    filtered = valid & pred["present"][:, :, None]
    association = assoc.compute()
    return {
        "loss": {key: value.item() for key, value in loss.items()},
        "visible_gt_boxes": int(visible.sum()),
        "raw_box_hits": int(valid.any(dim=1).sum()),
        "kept_box_hits": int(filtered.any(dim=1).sum()),
        "shared_gt": int(common.sum()),
        "raw_same_query_hits": int(valid.all(dim=0).any(dim=0).sum()),
        "kept_same_query_hits": int(filtered.all(dim=0).any(dim=0).sum()),
        **detect.compute(),
        "MDA": association["MDA"],
        "true_associations": association["true_associations"],
        "predicted_associations": association["predicted_associations"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/mdmt_resnet50.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--index", type=int, default=334, help="Zero-based training manifest index")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--output", default="runs/author_overfit/report.json")
    args = p.parse_args()
    if args.index < 0 or min(args.steps, args.report_every) < 1:
        p.error("--index must be nonnegative, and reporting interval and steps must be positive")
    cfg = load_config(args.config)
    cfg["data"]["augment"] = False
    device = torch.device(args.device)
    settings = cfg["training"]
    seed_all(settings["seed"])
    loader = make_loader(cfg, "train", device, batch_size=1, workers=0)
    if args.index >= len(loader.dataset):
        p.error(f"--index must be below {len(loader.dataset)}")
    images, target = loader.dataset[args.index]
    images = [image.unsqueeze(0).to(device) for image in images]
    net = build_model(cfg, initialize_weights=True).to(device)
    criterion = net.build_criterion()
    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=settings["lr"], weight_decay=settings["weight_decay"],
    )
    print(f"One fixed pair, sequence={target['sequence']} frame={target['frame']}, "
          f"instances={len(target['labels'])}, device={device}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(sample=dict(sequence=target["sequence"], frame=target["frame"], index=args.index),
                  config=cfg, arguments=vars(args), steps=[])
    for step in range(args.steps + 1):
        if step % args.report_every == 0 or step == args.steps:
            observation = dict(step=step, **measure(net, criterion, images, target, device, cfg))
            report["steps"].append(observation)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(observation, allow_nan=False), flush=True)
        if step == args.steps:
            break
        net.train()
        freeze_training_bn(net, cfg)
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = forward_loss(net, criterion, images, [target], device)
        loss["loss"].backward()
        cap = settings.get("clip_grad")
        torch.nn.utils.clip_grad_norm_(net.parameters(), cap if cap is not None else float("inf"),
                                      error_if_nonfinite=True)
        optimizer.step()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
