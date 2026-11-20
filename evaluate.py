"""Evaluate an author-model checkpoint on MDMT and optionally export query IDs."""

import argparse
import json
from pathlib import Path

import torch

from training.engine import build_criterion, build_model, evaluate, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--export-predictions")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive; workers must be nonnegative")
    if args.export_predictions and any(Path(args.export_predictions).glob("*.json")):
        parser.error("Choose an empty prediction directory to avoid mixing exports")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = torch.device(args.device)
    loader = make_loader(config, args.split, device, batch_size=args.batch_size,
                         workers=args.workers, limit=args.limit)
    net = build_model(config).to(device)
    net.load_state_dict(checkpoint["model"])
    result = evaluate(net, build_criterion(net, config), loader, config, device,
                      progress=True, export=args.export_predictions)
    result.update(epoch=checkpoint["epoch"], split=args.split, limited=args.limit is not None,
                  checkpoint=str(Path(args.checkpoint).resolve()), configuration=config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "configuration"}, indent=2))


if __name__ == "__main__":
    main()
