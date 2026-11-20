"""Prepare shared, sequence-disjoint MDMT manifests for a 25 vs 256 query test."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mdmt_resnet50.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/MDMT/query_ablation"))
    parser.add_argument("--val-sequence", type=str, default="64")
    args = parser.parse_args()

    base = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(base["data"]["train"]).resolve()
    output = args.output.resolve()
    manifests = {split: output / f"sparse_{split}.jsonl" for split in ("train", "val")}
    if source in manifests.values():
        parser.error("Choose an output directory that does not contain the source manifest")

    records = {"train": [], "val": []}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if len(sample["ids"]) > 25:
                continue
            split = "val" if str(sample["sequence"]) == args.val_sequence else "train"
            sample["images"] = [
                os.path.relpath(source.parent / image, output) for image in sample["images"]
            ]
            records[split].append(sample)
    if not all(records.values()):
        parser.error("Both splits must contain eligible samples. Check --val-sequence and the source manifest.")

    output.mkdir(parents=True, exist_ok=True)
    for split, samples in records.items():
        with manifests[split].open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
        counts = Counter(str(sample["sequence"]) for sample in samples)
        print(f"{split}: {len(samples)} image pairs, sequence counts={dict(counts)}")

    base["data"].update({split: os.path.relpath(path) for split, path in manifests.items()})
    # Disable random crops and flips so both runs see the same input per sample.
    base["data"]["augment"] = False
    base.setdefault("training", {}).update(epochs=5, batch_size=4, background_weight=0.1)
    for queries in (25, 256):
        config = deepcopy(base)
        config["model"]["num_queries"] = queries
        path = output / f"resnet_q{queries}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(os.path.relpath(path))


if __name__ == "__main__":
    main()
