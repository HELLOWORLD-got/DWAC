"""Adapter for the unmodified published mango_eval.py calAAS implementation.
Preserves its legacy matching behavior (including ambiguous duplicate handling).
Use alongside the explicit one-to-one paper MDA, not as an identical substitute.
"""

import argparse
from collections import defaultdict
import contextlib
import importlib.util
import io
import json
from pathlib import Path


def score(predictions, ground_truth, source):
    spec = importlib.util.spec_from_file_location("mdmt_official_eval", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scores = {}
    frames_count = empty_frames = scored_frames = 0
    for first in sorted(Path(predictions).glob("*-1.json")):
        clip = first.stem.rsplit("-", 1)[0]
        second = first.with_name(clip + "-2.json")
        a, b = json.loads(first.read_text()), json.loads(second.read_text())
        if a.keys() != b.keys():
            raise ValueError(f"Unpaired prediction frames for {clip}")
        frames = {
            int(key): mod.frameObject(
                int(key),
                [mod.MangoTarget(i, int(key), 1, *row[:5]) for i, row in enumerate(a[key])],
                [mod.MangoTarget(i, int(key), 2, *row[:5]) for i, row in enumerate(b[key])],
            )
            for key in a
        }
        for view in (1, 2):
            for line in (Path(ground_truth) / f"{clip}-{view}.txt").read_text().splitlines():
                v = line.split(",")
                frame = int(v[0]) - 1
                if frame not in frames:
                    continue
                x, y, w, h = map(float, v[2:6])
                gt = mod.GTTarget(0, frame, view, int(v[1]), x, y, x + w, y + h)
                (frames[frame].gtA if view == 1 else frames[frame].gtB).append(gt)
        ordered = [frames[key] for key in sorted(frames)]
        eligible = []
        zero_scores = skipped = 0
        for frame in ordered:
            gt_shared = {t.ID for t in frame.gtA} & {t.ID for t in frame.gtB}
            if gt_shared:
                eligible.append(frame)
            elif {t.ID for t in frame.resultA} & {t.ID for t in frame.resultB}:
                # With no shared GT, every predicted association is false.
                # Also avoids the legacy evaluator indexing a missing GT pair.
                zero_scores += 1
            else:
                skipped += 1
        with contextlib.redirect_stdout(io.StringIO()):
            numerator = mod.calAAS(eligible) * len(eligible) if eligible else 0.0
        count = len(eligible) + zero_scores
        scores[clip] = numerator / count if count else None
        empty_frames += skipped
        scored_frames += count
        frames_count += len(frames)
    if not scores:
        raise ValueError("No paired prediction JSON files")
    return dict(
        MDA_legacy_official=(sum(v for v in scores.values() if v is not None) /
                             max(1, sum(v is not None for v in scores.values()))),
        per_sequence=scores,
        frames=frames_count,
        scored_frames=scored_frames,
        empty_association_frames=empty_frames,
        implementation=str(source),
        note=("Legacy matching on frames with shared GT. Frames with no shared GT and no "
              "predicted associations are excluded; predicted associations with no shared GT score zero. "
              "Other legacy duplicate/orphan handling is unchanged. Frame subset is exactly the export keys."),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", default="data/processed/MDMT/official_test")
    p.add_argument("--source", default="reference/mdmt_official/mango_eval.py")
    p.add_argument("--output", default="runs/official_mda.json")
    a = p.parse_args()
    result = score(a.predictions, a.ground_truth, a.source)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
