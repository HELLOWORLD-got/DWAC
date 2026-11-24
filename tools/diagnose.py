"""Evaluate fixed, evenly spaced samples without augmentation; leave author code unchanged."""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm

from training.data import MultiViewDataset, collate
from training.engine import build_model, build_criterion, forward_loss
from training.metrics import AssociationMetrics, DetectionMetrics, decode, xyxy


def sample_indices(size, limit):
    """Cover the complete manifest rather than taking its first sequence only."""
    count = min(size, limit)
    return torch.linspace(0, size - 1, count).round().long().tolist()


def raw_diagnostics(pred, target, threshold):
    """GT coverage counts are nonexclusive upper bounds, not detection recall/AP."""
    visible = target['visible'].cpu()
    truth = target['boxes'].cpu()
    views, instances = visible.shape
    queries = pred['boxes'].shape[1]
    hits = torch.zeros((views, queries, instances), dtype=torch.bool)
    kept = hits.clone()
    best_values = []
    counts = defaultdict(int)
    per_view = {}
    for v in range(views):
        iou = box_iou(xyxy(pred['boxes'][v]), xyxy(truth[v])).nan_to_num(0.)
        hits[v] = (iou >= threshold) & visible[v][None]
        kept[v] = hits[v] & pred['present'][v, :, None]
        best_values.extend(iou.max(0).values[visible[v]].tolist())
        item = dict(visible_gt_boxes=int(visible[v].sum()),
                    raw_box_hits=int(hits[v].any(0).sum()),
                    kept_box_hits=int(kept[v].any(0).sum()),
                    predicted_boxes=int(pred['present'][v].sum()))
        per_view[str(v + 1)] = item
        for key, value in item.items():
            counts[key] += value
    shared = visible.sum(0) >= 2
    # A GT can be covered by different queries in each view, or by the SAME query.
    every_view = (hits.any(1) | ~visible).all(0) & shared
    same_query = (hits | ~visible[:, None, :]).all(0).any(0) & shared
    kept_same_query = (kept | ~visible[:, None, :]).all(0).any(0) & shared
    counts.update(shared_gt=int(shared.sum()),
                  shared_gt_with_raw_hit_in_every_view=int(every_view.sum()),
                  raw_same_query_hits=int(same_query.sum()),
                  kept_same_query_hits=int(kept_same_query.sum()))
    return dict(counts), best_values, per_view


def finish_counts(counts, best):
    result = dict(counts)
    n = result.get('visible_gt_boxes', 0)
    covered = result.get('shared_gt_with_raw_hit_in_every_view', 0)
    result.update(
        raw_box_coverage=result.get('raw_box_hits', 0) / n if n else None,
        kept_box_coverage=result.get('kept_box_hits', 0) / n if n else None,
        median_best_raw_iou=float(torch.tensor(best).median()) if best else None,
        same_query_fraction_when_all_views_covered=(
            result.get('raw_same_query_hits', 0) / covered if covered else None),
    )
    return result


def draw_comparison(pred, target, manifest, output):
    """Rows are views; columns show GT, all raw queries, and retained predictions."""
    width, height = 720, 430
    canvas = Image.new('RGB', (width * 3, height * len(target['images'])), 'white')
    for v, name in enumerate(target['images']):
        with Image.open(Path(manifest).parent / name) as source:
            image = source.convert('RGB').resize((width, height - 25))
        truth = xyxy(target['boxes'][v]) * torch.tensor([width, height - 25] * 2)
        boxes = xyxy(pred['boxes'][v]) * torch.tensor([width, height - 25] * 2)
        for column, title in enumerate(('GT (green)', 'All raw queries (orange)', 'After foreground/visibility filter')):
            panel = Image.new('RGB', (width, height), 'white')
            panel.paste(image, (0, 25))
            draw = ImageDraw.Draw(panel)
            draw.text((5, 5), f'view {v+1} | {title}', fill='black')
            indices = (target['visible'][v].nonzero().flatten().tolist() if column == 0
                       else list(range(len(boxes))) if column == 1
                       else pred['present'][v].nonzero().flatten().tolist())
            for index in indices:
                corners = (truth if column == 0 else boxes)[index].tolist()
                corners[1] += 25
                corners[3] += 25
                color = 'lime' if column == 0 else 'orange'
                draw.rectangle(corners, outline=color, width=2)
                label = (f"id{int(target['ids'][index])}" if column == 0
                         else f"q{index}:{float(pred['scores'][v,index]):.2f}")
                draw.text(corners[:2], label, fill=color)
            canvas.paste(panel, (column * width, v * height))
    canvas.save(output)


@torch.no_grad()
def diagnose_split(net, criterion, config, split, device, args, output):
    dataset = MultiViewDataset(
        config['data'][split], size=(config['data']['height'], config['data']['width']),
        num_views=config['model']['num_views'], num_classes=config['model']['num_classes'],
        num_queries=config['model']['num_queries'], class_mode=config['data'].get('class_mode'),
        augment=False,
    )
    indices = sample_indices(len(dataset), args.limit)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, collate_fn=collate,
                        multiprocessing_context='spawn' if args.workers else None)
    association = AssociationMetrics(config['evaluation']['iou_threshold'],
                                     config['evaluation']['ucr_threshold'])
    detection = DetectionMetrics(config['model']['num_classes'])
    counts = defaultdict(int)
    per_sequence = defaultdict(lambda: defaultdict(int))
    per_view = defaultdict(lambda: defaultdict(int))
    best, sequence_best, samples = [], defaultdict(list), []
    cls = reg = conf = visible_total = 0
    visualize = set(sample_indices(len(indices), min(args.visualize, len(indices)))) if args.visualize else set()
    for images, targets in tqdm(loader, desc=f'Diagnose {split}', unit='batch'):
        outputs, losses, n_visible = forward_loss(net, criterion, images, targets, device)
        cls += float(losses['loss_cls']) * len(targets)
        reg += float(losses['loss_reg']) * n_visible
        conf += float(losses['loss_conf']) * len(targets)
        visible_total += n_visible
        for pred, target in zip(decode(outputs, config['evaluation']['confidence_threshold']), targets):
            position = len(samples)
            index = indices[position]
            association.update(pred, target)
            detection.update(pred, target)
            row, values, view_counts = raw_diagnostics(pred, target, config['evaluation']['iou_threshold'])
            sequence = str(target['sequence'])
            for key, value in row.items():
                counts[key] += value
                per_sequence[sequence][key] += value
            for view, item in view_counts.items():
                for key, value in item.items():
                    per_view[view][key] += value
            per_sequence[sequence]['samples'] += 1
            best.extend(values)
            sequence_best[sequence].extend(values)
            samples.append(dict(index=index, sequence=sequence, frame=target['frame'],
                                **finish_counts(row, values)))
            if position in visualize:
                draw_comparison(pred, target, dataset.path,
                                output / f'{split}_{index:05d}_seq{sequence}_frame{target["frame"]}.jpg')
    n = len(samples)
    loss_report = dict(classification=cls / n, regression=reg / max(visible_total, 1), confidence=conf / n)
    loss_report['loss'] = sum(loss_report.values())
    return dict(samples=n, split_size=len(dataset), augmentation=False,
                manifest=str(dataset.path),
                manifest_sha256=hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
                sampling='deterministic evenly spaced indices across complete manifest',
                **finish_counts(counts, best), **association.compute(), **detection.compute(),
                losses=loss_report, per_view=dict(per_view),
                per_sequence={s: finish_counts(c, sequence_best[s]) for s, c in per_sequence.items()},
                sample_records=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--limit', type=int, default=240)
    parser.add_argument('--visualize', type=int, default=4)
    parser.add_argument('--output', default='runs/car_q25_diagnosis')
    args = parser.parse_args()
    if min(args.batch_size, args.limit) < 1 or min(args.workers, args.visualize) < 0:
        parser.error('batch size/limit must be positive; workers/visualize nonnegative')
    # Fail before any output or expensive inference if a checkpoint is missing.
    for name in args.checkpoints:
        if not Path(name).is_file():
            parser.error(f'Checkpoint not found: {name}')
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error('Choose an empty output directory to preserve previous diagnostic reports')
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summaries = []
    sampling_signatures = {}
    for number, name in enumerate(args.checkpoints):
        checkpoint = torch.load(name, map_location='cpu', weights_only=True)
        config = deepcopy(checkpoint['config'])
        net = build_model(config).to(device).eval()
        net.load_state_dict(checkpoint['model'], strict=True)
        criterion = build_criterion(net, config)
        output = root / f'{number+1:02d}_{Path(name).parent.name}_{Path(name).stem}'
        output.mkdir()
        report = dict(checkpoint=str(Path(name).resolve()), epoch=checkpoint['epoch'],
                      configuration=config, note='Coverage permits multiple GT hits per query; it is not AP/recall. '
                      'Only the evaluation metrics use exclusive matching. No augmentation or model changes.',
                      model_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(Path('model').glob('*.py'))})
        for split in args.splits:
            result = diagnose_split(net, criterion, config, split, device, args, output)
            signature = (result['manifest_sha256'], [r['index'] for r in result['sample_records']],
                         config['data']['height'], config['data']['width'],
                         config['data'].get('class_mode'), config['evaluation'])
            if split in sampling_signatures and signature != sampling_signatures[split]:
                raise ValueError(f'{split}: checkpoints use different data, sizes or evaluation settings; do not compare directly')
            sampling_signatures[split] = signature
            report[split] = result
            summaries.append(dict(checkpoint=name, epoch=checkpoint['epoch'], split=split,
                                  **{k: result[k] for k in ('samples', 'AP50', 'mAP', 'MDA',
                                     'raw_box_coverage', 'kept_box_coverage', 'median_best_raw_iou',
                                     'shared_gt', 'shared_gt_with_raw_hit_in_every_view',
                                     'raw_same_query_hits', 'kept_same_query_hits')}, losses=result['losses']))
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        del net, criterion, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (root / 'comparison.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
