"""Fit one fixed synchronized group with the unchanged author model and loss."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import trange
from training.data import MultiViewDataset, collate
from training.engine import (build_model, build_criterion, forward_loss,
                             freeze_training_bn, load_config, seed_all)
from training.gradients import clip_gradients
from training.metrics import AssociationMetrics, decode, xyxy
from torchvision.ops import box_iou
from tools.check import verify_source
from tools.diagnose import draw_comparison, raw_diagnostics, finish_counts


def real_gt_to_query(assignment, count):
    """Ignore padded-slot permutations; return query index for each real GT."""
    result = [-1] * count
    for query, gt in assignment.detach().cpu().tolist():
        if gt < count:
            result[gt] = query
    if any(q < 0 for q in result):
        raise ValueError('Real GT missing from Hungarian assignment')
    return result


def switch_fraction(previous, current):
    return (sum(a != b for a, b in zip(previous, current)) / len(current)
            if previous is not None and current else None)


def matched_geometry(pred, target, mapping, width, height):
    """Use the loss's joint Hungarian mapping, not per-view oracle selection."""
    truth = target['boxes'].cpu()
    visible = target['visible'].cpu()
    selected = pred['boxes'][:, mapping]
    delta = (selected - truth).abs()
    scale = torch.tensor([width, height])
    center = delta[..., :2][visible] * scale
    size = delta[..., 2:][visible] * scale
    relative = delta[..., :2][visible] / truth[..., 2:][visible].clamp_min(1e-6)
    ious = torch.stack([box_iou(xyxy(selected[v]), xyxy(truth[v])).diag()
                        for v in range(len(truth))])[visible]
    return dict(matched_mean_iou=float(ious.mean()),
                matched_box_hits=int((ious >= .5).sum()),
                visible_gt_boxes=int(visible.sum()),
                mean_abs_center_error_xy_pixels=center.mean(0).tolist(),
                mean_abs_size_error_wh_pixels=size.mean(0).tolist(),
                mean_abs_center_error_xy_relative_to_gt_size=relative.mean(0).tolist())


def regression_gradient_probe(net, loss):
    """Probe L_reg alone before normal total-loss backward; no gradient updates."""
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True,
                                    allow_unused=True)
    groups = {}
    for (name, _), gradient in zip(named, gradients):
        prefix = '.'.join(name.split('.')[:2]) if name.startswith('mvoq.') else name.split('.')[0]
        row = groups.setdefault(prefix, dict(tensors=0, connected=0, nonzero=0,
                                             norm_squared=0., finite=True))
        row['tensors'] += 1
        if gradient is not None:
            row['connected'] += 1
            row['finite'] &= bool(torch.isfinite(gradient).all())
            norm = float(gradient.detach().double().norm())
            row['nonzero'] += int(norm > 0)
            row['norm_squared'] += norm * norm
    for row in groups.values():
        squared = row.pop('norm_squared')
        if not row['finite'] or not math.isfinite(squared):
            raise FloatingPointError('Nonfinite regression gradient')
        row['l2_norm'] = math.sqrt(squared)
    return groups


def run(args):
    root = Path(__file__).resolve().parents[1]
    source = verify_source(root)
    if args.steps < 1 or args.log_every < 1 or args.index < 0:
        raise ValueError('steps/log-every must be positive and index nonnegative')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing experiments are never overwritten')
    checkpoint = (torch.load(args.checkpoint, map_location='cpu', weights_only=True)
                  if args.checkpoint else None)
    config = deepcopy(checkpoint['config']) if checkpoint else load_config(args.config)
    config['data']['augment'] = False
    seed_all(config['training']['seed'])
    if args.lr is not None:
        config['training']['lr'] = args.lr
    lr = config['training']['lr']
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Learning rate must be finite and positive')
    device = torch.device(args.device)
    dataset = MultiViewDataset(config['data'][args.split],
        size=(config['data']['height'], config['data']['width']),
        num_views=config['model']['num_views'], num_classes=config['model']['num_classes'],
        num_queries=config['model']['num_queries'], class_mode=config['data'].get('class_mode'),
        augment=False)
    if args.index >= len(dataset):
        raise ValueError(f'index={args.index} outside {len(dataset)} samples')
    images, targets = collate([dataset[args.index]])
    target = targets[0]
    if not target['visible'].any():
        raise ValueError('Choose a group containing visible GT to test localization')
    # Load/cache exactly one input group once. No DataLoader, crop, flip or jitter.
    images = [image.to(device) for image in images]
    net = build_model(config, initialize_weights=checkpoint is None).to(device)
    if checkpoint:
        net.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    criterion = build_criterion(net, config)
    if config['training'].get('optimizer', 'adamw').lower() != 'adamw':
        raise ValueError('Only AdamW is supported, matching the training entry point')
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
        lr=lr, weight_decay=config['training']['weight_decay'])
    output.mkdir(parents=True, exist_ok=True)
    info = dict(config=config, author_commit=source, checkpoint=args.checkpoint,
                initialization='checkpoint weights, fresh optimizer' if args.checkpoint else 'ImageNet backbone, fresh heads and optimizer',
                seed=config['training']['seed'], fixed_batch_size=1, steps=args.steps,
                sample=dict(split=args.split, index=args.index, sequence=target['sequence'],
                            frame=target['frame'], ids=target['ids'].tolist()),
                manifest_sha256=hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
                note='All measurements are on the fitted group, not validation generalization. '
                     'Author dropout and BN policy retained; no augmentation. '
                     'Train assignment switches include dropout randomness; eval switches are between logged steps.')
    (output / 'experiment.json').write_text(json.dumps(info, indent=2) + '\n')
    capture = {}
    hook = criterion.matcher.register_forward_hook(
        lambda module, inputs, result: capture.update(assignment=result[0].detach()))
    previous_eval = previous_train = None
    best = -1.
    started = time.monotonic()
    count = len(target['labels'])

    def snapshot(step, gradients=None, training_loss=None):
        nonlocal previous_eval, best
        net.eval()
        with torch.no_grad():
            outputs, losses, _ = forward_loss(net, criterion, images, targets, device)
            mapping = real_gt_to_query(capture['assignment'], count)
            pred = decode(outputs, config['evaluation']['confidence_threshold'])[0]
            counts, values, _ = raw_diagnostics(pred, target, config['evaluation']['iou_threshold'])
            association = AssociationMetrics(config['evaluation']['iou_threshold'], config['evaluation']['ucr_threshold'])
            association.update(pred, target)
            row = dict(step=step, elapsed_seconds=time.monotonic() - started,
                       eval_losses={k: float(v) for k, v in losses.items()},
                       train_loss_before_update=training_loss,
                       **finish_counts(counts, values),
                       associations=association.compute(), gt_to_query=mapping,
                       eval_assignment_switch_fraction=switch_fraction(previous_eval, mapping),
                       regression_gradients=gradients)
            row.update(matched_geometry(pred, target, mapping, config['data']['width'], config['data']['height']))
            previous_eval = mapping
            with (output / 'history.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False) + '\n')
            draw_comparison(pred, target, dataset.path, output / f'step_{step:05d}.jpg')
            # Preserve raw arrays for later analysis; no need to extract boxes from JPGs.
            arrays = {k: v.tolist() for k, v in pred.items()}
            arrays.update(gt_boxes=target['boxes'].tolist(), gt_visible=target['visible'].tolist(),
                          gt_ids=target['ids'].tolist(), gt_to_query=mapping)
            (output / f'predictions_{step:05d}.json').write_text(json.dumps(arrays, allow_nan=False))
            score = row['matched_mean_iou']
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), config=config, epoch=0, debug_step=step,
                                experiment='single_group_fit_not_generalization'), output / 'best.pt')
            return row

    try:
        initial = snapshot(0)
        bar = trange(1, args.steps + 1, desc='Fixed-group fit', unit='step')
        with (output / 'assignments.jsonl').open('w') as assignment_file:
            for step in bar:
                net.train()
                freeze_training_bn(net, config)
                optimizer.zero_grad(set_to_none=True)
                _, losses, _ = forward_loss(net, criterion, images, targets, device)
                mapping = real_gt_to_query(capture['assignment'], count)
                assignment_file.write(json.dumps(dict(step=step, gt_to_query=mapping,
                    switch_fraction=switch_fraction(previous_train, mapping))) + '\n')
                previous_train = mapping
                log = step == 1 or step % args.log_every == 0 or step == args.steps
                gradients = regression_gradient_probe(net, losses['loss_reg']) if log else None
                losses['loss'].backward()
                cap = config['training'].get('clip_grad')
                clip_gradients(net.named_parameters(), float('inf') if cap is None else cap)
                optimizer.step()
                if log:
                    final = snapshot(step, gradients, float(losses['loss'].detach()))
                    bar.set_postfix(loss=f"{final['eval_losses']['loss']:.3f}",
                                    IoU=f"{final['matched_mean_iou']:.3f}",
                                    hits=f"{final['matched_box_hits']}/{final['visible_gt_boxes']}")
        torch.save(dict(model=net.state_dict(), config=config, epoch=0, debug_step=args.steps,
                        experiment='single_group_fit_not_generalization'), output / 'last.pt')
        summary = dict(experiment=info, initial=initial, final=final,
                       best_logged_matched_mean_iou=best)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: final[k] for k in ('step', 'matched_mean_iou', 'matched_box_hits',
              'visible_gt_boxes', 'mean_abs_center_error_xy_pixels', 'mean_abs_size_error_wh_pixels')}, indent=2))
        return summary
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--checkpoint', help='Warm-start weights only; optimizer is restarted')
    initialization.add_argument('--config', default='configs/mdmt_resnet50.json')
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
