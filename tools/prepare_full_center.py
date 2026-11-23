"""Audit complete MDMT manifests and prepare a 256-query center-model run.

No annotations are truncated. Metadata from test is audited, never trained on.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path


def audit_manifest(path, queries=256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing full manifest: {path}; prepare the full MDMT data first')
    rows, keys, sequences, images = [], set(), set(), set()
    maximum = over100 = shared_empty = 0
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        sample = json.loads(line)
        n = len(sample['labels'])
        if n > queries:
            raise ValueError(f'{path}:{index+1}: {n} GT exceeds {queries} queries; nothing truncated')
        if any(label != 0 for label in sample['labels']):
            raise ValueError(f'{path}:{index+1}: expected class-agnostic labels')
        if len(sample['ids']) != n or len(set(sample['ids'])) != n:
            raise ValueError(f'{path}:{index+1}: invalid instance IDs')
        if any(len(sample[key]) != 2 for key in ('images', 'boxes', 'visible')):
            raise ValueError(f'{path}:{index+1}: expected two views')
        for view in range(2):
            if len(sample['boxes'][view]) != n or len(sample['visible'][view]) != n:
                raise ValueError(f'{path}:{index+1}: invalid per-view instance dimensions')
            for box, visible in zip(sample['boxes'][view], sample['visible'][view]):
                if len(box) != 4 or not all(math.isfinite(x) for x in box):
                    raise ValueError(f'{path}:{index+1}: invalid box')
                if visible and (not all(0 <= x <= 1 for x in box) or min(box[2:]) <= 0):
                    raise ValueError(f'{path}:{index+1}: expected positive normalized cxcywh')
            image = (path.parent / sample['images'][view]).resolve()
            if image not in images and not image.is_file():
                raise FileNotFoundError(f'Missing image: {image}')
            images.add(image)
            over100 += int(sum(sample['visible'][view]) > 100)
        if any(not (a or b) for a, b in zip(*sample['visible'])):
            raise ValueError(f'{path}:{index+1}: instance invisible in every view')
        key = (str(sample['sequence']), int(sample['frame']))
        if key in keys:
            raise ValueError(f'Duplicate sample {key} in {path}')
        keys.add(key)
        sequences.add(key[0])
        maximum = max(maximum, n)
        shared_empty += int(not any(a and b for a, b in zip(*sample['visible'])))
        rows.append(sample)
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    report = dict(manifest=str(path), samples=len(rows), maximum_instances=maximum,
                  sequences=sorted(sequences, key=int), images=len(images),
                  images_with_more_than_100_gt=over100,
                  frames_without_shared_gt=shared_empty,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return rows, report


def write_dense_subset(rows, source, destination, limit):
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(range(len(rows)), key=lambda i: (-len(rows[i]['labels']), i))[:limit]
    with destination.open('w') as stream:
        for index in selected:
            sample = deepcopy(rows[index])
            sample['images'] = [os.path.relpath((source.parent / name).resolve(), destination.parent.resolve())
                                for name in sample['images']]
            stream.write(json.dumps(sample, separators=(',', ':'))+'\n')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='configs/mdmt_resnet50_center.json')
    parser.add_argument('--data-dir', default='data/processed/MDMT')
    parser.add_argument('--output', default='configs/mdmt_resnet50_center_full.json')
    args = parser.parse_args()
    config = json.loads(Path(args.base).read_text())
    if (config['model'].get('mvoq_center_mode') != 'attention_reference' or
            config['training'].get('center_loss_weight', 0.) <= 0):
        raise ValueError('Base configuration must enable attention references and center supervision')
    config['model'].update(num_queries=256)
    # Preserve the working architecture and optimizer; expand only data/capacity.
    config['data'].update(height=416, width=720, augment=False)
    config['training'].update(epochs=15, batch_size=4, workers=4, seed=42,
                              limit_train=None, limit_val=None, clip_grad=1.0)
    root = Path(args.data_dir)
    report = {'queries':256, 'annotations_truncated':False, 'splits':{}}
    manifests = {'train':root/'train.jsonl', 'val':root/'val.jsonl', 'test':root/'test_official.jsonl'}
    records = {}
    for split, path in manifests.items():
        print(f'Auditing full {split} manifest and image paths ...', flush=True)
        records[split], report['splits'][split] = audit_manifest(path)
        config['data'][split] = str(path)
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        overlap = set(report['splits'][a]['sequences']) & set(report['splits'][b]['sequences'])
        if overlap:
            raise ValueError(f'Sequences shared by {a}/{b}: {sorted(overlap)}')
    report['sequence_disjoint'] = True
    report['sequence64_role'] = 'Training sequence; no longer an unseen-sequence evaluation'
    report['test_policy'] = 'Metadata audit only; training script does not evaluate official test'
    report['ap_protocol'] = 'Existing COCO maxDets=100; MDA/UCR use all 256 queries'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2)+'\n')
    smoke = deepcopy(config)
    smoke['training'].update(epochs=1, workers=0)
    report['dense_smoke_indices'] = {}
    for split, limit in [('train',16), ('val',8)]:
        path = root/'full_dense_smoke'/f'{split}.jsonl'
        report['dense_smoke_indices'][split] = write_dense_subset(records[split], manifests[split], path, limit)
        smoke['data'][split] = str(path)
    smoke_path = output.with_name(output.stem+'_smoke.json')
    smoke_path.write_text(json.dumps(smoke, indent=2)+'\n')
    report_path = root/'full_center_audit.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ('samples','maximum_instances','images_with_more_than_100_gt')}
                      for k,v in report['splits'].items()}, indent=2))
    print(f'Prepared: {output}, {smoke_path}, {report_path}')


if __name__ == '__main__':
    main()
