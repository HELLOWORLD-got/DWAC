"""Train the updated author model on synchronized MDMT samples."""

import argparse
import json
import sys
import time
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from training import distributed
from model import ModelConfig
from training.engine import build_criterion, build_model, evaluate, forward_loss, load_config, make_loader
from training.engine import freeze_training_bn, seed_all
from training.gradients import clip_gradients, gradient_report


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run(args, config, device, rank, world):
    train_settings = config["training"]
    batch = train_settings["batch_size"]
    workers = train_settings["workers"]
    local_batch = distributed.per_rank_batch_size(batch, world)
    show_bar = rank == 0 and (args.progress == "bar" or
                             (args.progress == "auto" and sys.stderr.isatty()))
    seed_all(train_settings["seed"] + rank)
    output = Path(args.output or (Path(args.resume).parent if args.resume else "runs/mdmt_author"))
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError(f"{output} already contains a run; pass --resume or choose a new --output")
    checkpoint = None
    start = 0
    best = (-1.0, -1.0)
    optimization_changes = None
    if args.resume:
        last = output / "last.pt"
        if last.exists() and Path(args.resume).resolve() != last.resolve():
            raise ValueError("Resume from this run's last.pt or choose a new --output")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved = checkpoint["config"]
        # Fill added defaults for legacy checkpoints without allowing mode changes.
        model_changed = asdict(ModelConfig(**config["model"])) != asdict(ModelConfig(**saved["model"]))
        if model_changed or any(config[key] != saved[key] for key in ("data", "evaluation")):
            raise ValueError("Resume requires the same model, data and evaluation settings")
        for key in ("batch_size", "lr", "weight_decay", "seed", "freeze_bn_stats",
                    "clip_grad", "limit_train", "limit_val"):
            if (train_settings[key] != saved["training"][key] and
                    not (args.allow_optimization_change and key in ("lr", "clip_grad"))):
                raise ValueError(f"Resume changed training setting {key}")
        if train_settings["background_weight"] != saved["training"].get("background_weight", 1.0):
            raise ValueError("Resume requires the same background weight")
        if train_settings.get("freeze_backbone_stem", False) != saved["training"].get("freeze_backbone_stem", False):
            raise ValueError("Resume requires the same backbone freezing setting")
        if train_settings.get("center_loss_weight", 0.) != saved["training"].get("center_loss_weight", 0.):
            raise ValueError("Resume requires the same center supervision weight")
        start = checkpoint["epoch"]
        best = tuple(checkpoint["best"])
        changed = {key: dict(old=saved['training'].get(key), new=train_settings.get(key))
                   for key in ('lr', 'clip_grad') if saved['training'].get(key) != train_settings.get(key)}
        optimization_changes = checkpoint.get('optimization_changes')
        if changed:
            if output.resolve() == Path(args.resume).resolve().parent:
                raise ValueError('Optimization changes require a new --output directory')
            optimization_changes = dict(source_checkpoint=str(Path(args.resume).resolve()),
                                        source_epoch=start, changes=changed,
                                        source_validation=checkpoint.get('validation'))
            best = (-1., -1.)  # best among the new optimization stage's epochs
    if start >= train_settings["epochs"]:
        if rank == 0:
            print(f"Checkpoint already completed {start} epochs.")
        return
    print(f"[rank {rank}] {device}: loading MDMT ...", flush=True)
    train = make_loader(config, "train", device, batch_size=local_batch, workers=workers,
                        limit=args.limit_train, rank=rank, world=world)
    val = (make_loader(config, "val", device, batch_size=batch, workers=workers,
                       limit=args.limit_val) if rank == 0 else None)
    print(f"[rank {rank}] building model ...", flush=True)
    raw = build_model(config, initialize_weights=checkpoint is None and rank == 0).to(device)
    if checkpoint is not None:
        raw.load_state_dict(checkpoint["model"])
    model = (DistributedDataParallel(raw, device_ids=[device.index] if device.type == "cuda" else None)
             if world > 1 else raw)
    criterion = build_criterion(raw, config)
    optimizer = torch.optim.AdamW(
        (p for p in raw.parameters() if p.requires_grad),
        lr=train_settings["lr"], weight_decay=train_settings["weight_decay"],
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.allow_optimization_change:
            for group in optimizer.param_groups:
                group['lr'] = train_settings['lr']
        distributed.restore_rng(checkpoint, device, rank, world)
        del checkpoint
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (output / "invocation.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        if optimization_changes:
            (output / 'optimization_changes.json').write_text(json.dumps(optimization_changes, indent=2)+'\n')
        print(f"{world} process(es), global batch={batch}, per-rank batch={local_batch}, "
              f"train={len(train.dataset)}, val={len(val.dataset)}, FP32", flush=True)

    for epoch in range(start + 1, train_settings["epochs"] + 1):
        if world > 1:
            train.sampler.set_epoch(epoch)
        model.train()
        freeze_training_bn(raw, config)
        began = time.monotonic()
        total_loss = 0.0
        grad_norm_max = 0.
        norm_overflow_recoveries = 0
        component_names = ("loss_cls", "loss_reg", "loss_conf", "loss_center")
        component_sums = [0.0] * len(component_names)
        samples = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with tqdm(train, desc=f"Train {epoch}/{train_settings['epochs']}", unit="batch",
                  dynamic_ncols=True, disable=not show_bar) as batches:
            for index, (images, targets) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                phase = 'forward'
                losses = {}
                failure_rng = distributed.capture_rng(device) if args.save_failure_batch else None
                try:
                    _, losses, _ = forward_loss(model, criterion, images, targets, device, world > 1)
                    loss = losses["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    phase = 'backward'
                    loss.backward()
                    # An infinite cap checks the gradient norm while preserving
                    # finite gradients when clipping was not requested.
                    cap = args.clip_grad if args.clip_grad is not None else float("inf")
                    phase = 'gradient_check'
                    grad_norm, overflow = clip_gradients(raw.named_parameters(), cap)
                    grad_norm_max = max(grad_norm_max, grad_norm)
                    norm_overflow_recoveries += int(overflow)
                    phase = 'optimizer_step'
                    optimizer.step()
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    try:
                        report = dict(rank=rank, epoch=epoch, batch=index, phase=phase, error=str(exc),
                                      samples=[dict(sequence=t['sequence'], frame=t['frame'],
                                                    instances=len(t['labels']), images=t.get('images')) for t in targets],
                                      losses={k: (float(v.detach()) if torch.isfinite(v.detach()) else None)
                                              for k, v in losses.items()},
                                      gradients=getattr(exc, 'diagnostics', None) or gradient_report(raw.named_parameters()))
                        output.mkdir(parents=True, exist_ok=True)
                        failure_path = output/f'failure_rank{rank}_epoch{epoch}_batch{index}'
                        failure_path.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                        if args.save_failure_batch:
                            save_checkpoint(failure_path.with_suffix('.pt'),
                                            dict(model={k:v.detach().cpu() for k,v in raw.state_dict().items()},
                                                 config=config, images=images, targets=targets, rng=failure_rng,
                                                 rank=rank, world_size=world, epoch=epoch, batch=index))
                        print(f'Failure diagnostics saved: {failure_path}', flush=True)
                    except Exception as diagnostic_error:
                        print(f'Could not save failure diagnostics: {diagnostic_error}', flush=True)
                    raise RuntimeError(f"rank={rank}, epoch={epoch}, batch={index}: {exc}") from exc
                value = loss.item()
                total_loss += value * len(targets)
                for component, name in enumerate(component_names):
                    if name in losses:
                        component_sums[component] += losses[name].item() * len(targets)
                samples += len(targets)
                if show_bar:
                    batches.set_postfix(loss=f"{value:.4f}", avg=f"{total_loss / samples:.4f}")
                elif rank == 0 and (index == 1 or index % 50 == 0 or index == len(train)):
                    print(f"Train {epoch} {index}/{len(train)} loss={value:.5f}", flush=True)
        totals = torch.tensor([total_loss, samples, *component_sums], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(totals)
        train_seconds = time.monotonic() - began
        train_peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        if rank == 0:
            print(f"Epoch {epoch}: validating {len(val.dataset)} samples on rank 0", flush=True)
            result = evaluate(raw, criterion, val, config, device, progress=show_bar)
            key = (result["MDA"], result["AP50"] if result["AP50"] is not None else -1.0)
            improved = key > best
            best = max(best, key)
        states = distributed.gather_rng(device, world)
        if rank == 0:
            record = dict(epoch=epoch, train_loss=float(totals[0] / totals[1]),
                          train_samples=int(totals[1]), train_seconds=train_seconds,
                          train_samples_per_second=float(totals[1]) / train_seconds,
                          rank0_train_peak_allocated_gib=train_peak,
                          elapsed_seconds=time.monotonic() - began, validation=result)
            record.update(rank0_grad_norm_max=grad_norm_max,
                          rank0_norm_overflow_recoveries=norm_overflow_recoveries)
            record["train_losses"] = {
                name: float(totals[index+2] / totals[1])
                for index, name in enumerate(("classification", "regression", "confidence", "center"))
            }
            position_logit = getattr(raw.mvoq, "position_logit", None)
            if position_logit is not None:
                record["mvoq_position_scale"] = float(position_logit.detach().sigmoid())
            state = dict(model=raw.state_dict(), optimizer=optimizer.state_dict(),
                         config=config, epoch=epoch, best=best,
                         rank_rng_states=states, world_size=world, validation=result)
            if optimization_changes:
                state['optimization_changes'] = optimization_changes
            if improved:
                save_checkpoint(output / "best.pt", state)
            save_checkpoint(output / "last.pt", state)
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"Epoch {epoch} | train_loss={record['train_loss']:.5f} "
                  f"| MDA={result['MDA']:.6g} AP50={result['AP50']} mAP={result['mAP']} "
                  f"TA={result['true_associations']} "
                  f"| {record['train_samples_per_second']:.2f} pairs/s", flush=True)
        if world > 1:
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mdmt_resnet50.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument('--allow-optimization-change', action='store_true',
                        help='Allow LR/clip changes on resume into a NEW output directory')
    parser.add_argument('--save-failure-batch', action='store_true',
                        help='Save model and input batch for backward-error replay')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Global batch across all ranks")
    parser.add_argument("--workers", type=int, help="DataLoader workers per rank")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--clip-grad", type=float, help="Optional gradient clipping norm")
    parser.add_argument("--background-weight", type=float, help="Loss weight of padded slots, in (0, 1]")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--progress", choices=("auto", "bar", "log"), default="auto")
    args = parser.parse_args()
    if args.allow_optimization_change and not args.resume:
        parser.error('--allow-optimization-change requires --resume')
    config = load_config(args.config)
    training = config["training"]
    if args.background_weight is not None:
        training["background_weight"] = args.background_weight
    if not 0 < training["background_weight"] <= 1:
        parser.error("--background-weight must be in (0, 1]")
    for key in ("epochs", "batch_size", "workers"):
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["workers"] < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")
    if any(value is not None and value <= 0 for value in (args.limit_train, args.limit_val, args.clip_grad)):
        parser.error("dataset limits and gradient norm must be positive")
    if args.random_init:
        config["model"]["pretrained"] = False
    # Training settings that affect resumed optimization must be recorded.
    if args.clip_grad is None:
        args.clip_grad = training.get("clip_grad")
    training["clip_grad"] = args.clip_grad
    if (not math.isfinite(training['lr']) or training['lr'] <= 0 or
            (args.clip_grad is not None and not math.isfinite(args.clip_grad))):
        parser.error('Learning rate and clipping cap must be finite and positive')
    training["limit_train"] = args.limit_train if args.limit_train is not None else training.get("limit_train")
    training["limit_val"] = args.limit_val if args.limit_val is not None else training.get("limit_val")
    args.limit_train, args.limit_val = training["limit_train"], training["limit_val"]
    print("Initializing device/process group ...", flush=True)
    device, rank, world = distributed.setup(args.device)
    try:
        run(args, config, device, rank, world)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
