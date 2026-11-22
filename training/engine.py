"""Connect the author's model and loss to MDMT annotations and metrics."""

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from tqdm import tqdm

from model import ModelConfig, TrainConfig, build_model as build_author_model
from model import pad_targets
from .background_loss import BackgroundWeightedLoss
from .data import MultiViewDataset, collate
from .metrics import AssociationMetrics, DetectionMetrics, decode, xyxy


def load_config(path):
    data = json.loads(Path(path).read_text())
    data["model"] = asdict(ModelConfig(**data["model"]))
    defaults = {**asdict(TrainConfig()), "workers": 4, "seed": 42,
                "freeze_bn_stats": True, "background_weight": 1.0}
    data["training"] = defaults | data.get("training", {})
    return data


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_criterion(net, config):
    criterion = net.build_criterion()
    weight = config["training"].get("background_weight", 1.0)
    if weight == 1.0:
        return criterion
    return BackgroundWeightedLoss(criterion, weight)


def build_model(config, initialize_weights=False):
    model_config = ModelConfig(**config["model"])
    # Load the configured weights explicitly, avoiding a silent random-init fallback.
    net = build_author_model(replace(model_config, pretrained=False))
    backbone = net.backbone
    is_vit = hasattr(backbone, "vit")
    if initialize_weights and model_config.pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if is_vit else ResNet50_Weights.IMAGENET1K_V2
        path = Path("weights") / weights.url.rsplit("/", 1)[-1]
        state = (
            torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
            else weights.get_state_dict(progress=True, check_hash=True)
        )
        if is_vit:
            backbone.vit.load_state_dict(state)
        else:
            prefixes = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
            stem = {}
            for key, value in state.items():
                prefix, _, suffix = key.partition(".")
                if prefix in prefixes:
                    stem[f"{prefixes.index(prefix)}.{suffix}"] = value
            backbone.stem.load_state_dict(stem)
    if is_vit:
        # The author's forward uses patch features and never calls the classifier.
        backbone.vit.heads.requires_grad_(False)
    if config["training"].get("freeze_backbone_stem", False):
        if is_vit or model_config.freeze_backbone:
            raise ValueError("freeze_backbone_stem requires ResNet and freeze_backbone=false")
        # Keep the randomly initialized 2048->256 projection trainable.
        backbone.stem.requires_grad_(False)
    return net


def freeze_training_bn(net, config):
    if config["model"]["freeze_backbone"]:
        net.backbone.eval()
    elif config["training"]["freeze_bn_stats"]:
        for layer in net.backbone.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
    if config["training"].get("freeze_backbone_stem", False):
        net.backbone.stem.eval()


def make_loader(config, split, device, *, batch_size, workers, limit=None, rank=0, world=1):
    dataset = MultiViewDataset(
        config["data"][split],
        size=(config["data"]["height"], config["data"]["width"]),
        num_views=config["model"]["num_views"],
        num_classes=config["model"]["num_classes"],
        num_queries=config["model"]["num_queries"],
        augment=split == "train" and config["data"]["augment"],
        limit=limit,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, seed=config["training"]["seed"])
        if world > 1 else None
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=split == "train" and sampler is None, num_workers=workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        multiprocessing_context="spawn" if workers and (world > 1 or device.type == "cuda") else None,
    )


def forward_loss(net, criterion, images, targets, device, distributed=False):
    outputs = net([image.to(device, non_blocking=True) for image in images])
    # Pack on CPU, then move each tensor once per batch.
    padded = pad_targets(
        [t["labels"] for t in targets],
        [t["boxes"] for t in targets],
        [t["visible"] for t in targets],
        num_queries=outputs["boxes"].shape[2],
        no_object_index=outputs["class_logits"].shape[-1] - 1,
    )
    visible_count = int(padded["visibility"].sum())
    padded = {k: v.to(device) for k, v in padded.items()}
    predictions = {k: outputs[k].float() for k in ("class_logits", "boxes", "confidences")}
    finite = torch.stack([torch.isfinite(t).all() for t in predictions.values()]).all()
    if not finite:
        samples = ", ".join(f"{t['sequence']}:{t['frame']}" for t in targets)
        raise FloatingPointError(f"Non-finite predictions on MDMT samples {samples}")
    losses = criterion(predictions, padded)
    if distributed:
        local_count = predictions["boxes"].new_tensor(visible_count)
        total_count = local_count.clone()
        dist.all_reduce(total_count)
        losses["loss_reg"] = (
            losses["loss_reg"] * local_count * dist.get_world_size() / total_count.clamp_min(1)
        )
    # An all-background batch otherwise has no gradient path through the box heads.
    losses["loss"] = (
        losses["loss_cls"] + losses["loss_reg"] + losses["loss_conf"] + predictions["boxes"].sum() * 0
    )
    return predictions, losses, visible_count


@torch.no_grad()
def evaluate(net, criterion, loader, config, device, *, progress=False, export=None):
    net.eval()
    settings = config["evaluation"]
    association = AssociationMetrics(settings["iou_threshold"], settings["ucr_threshold"])
    detection = DetectionMetrics(config["model"]["num_classes"])
    exports = {}
    n = visible_total = 0
    cls = reg = conf = 0.0
    for images, targets in tqdm(loader, desc="Val", unit="batch", disable=not progress):
        outputs, losses, visible_count = forward_loss(net, criterion, images, targets, device)
        n += len(targets)
        visible_total += visible_count
        cls += losses["loss_cls"].item() * len(targets)
        reg += losses["loss_reg"].item() * visible_count
        conf += losses["loss_conf"].item() * len(targets)
        for pred, target in zip(decode(outputs, settings["confidence_threshold"]), targets):
            association.update(pred, target)
            detection.update(pred, target)
            if export is not None:
                for view, (w, h) in enumerate(target["sizes"]):
                    key = f"{target['sequence']}-{view+1}"
                    boxes = xyxy(pred["boxes"][view]) * torch.tensor([w, h, w, h])
                    exports.setdefault(key, {})[str(target["frame"])] = [
                        [q, *boxes[q].tolist()]
                        for q in pred["present"][view].nonzero().flatten().tolist()
                    ]
    if export is not None:
        path = Path(export)
        path.mkdir(parents=True, exist_ok=True)
        for key, frames in exports.items():
            frames = dict(sorted(frames.items(), key=lambda item: int(item[0])))
            (path / f"{key}.json").write_text(json.dumps(frames))
    cls, reg, conf = cls / n, reg / max(visible_total, 1), conf / n
    return {**association.compute(), **detection.compute(), "samples": n,
            "losses": dict(loss=cls + reg + conf, classification=cls, regression=reg, confidence=conf)}
