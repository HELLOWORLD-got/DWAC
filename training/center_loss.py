"""Matched visible-instance spatial supervision for attention references."""
import math
import torch
from torch import nn


class CenterSupervisionLoss(nn.Module):
    def __init__(self, base, weight=.25, sigma=.75):
        super().__init__()
        if not math.isfinite(weight) or weight <= 0 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Center loss weight and sigma must be finite and positive")
        self.base = base
        self.weight = weight
        self.sigma = sigma

    @property
    def author(self):
        return self.base.author if hasattr(self.base, "author") else self.base

    def forward(self, outputs, targets):
        # Reuse the SAME assignment, with no second Hungarian solve.
        losses, assignments = self.base(outputs, targets, return_assignments=True)
        attention = outputs["center_attention"]
        grid, scale = outputs["center_grid"], outputs["center_grid_scale"]
        numerator = attention.sum() * 0
        count = 0
        for b, pairs in enumerate(assignments):
            query, slot = pairs.unbind(1)
            real = slot < targets["num_instances"][b]
            for view in range(attention.shape[1]):
                valid = real & targets["visibility"][b, view, slot].bool()
                if not bool(valid.any()):
                    continue
                centers = targets["boxes"][b, view, slot[valid], :2]
                distance = (grid[None] - centers[:, None]) * scale
                # log_softmax also handles targets outside the grid-center range.
                target_map = (-.5 * distance.square().sum(-1) / self.sigma**2).softmax(-1)
                prediction = attention[b, view, query[valid]].clamp_min(1e-8)
                numerator = numerator - (target_map * prediction.log()).sum()
                count += int(valid.sum())
        losses["loss_center"] = self.weight * numerator / max(count, 1)
        losses["loss"] = losses["loss"] + losses["loss_center"]
        return losses
