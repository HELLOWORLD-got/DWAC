"""Optional padding weights for dense MDMT runs with a large query budget."""

import torch
from torch import nn
from torch.nn import functional as F

from model.box_ops import box_loss


class BackgroundWeightedLoss(nn.Module):
    """Use the author matcher, with weighted means for classification and visibility."""

    def __init__(self, author_loss, background_weight):
        super().__init__()
        if not 0 < background_weight <= 1:
            raise ValueError("background weight must be in (0, 1]")
        self.author = author_loss
        self.background_weight = background_weight

    def forward(self, outputs, targets, *, return_assignments=False):
        cls, boxes, conf = (outputs[key] for key in ("class_logits", "boxes", "confidences"))
        assignment = torch.stack(self.author.matcher(cls, boxes, targets))
        matched = torch.empty_like(assignment[..., 1]).scatter_(
            1, assignment[..., 0], assignment[..., 1]
        )
        real = matched < targets["num_instances"][:, None]
        weights = torch.where(real, 1.0, self.background_weight).to(cls)

        labels = targets["labels"].gather(1, matched)
        nll = -cls.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        per_scene_weight = weights.sum(dim=1)
        loss_cls = ((nll * weights).sum(dim=1) / per_scene_weight).mean()

        view_slots = matched[:, None, :].expand(-1, boxes.shape[1], -1)
        visible = targets["visibility"].gather(2, view_slots).bool() & real[:, None, :]
        matched_boxes = targets["boxes"].gather(2, view_slots[..., None].expand(-1, -1, -1, 4))
        costs = box_loss(boxes, matched_boxes, self.author.weight_l1, self.author.weight_giou)
        loss_reg = (costs * visible).sum() / visible.sum().clamp_min(1)

        bce = F.binary_cross_entropy(conf.clamp(1e-6, 1 - 1e-6), visible.to(conf), reduction="none")
        # Only padded slots are downweighted. A real instance absent from a
        # particular view still contributes a full-weight negative there.
        loss_conf = ((bce * weights[:, None, :]).sum(dim=(1, 2)) /
                     (per_scene_weight * boxes.shape[1])).mean()

        loss_cls = loss_cls * self.author.weight_cls
        loss_reg = loss_reg * self.author.weight_reg
        loss_conf = loss_conf * self.author.weight_conf
        losses = dict(loss=loss_cls + loss_reg + loss_conf, loss_cls=loss_cls,
                      loss_reg=loss_reg, loss_conf=loss_conf)
        return (losses, assignment) if return_assignments else losses
