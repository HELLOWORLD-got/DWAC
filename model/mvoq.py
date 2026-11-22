"""Multi-View Object Query Module (MVOQ), Sec. IV-D, Eq. (11)-(14).

The i-th instance slot owns K learnable object queries, one per view. Sharing
the slot index gives a common reported ID; whether those boxes represent the
same physical object must still be learned through joint matching and loss.

Eq. (11) is a three-way attention: the object query is the query, the instance
feature is the key, the single-view feature is the value. Standard attention
requires equal key/value token counts, whereas one instance vector and a
spatial feature map do not satisfy that requirement. Repeating that one key
over spatial positions would produce uniform weights. The paper does not
specify the extra operation needed to resolve this. This implementation uses::

    q = q_obj + W_ins * f_ins
    f_obj = CA(q, F_k, F_k)

This is an interpretation, not an algebraically equivalent implementation of
Eq. (11). Instance information only changes attention weights; the residual
below by default retains object_queries, not the instance-conditioned query.
The opt-in instance_residual flag instead retains the conditioned query.
By default,
no spatial position encoding is added. An opt-in keys_values mode adds fixed
2D sine-cosine coordinates to both keys and values. The normalized_keys_values
mode additionally normalizes visual tokens, centers and normalizes the spatial
encoding, and learns its bounded scale. These properties matter
when diagnosing weak instance-conditioned localization.
"""

from __future__ import annotations

from typing import Sequence
import math

import torch
import torch.nn as nn

from .blocks import FFN, PositionEncoding2D


class TripletAttentionBlock(nn.Module):
    """One layer of the instance-conditioned query / view-feature attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        instance_residual: bool = False,
    ) -> None:
        super().__init__()
        self.instance_residual = instance_residual
        self.instance_projection = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        object_queries: torch.Tensor,
        instance_features: torch.Tensor,
        view_tokens: torch.Tensor,
        view_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioned = object_queries + self.instance_projection(instance_features)
        memory = view_tokens if view_positions is None else view_tokens + view_positions
        attn, _ = self.attn(conditioned, memory, memory, need_weights=False)
        residual = conditioned if self.instance_residual else object_queries
        x = self.norm1(residual + self.drop1(attn))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class MultiViewObjectQuery(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_views: int = 4,
        num_queries: int = 25,
        num_layers: int = 2,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        position_encoding: str = "none",
        instance_residual: bool = False,
    ) -> None:
        super().__init__()
        if position_encoding not in ("none", "keys_values", "normalized_keys_values"):
            raise ValueError("Unknown MVOQ position encoding mode")
        self.position_encoding_mode = position_encoding
        self.position_encoding = PositionEncoding2D(dim)
        if position_encoding == "normalized_keys_values":
            self.view_norm = nn.LayerNorm(dim)
            # sigmoid(logit) starts at 0.1 and stays in (0, 1).
            self.position_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))
        else:
            self.view_norm = nn.Identity()
            self.register_parameter("position_logit", None)
        self.dim = dim
        self.num_views = num_views
        self.num_queries = num_queries

        # {q_obj,k,i}: one object query per (view, instance).
        self.object_queries = nn.Parameter(
            torch.randn(num_views, num_queries, dim) * 0.02
        )
        # Layer weights are shared across views ("CA Blocks" repeated in Fig. 3).
        self.layers = nn.ModuleList(
            [TripletAttentionBlock(dim, num_heads, ffn_ratio, dropout,
                                   instance_residual=instance_residual)
             for _ in range(num_layers)]
        )

        # Eq. (12): box head per view, Sigmoid to keep the box in [0, 1]^4.
        # Fig. 3 labels them "View 1 Reg Head" ... "View n Reg Head", and the
        # text says box prediction is performed independently for each view.
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, 4),
                )
                for _ in range(num_views)
            ]
        )
        # Eq. (13): separate confidence head, per view as well.
        self.conf_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, 1),
                )
                for _ in range(num_views)
            ]
        )

    def prepare_view_tokens(self, feature: torch.Tensor):
        """Return the actual visual tokens and positional term used by K/V.

        Normalized mode puts both terms on a defined RMS scale, removes the
        position-constant component, and learns a bounded positional weight.
        No position is inferred from a batch index or an instance slot.
        """
        tokens = self.view_norm(feature.flatten(2).transpose(1, 2))
        positions = None
        if self.position_encoding_mode != "none":
            positions = self.position_encoding(
                *feature.shape[-2:], device=tokens.device, dtype=tokens.dtype,
            ).unsqueeze(0)
            if self.position_encoding_mode == "normalized_keys_values":
                positions = positions - positions.mean(dim=1, keepdim=True)
                rms = positions.float().square().mean().sqrt().clamp_min(1e-6)
                positions = positions / rms.to(positions.dtype)
                positions = positions * self.position_logit.sigmoid().to(positions.dtype)
        return tokens, positions

    def forward(
        self,
        instance_features: torch.Tensor,
        single_view_features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``instance_features``: ``(B, N, dim)``; K view maps ``(B, dim, H', W')``.

        Returns boxes ``(B, K, N, 4)`` in ``(x, y, w, h)`` normalized form and
        confidences ``(B, K, N)`` — Eq. (14)'s per-instance payload.
        """
        if len(single_view_features) != self.num_views:
            raise ValueError(
                f"MVOQ was built for K={self.num_views} views, got {len(single_view_features)}"
            )
        batch, num_queries, dim = instance_features.shape
        if num_queries != self.num_queries:
            raise ValueError(
                f"MVOQ was built for N={self.num_queries} queries, got {num_queries}"
            )

        per_view = []
        for k in range(self.num_views):
            view_tokens, positions = self.prepare_view_tokens(single_view_features[k])
            queries = self.object_queries[k].unsqueeze(0).expand(batch, -1, -1)
            for layer in self.layers:
                queries = layer(queries, instance_features, view_tokens, positions)
            per_view.append(queries)

        object_features = torch.stack(per_view, dim=1)  # (B, K, N, dim)
        boxes = torch.stack(
            [self.reg_heads[k](object_features[:, k]).sigmoid() for k in range(self.num_views)],
            dim=1,
        )
        confidences = torch.stack(
            [
                self.conf_heads[k](object_features[:, k]).sigmoid().squeeze(-1)
                for k in range(self.num_views)
            ],
            dim=1,
        )
        return {
            "boxes": boxes,
            "confidences": confidences,
            "object_features": object_features,
        }
