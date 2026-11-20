"""Multi-View Object Query Module (MVOQ), Sec. IV-D, Eq. (11)-(14).

The i-th instance feature owns K object queries, one per view (Eq. 11).  Those
queries are learnable and fixed per view, so instance i is bound to view k by
construction — that binding is what makes the association conflict-free.

Eq. (11) is a three-way attention: the object query is the query, the instance
feature is the key, the single-view feature is the value.  With a single key
the softmax over keys is trivially uniform, so the literal reading degenerates
to a mean over the value tokens.  The instance feature is therefore injected
into the query instead::

    q = q_obj + W_ins * f_ins
    f_obj = CA(q, F_k, F_k)

which keeps all three roles from the text — the object query supplies the
view-specific seed, the instance feature carries global identity and drifts the
attention towards the locations of this object, and the view feature provides
the local detail used for the box.  This is the only point where the paper's
formula is under-specified; the deviation is recorded in README.md.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import FFN


class TripletAttentionBlock(nn.Module):
    """One layer of the instance-conditioned query / view-feature attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
        conditioned = object_queries + self.instance_projection(instance_features)
        attn, _ = self.attn(conditioned, view_tokens, view_tokens, need_weights=False)
        x = self.norm1(object_queries + self.drop1(attn))
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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.num_queries = num_queries

        # {q_obj,k,i}: one object query per (view, instance).
        self.object_queries = nn.Parameter(
            torch.randn(num_views, num_queries, dim) * 0.02
        )
        # Layer weights are shared across views ("CA Blocks" repeated in Fig. 3).
        self.layers = nn.ModuleList(
            [TripletAttentionBlock(dim, num_heads, ffn_ratio, dropout) for _ in range(num_layers)]
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
            view_tokens = single_view_features[k].flatten(2).transpose(1, 2)
            queries = self.object_queries[k].unsqueeze(0).expand(batch, -1, -1)
            for layer in self.layers:
                queries = layer(queries, instance_features, view_tokens)
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
