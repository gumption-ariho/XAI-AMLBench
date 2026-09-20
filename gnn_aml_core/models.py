"""GATv2 and RGCN node classifiers + class-weighted BCE loss.

Both models share one call signature so the trainer, the API and GNNExplainer
can treat them the same way:  model(x, edge_index, edge_attr=None, edge_type=None) -> logits [N]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, RGCNConv

from gnn_aml_core.features import EDGE_DIM, NUM_RELATIONS


class GATv2Detector(nn.Module):
    """Graph Attention Network v2 that also attends over edge features (amount, cross-border, ...)."""

    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4, num_layers: int = 2,
                 edge_dim: int = EDGE_DIM, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(d, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden * heads))
            d = hidden * heads
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).view(-1)


class RGCNDetector(nn.Module):
    """Relational GCN: one weight matrix per edge type (domestic/cross-border x regular/large)."""

    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 2,
                 num_relations: int = NUM_RELATIONS, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.num_relations = num_relations
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.convs.append(RGCNConv(d, hidden, num_relations=num_relations))
            self.norms.append(nn.LayerNorm(hidden))
            d = hidden
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = F.relu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).view(-1)


def build_model(name: str, in_dim: int, **hp) -> nn.Module:
    name = name.lower()
    if name == "gatv2":
        return GATv2Detector(in_dim, **hp)
    if name == "rgcn":
        hp.pop("heads", None)
        return RGCNDetector(in_dim, **hp)
    raise ValueError(f"unknown model '{name}' (use gatv2 or rgcn)")


def class_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """Binary cross-entropy with positives up-weighted by (#neg / #pos) to counter class imbalance."""
    targets = targets.float()
    if pos_weight is None:
        pos = targets.sum().clamp(min=1.0)
        neg = (targets.numel() - targets.sum()).clamp(min=1.0)
        pos_weight = (neg / pos).item()
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
