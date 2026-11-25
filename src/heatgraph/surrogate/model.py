"""Heterogeneous GNN surrogate: hourly segment UTCI given a planting (Block E).

    slot --EdgeMLPConv (sum)--> segment     each planted slot sends a message conditioned on its
                                            shading geometry; messages are SUMMED
    segment --GraphSAGE--> segment          spatial context along the network
Residual updates, L layers, MLP head. Targets are standardised with buffers stored in the
module, so predictions come out in degC. With an engine baseline (segment.base) the network
predicts only the tree-induced change.

Design note (found in Stage 2 testing): attention-based aggregation such as GATv2 normalises
weights with a softmax over incoming edges, i.e. it averages, so a segment shaded by one tree
and one shaded by five look alike. Tree effects are close to additive in the number of
crowns, so the slot->segment relation uses sum aggregation. GATv2 remains available via
`slot_conv="gatv2"` for the ablation reported in the paper.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv, MessagePassing, SAGEConv

from .graphs import EDGE_FEATURES, SEG_HOURLY, SEG_STATIC, SLOT_FEATURES


class EdgeMLPConv(MessagePassing):
    """Bipartite sum-aggregation conv: m_ij = MLP([x_j, e_ij, x_i]); x_i' = sum_j m_ij."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x, edge_index, edge_attr):
        x_src, x_dst = x
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, size=(x_src.size(0), x_dst.size(0)))

    def message(self, x_j, x_i, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr, x_i], dim=-1))


class HeatGraphGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, heads: int = 2, dropout: float = 0.0,
                 slot_conv: str = "edgemlp"):
        super().__init__()
        seg_in, slot_in, edge_in = len(SEG_STATIC) + len(SEG_HOURLY), len(SLOT_FEATURES), len(EDGE_FEATURES)
        self.seg_enc = nn.Sequential(nn.Linear(seg_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.slot_enc = nn.Sequential(nn.Linear(slot_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        def slot_layer():
            if slot_conv == "gatv2":
                return GATv2Conv((hidden, hidden), hidden, heads=heads, concat=False, edge_dim=edge_in,
                                 add_self_loops=False)
            return EdgeMLPConv(hidden, edge_in)

        self.convs = nn.ModuleList([
            HeteroConv({("slot", "shades", "segment"): slot_layer(),
                        ("segment", "adj", "segment"): SAGEConv(hidden, hidden)}, aggr="sum")
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, base=None):
        h = {"segment": self.seg_enc(x_dict["segment"]), "slot": self.slot_enc(x_dict["slot"])}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_attr_dict)
            h = {"segment": norm(h["segment"] + torch.relu(out["segment"])), "slot": h["slot"]}
        out = self.head(h["segment"]).squeeze(-1) * self.y_std + self.y_mean
        return out if base is None else out + base

    def predict(self, data):
        """UTCI (degC) per segment for a HeteroData or Batch. If the data carry an
        unplanted-scene baseline (segment.base), the network output is the tree-induced
        change and the baseline is added back."""
        base = data["segment"].base if "base" in data["segment"] else None
        return self(data.x_dict, data.edge_index_dict, data.edge_attr_dict, base)
