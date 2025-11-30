from typing import Any, Dict, Optional
import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pe_dim: int, num_layers: int,
                 attn_type: str,):
        super().__init__()

        self.node_emb = Embedding(28, embedding_dim=in_channels - pe_dim)
        self.pe_lin = Linear(in_channels, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(20, in_channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(in_channels, in_channels),
                ReLU(),
                Linear(in_channels, in_channels),
            )
            conv = GPSConv(in_channels, GINConv(nn), heads=4,
                           attn_type=attn_type)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(in_channels, in_channels // 2),
            ReLU(),
            Linear(in_channels // 2, in_channels // 4),
            ReLU(),
            Linear(in_channels // 4, out_channels),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return x