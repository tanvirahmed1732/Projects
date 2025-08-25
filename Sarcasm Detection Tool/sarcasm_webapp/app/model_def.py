import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TypeProjector(nn.Module):
    def __init__(self, in_dim, out_dim, num_types):
        super().__init__()
        self.embs = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
    def forward(self, x, node_type):
        out = torch.zeros(x.size(0), self.embs[0].out_features, device=x.device)
        for t in range(len(self.embs)):
            mask = node_type == t
            if mask.any():
                out[mask] = self.embs[t](x[mask])
        return out

class CMGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4, num_layers=3, num_types=4, dropout=0.2):
        super().__init__()
        self.proj = TypeProjector(in_dim, hidden_dim, num_types)
        self.layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.cls_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.proj(data.x, data.node_type)
        for conv, norm in zip(self.layers, self.norms):
            x = norm(F.elu(conv(x, data.edge_index)))
        global_idx = (data.node_type == 3)  # node_type 3 = global node
        graph_repr = x[global_idx]
        return self.cls_head(graph_repr)
