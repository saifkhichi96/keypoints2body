from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except Exception as exc:  # pragma: no cover - dependency may be absent
    GATConv = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GraphSkeletonData:
    """Helper class to construct graph connectivity from parent ids."""

    def __init__(self, parent_ids: list[int]):
        self.parent_ids = parent_ids
        self.edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        edges: list[list[int]] = []
        for child_idx, parent_idx in enumerate(self.parent_ids):
            if parent_idx >= 0:
                edges.append([parent_idx, child_idx])
                edges.append([child_idx, parent_idx])

        if not edges:
            num_joints = len(self.parent_ids)
            for i in range(num_joints - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class GATRotationRegressor(nn.Module):
    """Graph-attention regressor for per-joint rotation output."""

    def __init__(
        self,
        parent_ids: list[int],
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if GATConv is None:
            raise ImportError(
                "torch_geometric is required for estimator_type='ikgat'. "
                "Install keypoints2body with learned extras."
            ) from _IMPORT_ERROR

        self.parent_ids = parent_ids
        self.num_joints = len(parent_ids)
        self.graph_data = GraphSkeletonData(parent_ids)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            layer = GATConv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
            self.gat_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.joint_pos_embed = nn.Embedding(self.num_joints, hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def create_batch_graph(self, x: torch.Tensor):
        """Create batched edge index and flattened node features."""
        batch_size, num_joints, feat_dim = x.shape
        node_features = x.view(-1, feat_dim)

        edge_index = self.graph_data.edge_index.to(x.device)
        batch_edge_index = []
        for b in range(batch_size):
            batch_edge_index.append(edge_index + (b * num_joints))
        batch_edge_index = torch.cat(batch_edge_index, dim=1)

        return node_features, batch_edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_joints, _ = x.shape

        h = self.input_proj(x)
        joint_indices = torch.arange(num_joints, device=x.device)
        pos_embed = self.joint_pos_embed(joint_indices)
        h = h + pos_embed.unsqueeze(0)

        residual = self.residual_proj(x)
        node_features, edge_index = self.create_batch_graph(h)

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_prev = node_features
            node_features = gat_layer(node_features, edge_index)
            node_features = F.elu(node_features)
            node_features = layer_norm(node_features)
            if i > 0:
                node_features = node_features + h_prev

        h = node_features.view(batch_size, num_joints, -1)
        h = h + residual
        return self.output_head(h)
