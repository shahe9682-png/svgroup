"""Graph Neural Network for SVG grouping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    # Fallback: Simple implementation without PyG
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, concat=True):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels * heads)
            self.heads = heads
            self.concat = concat
            self.out_channels = out_channels
            
        def forward(self, x, edge_index):
            out = self.lin(x)
            return out if not self.concat else out


class SVGGroupingGNN(nn.Module):
    """Graph Attention Network for predicting edge probabilities (grouping)."""
    
    def __init__(
        self,
        node_in_channels: int = 42,
        edge_in_channels: int = 6,
        hidden_channels: int = 64,
        num_classes: int = 20,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Node encoder
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels)
        
        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels
            self.convs.append(GATConv(in_ch, hidden_channels, heads=4, concat=False))
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_in_channels]
            batch: Batch assignment [N]
        
        Returns:
            Edge probabilities [E, 1]
        """
        # Encode nodes
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Predict edge probabilities
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=1)
        edge_probs = torch.sigmoid(self.edge_predictor(edge_features))
        
        return edge_probs


class SVGGroupingTrainer:
    """Trainer for the SVG grouping GNN."""
    
    def __init__(self, model: SVGGroupingGNN, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def train_step(self, data, labels):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch if hasattr(data, 'batch') else None,
        )
        
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, labels):
        """Single evaluation step."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch if hasattr(data, 'batch') else None,
            )
            loss = self.criterion(output, labels)
        
        return loss.item()
