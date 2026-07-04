"""Inference and grouping using trained GNN model."""

from __future__ import annotations

import torch
import numpy as np
from torch_geometric.data import Data

from svgroup.common.types import GroupNode, GroupingMetrics, GroupingRecord, Primitive
from svgroup.features.extractor import extract_features
from svgroup.graph.builder import build_graph


class SVGGroupingInference:
    """Inference engine for grouping primitives."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
    
    def group_primitives(
        self,
        primitives: list[Primitive],
        threshold: float = 0.5,
    ) -> dict[int, list[str]]:
        """
        Group primitives using the trained model.
        
        Returns:
            Dictionary mapping group_id to list of primitive IDs
        """
        if not primitives:
            return {}
        
        # Extract features
        node_features = []
        for prim in primitives:
            feat = extract_features(prim)
            node_features.append(feat)
        node_features = np.array(node_features)
        
        # Build graph
        graph = build_graph(primitives, k_neighbors=5)
        
        # Create PyG data object
        x = torch.tensor(node_features, dtype=torch.float32)
        
        edge_index = []
        edge_attr = []
        for edge in graph.edges:
            src_idx = next(i for i, p in enumerate(primitives) if p.id == edge.src)
            dst_idx = next(i for i, p in enumerate(primitives) if p.id == edge.dst)
            edge_index.append([src_idx, dst_idx])
            
            # Edge features: distance, overlap, containment, same_fill, adjacency, paint_order_diff
            rel = edge.rel
            edge_attr.append([
                rel.distance / 500.0,  # normalized
                rel.overlap,
                rel.containment,
                1.0 if rel.same_fill else 0.0,
                1.0 if rel.adjacency else 0.0,
                0.0,  # paint order diff (reserved)
            ])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Run inference
        with torch.no_grad():
            edge_probs = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Union-find clustering based on edge probabilities
        groups = self._union_find_clustering(primitives, graph, edge_probs, threshold)
        
        return groups
    
    def _union_find_clustering(
        self,
        primitives: list[Primitive],
        graph,
        edge_probs: torch.Tensor,
        threshold: float,
    ) -> dict[int, list[str]]:
        """Cluster primitives using union-find on high-probability edges."""
        # Initialize union-find
        parent = {p.id: p.id for p in primitives}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union nodes connected by high-probability edges
        edges_used = 0
        for i, edge in enumerate(graph.edges):
            prob = edge_probs[i].item()
            if prob >= threshold:
                union(edge.src, edge.dst)
                edges_used += 1
        
        # Group primitives by root
        groups_dict: dict[str, list[str]] = {}
        for prim in primitives:
            root = find(prim.id)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(prim.id)
        
        # Fallback: if too many groups (each primitive isolated), try spatial clustering
        if len(groups_dict) > len(primitives) * 0.7:  # If >70% are singleton groups
            groups_dict = self._spatial_fallback_clustering(primitives, graph, edge_probs)
        
        # Convert to integer-keyed groups
        groups = {i: members for i, members in enumerate(groups_dict.values())}
        
        return groups
    
    def _spatial_fallback_clustering(
        self,
        primitives: list[Primitive],
        graph,
        edge_probs: torch.Tensor,
    ) -> dict[str, list[str]]:
        """Fallback spatial clustering when model doesn't group well."""
        # Use top 30% of edges by probability
        edge_scores = [(i, edge, edge_probs[i].item()) 
                       for i, edge in enumerate(graph.edges)]
        edge_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Take top edges
        n_edges_to_use = max(len(primitives) // 2, len(edge_scores) // 3)
        
        parent = {p.id: p.id for p in primitives}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Use top probability edges
        for i, edge, prob in edge_scores[:n_edges_to_use]:
            union(edge.src, edge.dst)
        
        # Group by root
        groups_dict: dict[str, list[str]] = {}
        for prim in primitives:
            root = find(prim.id)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(prim.id)
        
        return groups_dict


def create_grouping_record(
    svg_id: str,
    primitives: list[Primitive],
    groups: dict[int, list[str]],
) -> GroupingRecord:
    """Create a GroupingRecord from primitives and groups."""
    # Build hierarchy (flat for now)
    group_nodes = []
    for group_id, member_ids in groups.items():
        # Calculate group bbox
        member_prims = [p for p in primitives if p.id in member_ids]
        if not member_prims:
            continue
        
        bboxes = [p.geometry.bbox for p in member_prims]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        group_node = GroupNode(
            group_id=f"group_{group_id}",
            label=f"Group {group_id}",
            bbox=(x0, y0, x1, y1),
            members=tuple(member_ids),
            children=(),
            confidence=0.9,
        )
        group_nodes.append(group_node)
    
    # Create root node
    all_bboxes = [p.geometry.bbox for p in primitives]
    if all_bboxes:
        x0 = min(b[0] for b in all_bboxes)
        y0 = min(b[1] for b in all_bboxes)
        x1 = max(b[2] for b in all_bboxes)
        y1 = max(b[3] for b in all_bboxes)
    else:
        x0, y0, x1, y1 = 0, 0, 0, 0
    
    root = GroupNode(
        group_id="root",
        label="Root",
        bbox=(x0, y0, x1, y1),
        members=(),
        children=tuple(group_nodes),
        confidence=1.0,
    )
    
    metrics = GroupingMetrics(
        n_primitives=len(primitives),
        n_groups=len(groups),
        max_depth=1,
    )
    
    return GroupingRecord(
        svg_id=svg_id,
        hierarchy=root,
        metrics=metrics,
    )
