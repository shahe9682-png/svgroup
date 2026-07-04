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
        Group primitives using intelligent adaptive algorithm.
        
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
        
        # Intelligent adaptive clustering
        groups = self._intelligent_clustering(primitives, graph, edge_probs, edge_attr, threshold)
        
        return groups
    
    def _intelligent_clustering(
        self,
        primitives: list[Primitive],
        graph,
        edge_probs: torch.Tensor,
        edge_attr: torch.Tensor,
        threshold: float,
    ) -> dict[int, list[str]]:
        """
        Intelligent adaptive clustering that works for all SVG types.
        Combines model predictions with visual/spatial features.
        """
        n_prims = len(primitives)
        
        # Step 1: Analyze SVG characteristics
        svg_stats = self._analyze_svg_characteristics(primitives, edge_probs, edge_attr)
        
        # Step 2: Determine optimal strategy based on characteristics
        strategy = self._select_grouping_strategy(svg_stats, n_prims)
        
        # Step 3: Apply hybrid clustering
        groups_dict = self._hybrid_clustering(
            primitives, graph, edge_probs, edge_attr, strategy
        )
        
        # Step 4: Post-process to ensure quality
        groups_dict = self._post_process_groups(primitives, groups_dict, strategy)
        
        # Convert to integer-keyed groups
        groups = {i: members for i, members in enumerate(groups_dict.values())}
        
        return groups
    
    def _analyze_svg_characteristics(
        self,
        primitives: list[Primitive],
        edge_probs: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> dict:
        """Analyze SVG to understand its structure."""
        n_prims = len(primitives)
        
        # Model confidence analysis
        probs_list = edge_probs.numpy().flatten()
        avg_prob = float(np.mean(probs_list))
        max_prob = float(np.max(probs_list))
        high_conf_ratio = float(np.sum(probs_list > 0.5) / len(probs_list)) if len(probs_list) > 0 else 0.0
        
        # Spatial analysis
        edge_attr_np = edge_attr.numpy()
        avg_distance = float(np.mean(edge_attr_np[:, 0])) if len(edge_attr_np) > 0 else 0.5
        overlap_ratio = float(np.sum(edge_attr_np[:, 1] > 0) / len(edge_attr_np)) if len(edge_attr_np) > 0 else 0.0
        
        # Color diversity
        unique_fills = len(set(p.style.fill for p in primitives if p.style.fill))
        color_diversity = unique_fills / n_prims if n_prims > 0 else 1.0
        
        # Size distribution
        sizes = [(p.geometry.bbox[2] - p.geometry.bbox[0]) * (p.geometry.bbox[3] - p.geometry.bbox[1]) 
                 for p in primitives]
        size_variance = float(np.var(sizes)) if sizes else 0.0
        
        return {
            'n_primitives': n_prims,
            'avg_model_confidence': avg_prob,
            'max_model_confidence': max_prob,
            'high_confidence_ratio': high_conf_ratio,
            'avg_distance': avg_distance,
            'overlap_ratio': overlap_ratio,
            'color_diversity': color_diversity,
            'size_variance': size_variance,
        }
    
    def _select_grouping_strategy(self, stats: dict, n_prims: int) -> dict:
        """Select optimal grouping strategy based on SVG characteristics."""
        strategy = {
            'use_model': True,
            'model_weight': 0.5,
            'spatial_weight': 0.3,
            'color_weight': 0.2,
            'adaptive_threshold': 0.5,
            'min_group_size': 1,
            'target_groups': max(2, n_prims // 4),  # Default: 25% of primitives
        }
        
        # High model confidence → Trust the model more
        if stats['high_confidence_ratio'] > 0.3:
            strategy['model_weight'] = 0.7
            strategy['spatial_weight'] = 0.2
            strategy['color_weight'] = 0.1
            strategy['adaptive_threshold'] = 0.5
        
        # Low model confidence → Use spatial/visual features
        elif stats['avg_model_confidence'] < 0.3:
            strategy['model_weight'] = 0.2
            strategy['spatial_weight'] = 0.5
            strategy['color_weight'] = 0.3
            strategy['adaptive_threshold'] = 0.2
        
        # High overlap → Likely layered elements
        if stats['overlap_ratio'] > 0.3:
            strategy['spatial_weight'] += 0.1
            strategy['target_groups'] = max(2, n_prims // 3)
        
        # High color diversity → Each color might be a group
        if stats['color_diversity'] > 0.5:
            strategy['color_weight'] += 0.2
            strategy['target_groups'] = min(stats['n_primitives'], 
                                           int(stats['n_primitives'] * stats['color_diversity']))
        
        # Simple SVGs (few primitives) → More grouping
        if n_prims < 10:
            strategy['target_groups'] = max(2, n_prims // 3)
            strategy['adaptive_threshold'] = 0.3
        
        # Complex SVGs (many primitives) → More conservative
        elif n_prims > 50:
            strategy['target_groups'] = max(5, n_prims // 8)
        
        return strategy
    
    def _hybrid_clustering(
        self,
        primitives: list[Primitive],
        graph,
        edge_probs: torch.Tensor,
        edge_attr: torch.Tensor,
        strategy: dict,
    ) -> dict[str, list[str]]:
        """Hybrid clustering using multiple signals."""
        
        # Calculate composite edge scores
        edge_scores = self._calculate_composite_scores(
            primitives, graph, edge_probs, edge_attr, strategy
        )
        
        # Sort by composite score
        sorted_edges = sorted(edge_scores, key=lambda x: x['score'], reverse=True)
        
        # Union-find with adaptive threshold
        parent = {p.id: p.id for p in primitives}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Add edges until we reach target number of groups
        target_groups = strategy['target_groups']
        current_components = len(primitives)
        
        for edge_info in sorted_edges:
            if current_components <= target_groups:
                break
            
            # Use adaptive threshold
            if edge_info['score'] >= strategy['adaptive_threshold']:
                if union(edge_info['src'], edge_info['dst']):
                    current_components -= 1
        
        # Group by root
        groups_dict: dict[str, list[str]] = {}
        for prim in primitives:
            root = find(prim.id)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(prim.id)
        
        return groups_dict
    
    def _calculate_composite_scores(
        self,
        primitives: list[Primitive],
        graph,
        edge_probs: torch.Tensor,
        edge_attr: torch.Tensor,
        strategy: dict,
    ) -> list[dict]:
        """Calculate composite scores combining multiple signals."""
        prim_map = {p.id: p for p in primitives}
        edge_scores = []
        
        for i, edge in enumerate(graph.edges):
            src_prim = prim_map[edge.src]
            dst_prim = prim_map[edge.dst]
            
            # Model score
            model_score = edge_probs[i].item()
            
            # Spatial score (closer = higher score)
            distance_norm = edge_attr[i][0].item()  # Already normalized
            spatial_score = 1.0 - distance_norm
            
            # Overlap/containment score
            overlap = edge_attr[i][1].item()
            containment = edge_attr[i][2].item()
            overlap_score = overlap + containment * 0.5
            
            # Color similarity score
            same_fill = edge_attr[i][3].item()
            color_score = same_fill
            
            # Composite score
            composite = (
                strategy['model_weight'] * model_score +
                strategy['spatial_weight'] * spatial_score +
                strategy['color_weight'] * color_score +
                0.1 * overlap_score  # Bonus for overlap
            )
            
            edge_scores.append({
                'src': edge.src,
                'dst': edge.dst,
                'score': composite,
                'model': model_score,
                'spatial': spatial_score,
                'color': color_score,
            })
        
        return edge_scores
    
    def _post_process_groups(
        self,
        primitives: list[Primitive],
        groups_dict: dict[str, list[str]],
        strategy: dict,
    ) -> dict[str, list[str]]:
        """Post-process groups to ensure quality."""
        
        # If still too many singleton groups, merge small nearby groups
        n_groups = len(groups_dict)
        n_prims = len(primitives)
        
        if n_groups > n_prims * 0.6:  # More than 60% are separate groups
            groups_dict = self._merge_small_groups(primitives, groups_dict)
        
        return groups_dict
    
    def _merge_small_groups(
        self,
        primitives: list[Primitive],
        groups_dict: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Merge small nearby groups intelligently."""
        prim_map = {p.id: p for p in primitives}
        
        # Calculate group properties
        group_info = []
        for root, members in groups_dict.items():
            member_prims = [prim_map[mid] for mid in members]
            
            # Calculate center and bbox
            bboxes = [p.geometry.bbox for p in member_prims]
            x_coords = [(b[0] + b[2]) / 2 for b in bboxes]
            y_coords = [(b[1] + b[3]) / 2 for b in bboxes]
            cx, cy = sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
            
            # Common properties
            fills = [p.style.fill for p in member_prims if p.style.fill]
            common_fill = fills[0] if fills and all(f == fills[0] for f in fills) else None
            
            # Avg size
            sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            avg_size = sum(sizes) / len(sizes)
            
            group_info.append({
                'root': root,
                'members': members,
                'center': (cx, cy),
                'fill': common_fill,
                'size': len(members),
                'avg_element_size': avg_size,
            })
        
        # Sort by size (smallest first)
        group_info.sort(key=lambda x: x['size'])
        
        # Merge strategy: small groups with nearby similar groups
        parent = {g['root']: g['root'] for g in group_info}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Merge small groups (size 1-2) with nearest similar group
        for i, g1 in enumerate(group_info):
            if g1['size'] > 2:  # Only merge small groups
                continue
            
            # Find best merge candidate
            best_candidate = None
            best_score = -1
            
            for j, g2 in enumerate(group_info):
                if i == j or find(g1['root']) == find(g2['root']):
                    continue
                
                # Calculate distance
                dx = g1['center'][0] - g2['center'][0]
                dy = g1['center'][1] - g2['center'][1]
                dist = (dx * dx + dy * dy) ** 0.5
                
                # Calculate similarity score
                score = 0.0
                
                # Proximity bonus (closer = higher)
                if dist < 50:
                    score += 1.0
                elif dist < 150:
                    score += 0.5
                elif dist < 300:
                    score += 0.2
                
                # Same color bonus
                if g1['fill'] and g2['fill'] and g1['fill'] == g2['fill']:
                    score += 0.8
                
                # Similar size bonus
                size_ratio = min(g1['avg_element_size'], g2['avg_element_size']) / max(g1['avg_element_size'], g2['avg_element_size'])
                if size_ratio > 0.5:
                    score += 0.3
                
                if score > best_score:
                    best_score = score
                    best_candidate = g2
            
            # Merge if good candidate found
            if best_candidate and best_score > 0.5:
                union(g1['root'], best_candidate['root'])
        
        # Rebuild groups
        merged_groups: dict[str, list[str]] = {}
        for g in group_info:
            root = find(g['root'])
            if root not in merged_groups:
                merged_groups[root] = []
            merged_groups[root].extend(g['members'])
        
        return merged_groups


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
