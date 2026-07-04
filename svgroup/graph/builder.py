"""Graph construction from primitives."""

from __future__ import annotations

import numpy as np

from svgroup.common.types import (
    EdgeRelation,
    Primitive,
    PrimitiveGraph,
    PrimitiveGraphEdge,
)


def build_graph(primitives: list[Primitive], k_neighbors: int = 5) -> PrimitiveGraph:
    """Build a k-NN graph connecting nearby primitives."""
    nodes = tuple(p.id for p in primitives)
    
    if len(primitives) < 2:
        return PrimitiveGraph(nodes=nodes, edges=())
    
    edges: list[PrimitiveGraphEdge] = []
    
    # Build k-NN edges based on centroid distance
    for i, p1 in enumerate(primitives):
        distances = []
        for j, p2 in enumerate(primitives):
            if i != j:
                dist = _euclidean_distance(p1.geometry.centroid, p2.geometry.centroid)
                distances.append((j, dist, p2))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        for j, dist, p2 in distances[:k_neighbors]:
            rel = _compute_relation(p1, p2, dist)
            edge = PrimitiveGraphEdge(src=p1.id, dst=p2.id, rel=rel)
            edges.append(edge)
    
    return PrimitiveGraph(nodes=nodes, edges=tuple(edges))


def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def _compute_relation(p1: Primitive, p2: Primitive, distance: float) -> EdgeRelation:
    """Compute edge relation features between two primitives."""
    # Calculate overlap
    overlap = _bbox_overlap(p1.geometry.bbox, p2.geometry.bbox)
    
    # Calculate containment
    containment = _bbox_containment(p1.geometry.bbox, p2.geometry.bbox)
    
    # Check same fill
    same_fill = p1.style.fill == p2.style.fill and p1.style.fill != "none"
    
    # Check adjacency (if bboxes are close)
    adjacency = distance < 10.0
    
    return EdgeRelation(
        distance=distance,
        overlap=overlap,
        containment=containment,
        same_fill=same_fill,
        adjacency=adjacency,
    )


def _bbox_overlap(bbox1: tuple[float, float, float, float], 
                  bbox2: tuple[float, float, float, float]) -> float:
    """Calculate overlap ratio between two bboxes."""
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return 0.0
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return float(min(1.0, intersection / union))


def _bbox_containment(bbox1: tuple[float, float, float, float],
                      bbox2: tuple[float, float, float, float]) -> float:
    """Calculate containment of bbox2 within bbox1."""
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Check if bbox2 is contained in bbox1
    if x0_2 >= x0_1 and y0_2 >= y0_1 and x1_2 <= x1_1 and y1_2 <= y1_1:
        return 1.0
    
    # Partial containment
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return 0.0
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    
    if area2 <= 0:
        return 0.0
    
    return float(min(1.0, intersection / area2))
