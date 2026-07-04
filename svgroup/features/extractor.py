"""Feature extraction for primitives - 42-dimensional feature vectors."""

from __future__ import annotations

import hashlib

import numpy as np

from svgroup.common.types import Primitive


def extract_features(primitive: Primitive) -> np.ndarray:
    """Extract a 42-dimensional feature vector from a primitive."""
    features = []
    
    # Geometric features (10 dims)
    geom = primitive.geometry
    bbox = geom.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    features.extend([
        geom.centroid[0],  # cx
        geom.centroid[1],  # cy
        width,
        height,
        geom.area,
        geom.length,
        geom.curvature,
        width / (height + 1e-6),  # aspect ratio
        bbox[0],  # x0
        bbox[1],  # y0
    ])
    
    # Style features (8 dims)
    style = primitive.style
    fill_hash = _color_to_hash(style.fill)
    stroke_hash = _color_to_hash(style.stroke)
    
    features.extend([
        fill_hash,
        stroke_hash,
        style.stroke_width,
        style.opacity,
        1.0 if style.dash else 0.0,
        1.0 if style.fill != "none" else 0.0,
        1.0 if style.stroke != "none" else 0.0,
        primitive.paint_order / 1000.0,  # normalized
    ])
    
    # Primitive kind one-hot (8 dims)
    kind_map = {
        "path": 0, "line": 1, "rect": 2, "circle": 3,
        "ellipse": 4, "polygon": 5, "polyline": 6, "text": 7,
    }
    kind_onehot = [0.0] * 8
    if primitive.kind in kind_map:
        kind_onehot[kind_map[primitive.kind]] = 1.0
    features.extend(kind_onehot)
    
    # Additional geometric features (16 dims)
    features.extend([
        bbox[2],  # x1
        bbox[3],  # y1
        np.sqrt(width**2 + height**2),  # diagonal
        width * height,  # area alt
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
    ])
    
    return np.array(features, dtype=np.float32)


def _color_to_hash(color: str) -> float:
    """Convert color string to a hash value in [0, 1]."""
    if color == "none":
        return 0.0
    hash_obj = hashlib.md5(color.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return (hash_int % 1000) / 1000.0
