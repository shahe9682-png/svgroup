"""SVG parsing to normalized primitives."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import svgelements as se
from shapely.geometry import box as shapely_box

from svgroup.common.types import BBox, Geometry, Point, Primitive, PrimitiveKind, Style


def parse_svg(svg_path: Path) -> list[Primitive]:
    """Parse an SVG file into a list of normalized primitives."""
    svg = se.SVG.parse(str(svg_path))
    primitives: list[Primitive] = []
    
    paint_order = 0
    for element in svg.elements():
        if isinstance(element, (se.Path, se.Rect, se.Circle, se.Ellipse, 
                               se.Line, se.Polygon, se.Polyline, se.Text)):
            try:
                prim = _element_to_primitive(element, paint_order)
                if prim:
                    primitives.append(prim)
                    paint_order += 1
            except Exception:
                continue
    
    return primitives


def _element_to_primitive(element: se.GraphicElement, paint_order: int) -> Primitive | None:
    """Convert an SVG element to a Primitive."""
    try:
        # Determine kind
        kind = _get_primitive_kind(element)
        if not kind:
            return None
        
        # Extract geometry
        geometry = _extract_geometry(element)
        if not geometry:
            return None
        
        # Extract style
        style = _extract_style(element)
        
        # Generate ID
        elem_id = element.id if hasattr(element, 'id') and element.id else f"prim_{paint_order}"
        
        return Primitive(
            id=elem_id,
            kind=kind,
            geometry=geometry,
            style=style,
            paint_order=paint_order,
            raw_attrs={},
        )
    except Exception:
        return None


def _get_primitive_kind(element: se.GraphicElement) -> PrimitiveKind | None:
    """Map SVG element to primitive kind."""
    if isinstance(element, se.Path):
        return "path"
    elif isinstance(element, se.Line):
        return "line"
    elif isinstance(element, se.Rect):
        return "rect"
    elif isinstance(element, se.Circle):
        return "circle"
    elif isinstance(element, se.Ellipse):
        return "ellipse"
    elif isinstance(element, se.Polygon):
        return "polygon"
    elif isinstance(element, se.Polyline):
        return "polyline"
    elif isinstance(element, se.Text):
        return "text"
    return None


def _extract_geometry(element: se.GraphicElement) -> Geometry | None:
    """Extract geometry information from an element."""
    try:
        bbox_data = element.bbox()
        if not bbox_data or len(bbox_data) < 4:
            return None
        
        x0, y0, x1, y1 = bbox_data[:4]
        
        # Ensure valid bbox
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        
        bbox: BBox = (float(x0), float(y0), float(x1), float(y1))
        
        # Calculate centroid
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        centroid: Point = (cx, cy)
        
        # Calculate area and length
        width = x1 - x0
        height = y1 - y0
        area = float(width * height)
        length = float(2 * (width + height))
        
        return Geometry(
            bbox=bbox,
            centroid=centroid,
            area=area,
            length=length,
            points=(),
            curvature=0.0,
        )
    except Exception:
        return None


def _extract_style(element: se.GraphicElement) -> Style:
    """Extract style information from an element."""
    fill = "none"
    stroke = "none"
    stroke_width = 0.0
    opacity = 1.0
    dash = False
    
    try:
        if hasattr(element, 'fill') and element.fill:
            fill = str(element.fill)
        if hasattr(element, 'stroke') and element.stroke:
            stroke = str(element.stroke)
        if hasattr(element, 'stroke_width') and element.stroke_width:
            stroke_width = float(element.stroke_width)
        if hasattr(element, 'opacity') and element.opacity is not None:
            opacity = float(element.opacity)
    except Exception:
        pass
    
    return Style(
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
        opacity=opacity,
        dash=dash,
    )
