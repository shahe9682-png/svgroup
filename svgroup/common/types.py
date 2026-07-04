"""Shared typed data contracts for the SVG grouping pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PrimitiveKind = Literal[
    "path",
    "line",
    "rect",
    "circle",
    "ellipse",
    "polygon",
    "polyline",
    "text",
]

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


class Geometry(BaseModel):
    """Normalized geometry summary for one SVG primitive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bbox: BBox
    centroid: Point
    area: float = Field(ge=0)
    length: float = Field(ge=0)
    points: tuple[Point, ...] = ()
    curvature: float = Field(default=0.0, ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BBox) -> BBox:
        x0, y0, x1, y1 = bbox
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox must be ordered as x0 <= x1 and y0 <= y1")
        return bbox


class Style(BaseModel):
    """Flattened SVG style for one primitive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = Field(default=0.0, ge=0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    dash: bool = False


class Primitive(BaseModel):
    """One normalized SVG primitive in paint order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: PrimitiveKind
    geometry: Geometry
    style: Style
    paint_order: int = Field(ge=0)
    raw_attrs: dict[str, str] = Field(default_factory=dict)


class EdgeRelation(BaseModel):
    """Relationship features between two primitives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distance: float = Field(ge=0)
    overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    containment: float = Field(default=0.0, ge=0.0, le=1.0)
    same_fill: bool = False
    adjacency: bool = False


class PrimitiveGraphEdge(BaseModel):
    """Directed or undirected primitive graph edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    src: str
    dst: str
    rel: EdgeRelation


class PrimitiveGraph(BaseModel):
    """Graph contract: nodes are primitive ids and edges encode relationships."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[str, ...]
    edges: tuple[PrimitiveGraphEdge, ...] = ()

    @model_validator(mode="after")
    def validate_edge_nodes(self) -> PrimitiveGraph:
        node_set = set(self.nodes)
        for edge in self.edges:
            if edge.src not in node_set or edge.dst not in node_set:
                raise ValueError("All graph edges must reference known node ids")
        return self


class GroupNode(BaseModel):
    """Recursive hierarchy node in a grouping result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    label: str
    bbox: BBox
    members: tuple[str, ...] = ()
    children: tuple[GroupNode, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BBox) -> BBox:
        x0, y0, x1, y1 = bbox
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox must be ordered as x0 <= x1 and y0 <= y1")
        return bbox


class GroupingMetrics(BaseModel):
    """Basic structural metrics shipped with every grouping record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_primitives: int = Field(ge=0)
    n_groups: int = Field(ge=0)
    max_depth: int = Field(ge=0)


class GroupingRecord(BaseModel):
    """Output record emitted per SVG file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    svg_id: str
    hierarchy: GroupNode
    metrics: GroupingMetrics


class RunMetadata(BaseModel):
    """Minimal reproducibility metadata for experiments and batch jobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0)
    config_hash: str
    code_version: str
    data_version: str | None = None
    checkpoint: str | None = None


GroupNode.model_rebuild()
