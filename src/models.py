from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FigureRecord:
    """One raster figure extracted from one page of one source document."""

    document_id: str
    document_name: str
    page_number: int
    image_index: int
    width: int
    height: int
    image_bytes: bytes = field(repr=False)
    extension: str = "png"
    caption: str | None = None
    nearby_text: str = ""

    @property
    def source_label(self) -> str:
        return f"{self.document_name} · page {self.page_number}"


@dataclass(frozen=True)
class FigureExplanation:
    """Text-grounded figure explanation with explicit evidence categories."""

    direct_evidence: tuple[str, ...]
    reasonable_inference: tuple[str, ...]
    unknown: tuple[str, ...]
