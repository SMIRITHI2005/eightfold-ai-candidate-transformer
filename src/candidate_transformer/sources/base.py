"""Source adapter base classes and document model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import EvidenceSourceType


@dataclass(slots=True)
class SourceDocument:
    """A loaded source document and its metadata."""

    path: Path | None
    source: str
    source_type: EvidenceSourceType
    text: str
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    page_no: int | None = None


class SourceAdapter(ABC):
    """Base class for all source adapters."""

    source_type: EvidenceSourceType

    @abstractmethod
    def can_handle(self, source: Path | str) -> bool:
        """Return True when the adapter can process the given source reference."""

    @abstractmethod
    def load(self, source: Path | str) -> SourceDocument:
        """Load a source document from a local path or remote URL."""
