"""ATS JSON adapter."""

from __future__ import annotations

from pathlib import Path

from ..models import EvidenceSourceType
from .base import SourceDocument
from .files import JSONFileAdapter


class ATSJSONAdapter(JSONFileAdapter):
    """Adapter for ATS JSON exports."""

    def __init__(self) -> None:
        super().__init__(EvidenceSourceType.ats_json)

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def load(self, path: Path) -> SourceDocument:
        return super().load(path)
