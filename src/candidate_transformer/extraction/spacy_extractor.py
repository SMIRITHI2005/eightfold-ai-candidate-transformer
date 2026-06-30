"""Optional spaCy extractor."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..models import Evidence, EvidenceSourceType
from ..normalization import canonicalize_value


@lru_cache(maxsize=1)
def _load_spacy(model_name: str):
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load(model_name)
    except Exception:
        return None


class SpacyExtractor:
    """Extract entities using an optional spaCy model."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name

    def extract(self, text: str, source: str, source_type: EvidenceSourceType, page_no: int | None = None) -> list[Evidence]:
        nlp = _load_spacy(self.model_name)
        if nlp is None:
            return []
        doc = nlp(text)
        evidence: list[Evidence] = []
        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            field = self._map_label(ent.label_)
            if field is None:
                continue
            normalized = canonicalize_value(field, ent.text)
            if not normalized:
                continue
            marker = (field, str(normalized).lower())
            if marker in seen:
                continue
            seen.add(marker)
            evidence.append(
                Evidence(
                    field=field,
                    value=normalized,
                    source=source,
                    source_type=EvidenceSourceType.spaCy,
                    confidence=0.82,
                    raw_text=ent.text,
                    page_no=page_no,
                )
            )
        return evidence

    def _map_label(self, label: str) -> str | None:
        if label == "PERSON":
            return "full_name"
        if label in {"ORG"}:
            return "companies"
        if label in {"GPE", "LOC"}:
            return "location"
        if label in {"DATE"}:
            return "experience"
        return None
