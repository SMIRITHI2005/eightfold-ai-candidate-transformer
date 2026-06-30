"""Regex-based extraction helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..models import Evidence, EvidenceSourceType
from ..normalization import canonicalize_value

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
URL_RE = re.compile(r"\bhttps?://[^\s<>()\[\]{}]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{4}")


def extract_regex_evidence(text: str, source: str, source_type: EvidenceSourceType, page_no: int | None = None) -> list[Evidence]:
    """Extract email, phone, and URL evidence from raw text."""

    evidence: list[Evidence] = []
    seen: set[tuple[str, str]] = set()

    def add(field: str, match: str, confidence: float) -> None:
        normalized = canonicalize_value(field, match)
        if not normalized:
            return
        marker = (field, str(normalized).lower())
        if marker in seen:
            return
        seen.add(marker)
        evidence.append(
            Evidence(
                field=field,
                value=normalized,
                source=source,
                source_type=source_type,
                confidence=confidence,
                raw_text=match,
                page_no=page_no,
            )
        )

    for match in EMAIL_RE.findall(text):
        add("emails", match, 0.99)
    for match in URL_RE.findall(text):
        add("urls", match, 0.97)
    for match in PHONE_RE.findall(text):
        add("phones", match, 0.96)
    return evidence


def find_heading_candidate(lines: Iterable[str]) -> str | None:
    """Infer a candidate name from the leading non-empty text lines."""

    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped.split()) <= 5:
            return stripped
    return None
