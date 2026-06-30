"""Hybrid extraction pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import AppSettings
from ..models import Evidence, EvidenceSourceType
from ..normalization import canonicalize_value
from ..sources.base import SourceDocument
from ..sources.files import flatten_json_payload, value_as_text
from .ollama_client import OllamaClient
from .regex import extract_regex_evidence, find_heading_candidate
from .spacy_extractor import SpacyExtractor


ATS_FIELD_ALIASES: dict[str, str] = {
    "full_name": "full_name",
    "candidate_name": "full_name",
    "name": "full_name",
    "person.name": "full_name",
    "contact.name": "full_name",
    "headline": "headline",
    "current_title": "headline",
    "job_title": "headline",
    "title": "headline",
    "summary": "summary",
    "profile_summary": "summary",
    "about": "summary",
    "email": "emails",
    "contact_email": "emails",
    "email_address": "emails",
    "phone": "phones",
    "mobile": "phones",
    "contact_phone": "phones",
    "website": "urls",
    "url": "urls",
    "linkedin": "urls",
    "github": "urls",
    "location": "location",
    "city": "location",
    "state": "location",
    "country": "location",
    "skill": "skills",
    "skills": "skills",
    "technology": "skills",
    "tech_stack": "skills",
    "company": "companies",
    "current_company": "companies",
    "employer": "companies",
    "organization": "companies",
    "experience": "experience",
    "employment_history": "experience",
    "work_history": "experience",
    "education": "education",
    "academic_history": "education",
    "contact": "contacts",
}

NOTE_LINE_RE = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /_.-]{1,40})\s*[:=]\s*(?P<value>.+?)\s*$")
MULTI_VALUE_SPLIT_RE = re.compile(r"\s*(?:,|\||;|/|\n)\s*")


@dataclass(slots=True)
class HybridExtractor:
    """Combine structured, regex, spaCy, and Ollama extraction paths."""

    settings: AppSettings = field(default_factory=AppSettings)
    spacy_extractor: SpacyExtractor | None = None
    ollama_client: OllamaClient | None = None

    def __post_init__(self) -> None:
        if self.spacy_extractor is None:
            self.spacy_extractor = SpacyExtractor(self.settings.spacy_model)
        if self.ollama_client is None:
            self.ollama_client = OllamaClient(self.settings.ollama_host, self.settings.ollama_model)

    def extract(self, document: SourceDocument) -> list[Evidence]:
        evidence: list[Evidence] = []
        evidence.extend(self._extract_structured(document))
        evidence.extend(self._extract_heading_candidate(document))
        evidence.extend(self._extract_text_lines(document))
        evidence.extend(self._extract_note_lines(document))
        evidence.extend(extract_regex_evidence(document.text, document.source, document.source_type, document.page_no))
        evidence.extend(self.spacy_extractor.extract(document.text, document.source, document.source_type, document.page_no) if self.spacy_extractor else [])
        evidence.extend(self._extract_semantic(document))
        return self._dedupe(evidence)

    def _extract_structured(self, document: SourceDocument) -> list[Evidence]:
        if document.raw is None:
            return []
        flattened = flatten_json_payload(document.raw)
        evidence: list[Evidence] = []
        for key, value in flattened.items():
            field = self._map_structured_field(key, value, document.source_type)
            if field is None:
                continue
            if isinstance(value, list):
                for item in value:
                    normalized = canonicalize_value(field, item)
                    if normalized in (None, ""):
                        continue
                    evidence.append(
                        Evidence(
                            field=field,
                            value=normalized,
                            source=document.source,
                            source_type=document.source_type,
                            confidence=0.95,
                            raw_text=value_as_text(item),
                            page_no=document.page_no,
                        )
                    )
                continue
            if field in {"skills", "emails", "phones", "urls", "companies"} and isinstance(value, str):
                split_values = self._split_note_value(field, value)
                if len(split_values) > 1:
                    for item in split_values:
                        normalized = canonicalize_value(field, item)
                        if normalized in (None, ""):
                            continue
                        evidence.append(
                            Evidence(
                                field=field,
                                value=normalized,
                                source=document.source,
                                source_type=document.source_type,
                                confidence=0.95,
                                raw_text=value_as_text(item),
                                page_no=document.page_no,
                            )
                        )
                    continue
            normalized = canonicalize_value(field, value)
            if normalized in (None, ""):
                continue
            evidence.append(
                Evidence(
                    field=field,
                    value=normalized,
                    source=document.source,
                    source_type=document.source_type,
                    confidence=0.95,
                    raw_text=value_as_text(value),
                    page_no=document.page_no,
                )
            )
        return evidence

    def _extract_note_lines(self, document: SourceDocument) -> list[Evidence]:
        if document.source_type != EvidenceSourceType.recruiter_notes_txt:
            return []
        return self._extract_line_based_evidence(document, confidence=0.63)

    def _extract_text_lines(self, document: SourceDocument) -> list[Evidence]:
        if document.source_type == EvidenceSourceType.recruiter_notes_txt:
            return []
        return self._extract_line_based_evidence(document, confidence=0.74)

    def _extract_line_based_evidence(self, document: SourceDocument, confidence: float) -> list[Evidence]:
        evidence: list[Evidence] = []
        for line in document.text.splitlines():
            match = NOTE_LINE_RE.match(line)
            if not match:
                continue
            label = match.group("label").strip().lower().replace(" ", "_")
            value = match.group("value").strip()
            field = ATS_FIELD_ALIASES.get(label) or self._map_structured_field(label, value, document.source_type)
            if field is None:
                continue
            if field in {"experience", "education", "contacts"}:
                continue
            values = self._split_note_value(field, value)
            for item in values:
                normalized = canonicalize_value(field, item)
                if normalized in (None, ""):
                    continue
                evidence.append(
                    Evidence(
                        field=field,
                        value=normalized,
                        source=document.source,
                        source_type=document.source_type,
                        confidence=confidence,
                        raw_text=line.strip(),
                        page_no=document.page_no,
                    )
                )
        return evidence

    def _extract_heading_candidate(self, document: SourceDocument) -> list[Evidence]:
        heading = find_heading_candidate(document.text.splitlines())
        if not heading:
            return []
        return [
            Evidence(
                field="full_name",
                value=canonicalize_value("full_name", heading),
                source=document.source,
                source_type=document.source_type,
                confidence=0.55,
                raw_text=heading,
                page_no=document.page_no,
            )
        ]

    def _extract_semantic(self, document: SourceDocument) -> list[Evidence]:
        if self.ollama_client is None:
            return []
        prompt = self._build_prompt(document.text)
        payload = self.ollama_client.generate_json(prompt)
        evidence: list[Evidence] = []
        if payload:
            for field, value in payload.items():
                mapped = self._map_semantic_field(field)
                if mapped is None:
                    continue
                if isinstance(value, list):
                    for item in value:
                        normalized = canonicalize_value(mapped, item)
                        if normalized in (None, ""):
                            continue
                        evidence.append(
                            Evidence(
                                field=mapped,
                                value=normalized,
                                source=document.source,
                                source_type=EvidenceSourceType.ollama,
                                confidence=0.72,
                                raw_text=value_as_text(item),
                                page_no=document.page_no,
                            )
                        )
                    continue
                normalized = canonicalize_value(mapped, value)
                if normalized in (None, ""):
                    continue
                evidence.append(
                    Evidence(
                        field=mapped,
                        value=normalized,
                        source=document.source,
                        source_type=EvidenceSourceType.ollama,
                        confidence=0.72,
                        raw_text=value_as_text(value),
                        page_no=document.page_no,
                    )
                )
        heading = find_heading_candidate(document.text.splitlines())
        if heading:
            evidence.append(
                Evidence(
                    field="full_name",
                    value=canonicalize_value("full_name", heading),
                    source=document.source,
                    source_type=EvidenceSourceType.ollama,
                    confidence=0.55,
                    raw_text=heading,
                    page_no=document.page_no,
                )
            )
        return evidence

    def _build_prompt(self, text: str) -> str:
        trimmed = text[:12000]
        return (
            "Extract a compact JSON object with keys full_name, headline, summary, skills, "
            "companies, experience, education, location. Return only valid JSON.\n\n"
            f"TEXT:\n{trimmed}"
        )

    def _split_note_value(self, field: str, value: str) -> list[str]:
        if field not in {"skills", "companies", "emails", "phones", "urls"}:
            return [value]
        parts = [part.strip() for part in MULTI_VALUE_SPLIT_RE.split(value) if part.strip()]
        return parts or [value]

    def _map_structured_field(self, key: str, value: Any, source_type: EvidenceSourceType) -> str | None:
        normalized_key = key.lower()
        alias = ATS_FIELD_ALIASES.get(normalized_key)
        if alias is not None:
            return alias
        if source_type == EvidenceSourceType.ats_json:
            cleaned = normalized_key.replace("__", ".").replace("_", ".")
            alias = ATS_FIELD_ALIASES.get(cleaned)
            if alias is not None:
                return alias
            if cleaned.endswith(".name"):
                return "full_name"
        if any(token in normalized_key for token in ("name", "candidate_name", "full_name")):
            return "full_name"
        if any(token in normalized_key for token in ("headline", "title", "role")) and not isinstance(value, list):
            return "headline"
        if any(token in normalized_key for token in ("summary", "about")):
            return "summary"
        if any(token in normalized_key for token in ("email",)):
            return "emails"
        if any(token in normalized_key for token in ("phone", "mobile")):
            return "phones"
        if any(token in normalized_key for token in ("url", "website", "linkedin", "github")):
            return "urls"
        if "location" in normalized_key or normalized_key.endswith("city"):
            return "location"
        if any(token in normalized_key for token in ("skill", "stack", "technology")):
            return "skills"
        if any(token in normalized_key for token in ("company", "employer", "organization")):
            return "companies"
        if any(token in normalized_key for token in ("experience", "employment", "work_history")):
            return "experience"
        if any(token in normalized_key for token in ("education", "academic")):
            return "education"
        if any(token in normalized_key for token in ("contact",)):
            return "contacts"
        return None

    def _map_semantic_field(self, key: str) -> str | None:
        semantic_map = {
            "full_name": "full_name",
            "headline": "headline",
            "summary": "summary",
            "skills": "skills",
            "companies": "companies",
            "experience": "experience",
            "education": "education",
            "location": "location",
        }
        return semantic_map.get(key.lower())

    def _dedupe(self, evidence: Iterable[Evidence]) -> list[Evidence]:
        unique: dict[tuple[str, str, str], Evidence] = {}
        for item in evidence:
            canonical_value = canonicalize_value(item.field, item.value)
            marker = (item.field, str(canonical_value).lower(), item.source)
            existing = unique.get(marker)
            if existing is None or item.confidence > existing.confidence:
                unique[marker] = item.model_copy(update={"value": canonical_value})
        return sorted(unique.values(), key=lambda item: (item.field, item.source, item.timestamp.isoformat()))
