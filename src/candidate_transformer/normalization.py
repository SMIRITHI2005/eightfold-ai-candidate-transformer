"""Normalization helpers for candidate data."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import phonenumbers
from dateutil import parser as date_parser

from .models import NormalizedRecord

_SKILL_SYNONYMS = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "py": "Python",
    "python": "Python",
    "node": "Node.js",
    "nodejs": "Node.js",
    "reactjs": "React",
    "react": "React",
    "spacy": "spaCy",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "aws": "AWS",
    "amazon web services": "AWS",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "git": "Git",
}

_URL_RE = re.compile(r"\bhttps?://[^\s<>()]+", re.IGNORECASE)


def normalize_text(value: Any) -> str:
    """Collapse whitespace and trim surrounding punctuation."""

    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,;.")


def normalize_phone(value: Any, default_region: str = "US") -> str | None:
    """Normalize a phone number to E.164 when possible."""

    text = normalize_text(value)
    if not text:
        return None
    try:
        parsed = phonenumbers.parse(text, default_region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_possible_number(parsed):
        return None
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def normalize_email(value: Any) -> str | None:
    """Normalize email casing and whitespace."""

    text = normalize_text(value).lower()
    if not text or "@" not in text:
        return None
    return text


def normalize_url(value: Any) -> str | None:
    """Normalize URLs and remove trailing punctuation."""

    text = normalize_text(value)
    if not text:
        return None
    match = _URL_RE.search(text)
    if match:
        return match.group(0).rstrip(".,)]}")
    return text


def normalize_date(value: Any) -> str | None:
    """Normalize dates to YYYY-MM."""

    text = normalize_text(value)
    if not text:
        return None
    try:
        parsed = date_parser.parse(text, default=datetime(1900, 1, 1))
    except (ValueError, TypeError, OverflowError):
        return None
    if isinstance(parsed, datetime):
        return parsed.strftime("%Y-%m")
    if isinstance(parsed, date):
        return parsed.strftime("%Y-%m")
    return None


def normalize_skill(value: Any) -> str | None:
    """Normalize skills to canonical names."""

    text = normalize_text(value)
    if not text:
        return None
    canonical = _SKILL_SYNONYMS.get(text.lower())
    if canonical:
        return canonical
    return text.title() if text.islower() else text


def canonicalize_value(field: str, value: Any) -> Any:
    """Normalize a value according to its canonical field semantics."""

    if isinstance(value, (dict, list)):
        return value

    field_name = field.lower()
    if field_name in {"email", "emails", "contact_email"}:
        return normalize_email(value)
    if field_name in {"phone", "phones", "contact_phone"}:
        return normalize_phone(value)
    if field_name in {"url", "urls", "website", "linkedin", "github"}:
        return normalize_url(value)
    if field_name.endswith("date") or field_name.endswith("_date"):
        return normalize_date(value)
    if field_name in {"skill", "skills"}:
        return normalize_skill(value)
    return normalize_text(value)


def canonicalize_record(record: NormalizedRecord) -> NormalizedRecord:
    """Return a normalized copy of a record's raw payload when available."""

    if not record.raw:
        return record
    normalized_raw = {
        key: canonicalize_value(key, value) if not isinstance(value, (list, dict)) else value
        for key, value in record.raw.items()
    }
    return record.model_copy(update={"raw": normalized_raw})
