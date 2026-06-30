"""Core domain models for the candidate transformer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceSourceType(str, Enum):
    """Enumerates the supported evidence source categories."""

    ats_csv = "ats_csv"
    ats_json = "ats_json"
    resume_pdf = "resume_pdf"
    resume_docx = "resume_docx"
    resume_txt = "resume_txt"
    linkedin_export = "linkedin_export"
    linkedin_profile_web = "linkedin_profile_web"
    github_profile_json = "github_profile_json"
    github_profile_web = "github_profile_web"
    recruiter_notes_txt = "recruiter_notes_txt"
    spaCy = "spacy"
    ollama = "ollama"
    hybrid = "hybrid"


class Evidence(BaseModel):
    """Represents a single extracted datum with provenance."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: Any
    source: str
    source_type: EvidenceSourceType
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""
    page_no: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FieldDecision(BaseModel):
    """Consensus result for a canonical field."""

    model_config = ConfigDict(extra="forbid")

    selected: Any
    reason: list[str] = Field(default_factory=list)
    score: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    """Normalized work experience entry."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    summary: str | None = None
    source: str | None = None
    confidence: float | None = None
    page_no: int | None = None


class EducationEntry(BaseModel):
    """Normalized education entry."""

    model_config = ConfigDict(extra="forbid")

    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    source: str | None = None
    confidence: float | None = None
    page_no: int | None = None


class ContactPoint(BaseModel):
    """Normalized contact point."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    value: str
    source: str | None = None
    confidence: float | None = None
    page_no: int | None = None


class CandidateProfile(BaseModel):
    """Canonical internal profile used by the transformer."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str | None = None
    full_name: str | None = None
    headline: str | None = None
    summary: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    location: str | None = None
    skills: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    contacts: list[ContactPoint] = Field(default_factory=list)
    provenance: dict[str, FieldDecision] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("emails", "phones", "urls", "skills", "companies", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class TransformationResult(BaseModel):
    """Complete transformation result."""

    model_config = ConfigDict(extra="forbid")

    profile: CandidateProfile
    evidence: list[Evidence] = Field(default_factory=list)
    graph_stats: dict[str, Any] = Field(default_factory=dict)


class ProjectionFieldResult(BaseModel):
    """Projected field result with provenance and explanation."""

    model_config = ConfigDict(extra="forbid")

    selected: Any
    reason: list[str] = Field(default_factory=list)
    confidence: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ProjectionOutput(BaseModel):
    """Projection result returned by the projection engine."""

    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any]
    provenance: dict[str, ProjectionFieldResult] = Field(default_factory=dict)
    confidence: dict[str, float | None] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class NormalizedRecord(BaseModel):
    """Intermediate normalized record from a source document."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_type: EvidenceSourceType
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: Mapping[str, Any] | None = None
    page_no: int | None = None

