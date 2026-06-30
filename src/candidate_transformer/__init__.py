"""Candidate transformer package."""

from .graph import EvidenceGraphBuilder
from .models import (
    CandidateProfile,
    ContactPoint,
    EducationEntry,
    Evidence,
    EvidenceSourceType,
    ExperienceEntry,
    FieldDecision,
    NormalizedRecord,
    ProjectionFieldResult,
    ProjectionOutput,
    TransformationResult,
)

__all__ = [
    "CandidateProfile",
    "ContactPoint",
    "EducationEntry",
    "Evidence",
    "EvidenceGraphBuilder",
    "EvidenceSourceType",
    "ExperienceEntry",
    "FieldDecision",
    "NormalizedRecord",
    "ProjectionFieldResult",
    "ProjectionOutput",
    "TransformationResult",
]

__version__ = "0.1.0"
