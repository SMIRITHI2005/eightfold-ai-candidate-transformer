"""Consensus-based conflict resolution for candidate fields."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timezone
from typing import Any

from .models import CandidateProfile, Evidence, FieldDecision
from .normalization import canonicalize_value, normalize_text

SOURCE_WEIGHTS = {
    "resume": 0.95,
    "ats": 0.90,
    "linkedin": 0.85,
    "github": 0.75,
    "notes": 0.60,
}

FIELD_SOURCE_HINTS = {
    "full_name": ["name", "full_name", "candidate_name"],
    "headline": ["headline", "title", "summary_line"],
    "summary": ["summary", "about", "profile_summary"],
    "emails": ["email", "emails", "contact_email"],
    "phones": ["phone", "phones", "contact_phone"],
    "urls": ["url", "urls", "website", "linkedin", "github"],
    "location": ["location", "city", "region"],
    "skills": ["skill", "skills", "tech_stack"],
    "companies": ["company", "companies", "employer"],
    "experience": ["experience", "employment_history"],
    "education": ["education", "academics"],
}

LIST_FIELDS = {"emails", "phones", "urls", "skills", "companies", "experience", "education", "contacts"}


def source_weight(source_name: str, source_type: str | None = None) -> float:
    """Return the configured weight for a source."""

    candidate = (source_type or source_name).lower()
    for key, value in SOURCE_WEIGHTS.items():
        if key in candidate:
            return value
    return 0.5


def _agreement_score(selected_value: Any, all_values: list[Any]) -> float:
    normalized_selected = normalize_text(selected_value).lower()
    if not all_values:
        return 0.0
    normalized_values = [normalize_text(value).lower() for value in all_values if value is not None]
    count = Counter(normalized_values)[normalized_selected]
    return count / max(len(normalized_values), 1)


def resolve_field(field: str, evidence_items: list[Evidence]) -> FieldDecision:
    """Resolve a canonical field using weighted consensus."""

    if not evidence_items:
        return FieldDecision(selected=[] if field in LIST_FIELDS else None, reason=[f"No evidence found for {field}"], score=0.0)

    if field in LIST_FIELDS:
        return _resolve_list_field(field, evidence_items)

    return _resolve_scalar_field(field, evidence_items)


def _resolve_scalar_field(field: str, evidence_items: list[Evidence]) -> FieldDecision:
    scored_items: list[tuple[float, Evidence, float, list[Evidence]]] = []
    all_values = [item.value for item in evidence_items]
    grouped_by_value: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence_items:
        grouped_by_value[normalize_text(canonicalize_value(field, item.value)).lower()].append(item)

    for item in evidence_items:
        canonical_value = canonicalize_value(field, item.value)
        agreement = _agreement_score(canonical_value, all_values)
        weight = source_weight(item.source, item.source_type.value)
        score = weight + agreement + item.confidence
        peer_items = grouped_by_value[normalize_text(canonical_value).lower()]
        scored_items.append((score, item, agreement, peer_items))

    scored_items.sort(
        key=lambda entry: (
            entry[0],
            source_weight(entry[1].source, entry[1].source_type.value),
            entry[1].confidence,
            normalize_text(entry[1].value).lower(),
            entry[1].timestamp.astimezone(timezone.utc).isoformat(),
        ),
        reverse=True,
    )
    score, selected, agreement, peers = scored_items[0]
    reasons = [
        f"Selected {selected.source_type.value} evidence from {selected.source}",
        f"Agreement score: {agreement:.2f}",
        f"Source weight: {source_weight(selected.source, selected.source_type.value):.2f}",
        f"Extraction confidence: {selected.confidence:.2f}",
    ]
    if len(peers) > 1:
        peer_sources = sorted({peer.source for peer in peers})
        reasons.insert(0, f"{', '.join(peer_sources)} agree on this value")
    return FieldDecision(
        selected=canonicalize_value(field, selected.value),
        reason=reasons,
        score=score,
        evidence=peers,
    )


def _resolve_list_field(field: str, evidence_items: list[Evidence]) -> FieldDecision:
    grouped_by_value: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence_items:
        grouped_by_value[normalize_text(canonicalize_value(field, item.value)).lower()].append(item)

    ranked_values: list[tuple[float, Any, list[Evidence], list[str]]] = []
    for normalized_value, peers in grouped_by_value.items():
        best_peer = sorted(
            peers,
            key=lambda item: (
                source_weight(item.source, item.source_type.value),
                item.confidence,
                item.timestamp.astimezone(timezone.utc).isoformat(),
            ),
            reverse=True,
        )[0]
        agreement = len(peers) / max(len(evidence_items), 1)
        score = source_weight(best_peer.source, best_peer.source_type.value) + agreement + best_peer.confidence
        ranked_values.append(
            (
                score,
                canonicalize_value(field, best_peer.value),
                peers,
                sorted({peer.source for peer in peers}),
            )
        )

    ranked_values.sort(
        key=lambda entry: (
            entry[0],
            normalize_text(entry[1]).lower(),
        ),
        reverse=True,
    )
    selected_values = [value for _, value, _, _ in ranked_values if value not in (None, "")]
    reasons: list[str] = []
    if ranked_values:
        top_score, top_value, top_peers, top_sources = ranked_values[0]
        reasons.append(f"Top value {top_value} is supported by {', '.join(top_sources)}")
        reasons.append(f"Top score: {top_score:.2f}")
        if len(top_peers) > 1:
            reasons.append(f"Agreement count: {len(top_peers)}")
    else:
        reasons.append(f"No consensus could be derived for {field}")

    return FieldDecision(
        selected=selected_values,
        reason=reasons,
        score=max((score for score, _, _, _ in ranked_values), default=0.0),
        evidence=evidence_items,
    )


def resolve_profile(evidence: list[Evidence]) -> CandidateProfile:
    """Build the canonical candidate profile from evidence."""

    field_groups: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        field_groups[item.field].append(item)

    decisions: dict[str, FieldDecision] = {}
    profile_kwargs: dict[str, Any] = {}

    for field in [
        "full_name",
        "headline",
        "summary",
        "emails",
        "phones",
        "urls",
        "location",
        "skills",
        "companies",
        "experience",
        "education",
        "contacts",
    ]:
        decision = resolve_field(field, field_groups.get(field, []))
        decisions[field] = decision
        profile_kwargs[field] = decision.selected

    profile_kwargs["provenance"] = decisions
    profile_kwargs["evidence"] = evidence
    if not profile_kwargs.get("candidate_id"):
        profile_kwargs["candidate_id"] = _derive_candidate_id(evidence)
    return CandidateProfile(**profile_kwargs)


def _derive_candidate_id(evidence: list[Evidence]) -> str:
    """Derive a stable candidate identifier from the strongest available identity evidence."""

    candidates = [item for item in evidence if item.field in {"full_name", "emails", "phones"}]
    if not candidates:
        return "candidate-unknown"
    primary = sorted(
        candidates,
        key=lambda item: (
            source_weight(item.source, item.source_type.value),
            item.confidence,
            item.timestamp.astimezone(timezone.utc).isoformat(),
        ),
        reverse=True,
    )[0]
    basis = canonicalize_value(primary.field, primary.value) or normalize_text(primary.value)
    return f"candidate-{basis.lower().replace(' ', '-')[:48]}"
