"""Evidence graph construction using NetworkX."""

from __future__ import annotations

import json
from datetime import timezone
from typing import Any

import networkx as nx

from .models import CandidateProfile, Evidence


class EvidenceGraphBuilder:
    """Build a multi-source evidence graph from a canonical profile."""

    def build(self, profile: CandidateProfile) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        candidate_id = profile.candidate_id or "candidate-unknown"
        graph.add_node(candidate_id, kind="Candidate", label=profile.full_name or candidate_id)

        self._add_scalar_node(graph, candidate_id, "Contact", "email", profile.emails, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Contact", "phone", profile.phones, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Contact", "url", profile.urls, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Skill", "skill", profile.skills, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Company", "company", profile.companies, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Education", "education", profile.education, profile.evidence)
        self._add_scalar_node(graph, candidate_id, "Contact", "contact", profile.contacts, profile.evidence)
        return graph

    def _add_scalar_node(
        self,
        graph: nx.MultiDiGraph,
        candidate_id: str,
        node_kind: str,
        field_name: str,
        values: list[Any],
        evidence: list[Evidence],
    ) -> None:
        for index, value in enumerate(values):
            if value is None:
                continue
            node_value = self._stable_value(value)
            node_id = f"{node_kind.lower()}:{field_name}:{index}:{node_value}"
            graph.add_node(node_id, kind=node_kind, value=value)
            matched = self._matched_evidence(field_name, value, evidence)
            graph.add_edge(
                candidate_id,
                node_id,
                source=matched.source if matched else "unknown",
                confidence=matched.confidence if matched else 0.0,
                timestamp=matched.timestamp.astimezone(timezone.utc).isoformat() if matched else None,
            )

    def _matched_evidence(self, field_name: str, value: Any, evidence: list[Evidence]) -> Evidence | None:
        for item in evidence:
            if item.field == field_name and self._stable_value(item.value) == self._stable_value(value):
                return item
        return None

    def _stable_value(self, value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        return str(value).strip()
