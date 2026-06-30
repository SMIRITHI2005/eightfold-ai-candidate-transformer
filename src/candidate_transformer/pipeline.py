"""End-to-end candidate transformation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .graph import EvidenceGraphBuilder
from .models import CandidateProfile, Evidence, NormalizedRecord, TransformationResult
from .resolution import resolve_profile
from .sources.base import SourceAdapter, SourceDocument
from .sources.files import auto_detect_adapter
from .extraction.pipeline import HybridExtractor


@dataclass(slots=True)
class CandidateTransformer:
    """Coordinate source loading, extraction, resolution, and graph building."""

    extractor: HybridExtractor = field(default_factory=HybridExtractor)
    graph_builder: EvidenceGraphBuilder = field(default_factory=EvidenceGraphBuilder)
    adapters: list[SourceAdapter] = field(default_factory=list)

    def load_documents(self, paths: list[Path], source_type: str | None = None) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for path in paths:
            adapter = auto_detect_adapter(path, source_type, self.adapters)
            documents.append(adapter.load(path))
        return documents

    def transform_documents(self, documents: list[SourceDocument]) -> TransformationResult:
        evidence: list[Evidence] = []
        for document in documents:
            evidence.extend(self.extractor.extract(document))

        profile = resolve_profile(evidence)
        graph = self.graph_builder.build(profile)
        stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }
        return TransformationResult(profile=profile, evidence=evidence, graph_stats=stats)

    def transform_paths(self, paths: list[Path], source_type: str | None = None) -> TransformationResult:
        documents = self.load_documents(paths, source_type=source_type)
        return self.transform_documents(documents)


def build_networkx_graph(profile: CandidateProfile) -> nx.MultiDiGraph:
    """Compatibility helper for graph generation."""

    return EvidenceGraphBuilder().build(profile)
