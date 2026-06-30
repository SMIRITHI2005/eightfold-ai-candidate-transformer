from __future__ import annotations

from pathlib import Path

from candidate_transformer.pipeline import CandidateTransformer


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_docx_resume_fixture() -> None:
    transformer = CandidateTransformer()
    result = transformer.transform_paths([FIXTURES / "resume_sample.docx"])

    assert result.profile.full_name == "John Doe"
    assert "john.doe@example.com" in result.profile.emails
    assert "+12125550100" in result.profile.phones
    assert "Python" in result.profile.skills
    assert "AWS" in result.profile.skills
    assert "Kubernetes" in result.profile.skills
    assert result.graph_stats["nodes"] >= 1
    assert result.graph_stats["edges"] >= 1


def test_parse_pdf_resume_fixture() -> None:
    transformer = CandidateTransformer()
    result = transformer.transform_paths([FIXTURES / "resume_sample.pdf"])

    assert result.profile.full_name == "John Doe"
    assert "john.doe@example.com" in result.profile.emails
    assert "+12125550101" in result.profile.phones
    assert "Python" in result.profile.skills
    assert "AWS" in result.profile.skills
    assert "Kubernetes" in result.profile.skills
    assert result.profile.provenance["full_name"].reason
    assert result.graph_stats["nodes"] >= 1
    assert result.graph_stats["edges"] >= 1