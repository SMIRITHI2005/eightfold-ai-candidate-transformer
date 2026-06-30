from __future__ import annotations

import json
from pathlib import Path

from candidate_transformer.pipeline import CandidateTransformer
from candidate_transformer.projection import ProjectionConfig, ProjectionEngine


def test_transform_ats_json_and_project(tmp_path: Path) -> None:
    payload = {
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "(555) 123-4567",
        "skills": ["python", "aws", "kubernetes"],
        "experience": [
            {
                "title": "Senior Engineer",
                "company": "Acme Corp",
                "start_date": "January 2021",
                "end_date": "2023-02",
            }
        ],
        "education": [
            {
                "institution": "State University",
                "degree": "BS Computer Science",
                "end_date": "2018",
            }
        ],
    }
    input_file = tmp_path / "sample_ats.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    transformer = CandidateTransformer()
    result = transformer.transform_paths([input_file])

    assert result.profile.full_name == "John Doe"
    assert "john.doe@example.com" in result.profile.emails
    assert any(phone.startswith("+") for phone in result.profile.phones)
    assert result.graph_stats["nodes"] >= 1
    assert result.graph_stats["edges"] >= 1

    projection = ProjectionConfig.model_validate(
        {
            "fields": {
                "candidate_name": {"source": "full_name", "include_provenance": True, "include_confidence": True},
                "emails": {"source": "emails", "include_provenance": True},
            }
        }
    )
    projected = ProjectionEngine(projection).project(result.profile)

    assert projected.data["candidate_name"] == "John Doe"
    assert projected.provenance["candidate_name"].selected == "John Doe"
    assert "emails" in projected.provenance


def test_runtime_projection_config_shape(tmp_path: Path) -> None:
    payload = {
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "(555) 123-4567",
        "skills": ["python", "aws", "kubernetes"],
    }
    input_file = tmp_path / "sample_ats.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    transformer = CandidateTransformer()
    result = transformer.transform_paths([input_file])

    projection = ProjectionConfig.model_validate(
        {
            "fields": [
                {"path": "full_name", "type": "string", "required": True},
                {"path": "primary_email", "from": "emails[0]", "type": "string", "required": True},
                {"path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164"},
                {"path": "skills", "from": "skills", "type": "string[]", "normalize": "canonical"},
            ],
            "include_confidence": True,
            "include_provenance": True,
            "on_missing": "null",
        }
    )
    projected = ProjectionEngine(projection).project(result.profile)

    assert projected.data["full_name"] == "John Doe"
    assert projected.data["primary_email"] == "john.doe@example.com"
    assert projected.data["phone"] == "+15551234567"
    assert set(projected.data["skills"]) == {"Python", "AWS", "Kubernetes"}
    assert projected.confidence["full_name"] is not None
    assert projected.provenance["primary_email"].provenance["canonical_path"] == "emails"
    assert projected.output_schema["on_missing"] == "null"


def test_transform_recruiter_notes_and_ats_aliases(tmp_path: Path) -> None:
    ats_payload = {
        "candidate_name": "Jane Smith",
        "contact_email": "jane.smith@acme.example",
        "contact_phone": "+1 (212) 555-0119",
        "current_company": "Acme Corp",
        "tech_stack": ["js", "postgres", "aws"],
    }
    ats_file = tmp_path / "candidate_ats.json"
    ats_file.write_text(json.dumps(ats_payload), encoding="utf-8")

    notes_file = tmp_path / "recruiter_notes.txt"
    notes_file.write_text(
        """Name: Jane Smith
Email: jane.smith@acme.example
Phone: 212-555-0119
Company: Acme Corp
Skills: JavaScript, PostgreSQL
""",
        encoding="utf-8",
    )

    transformer = CandidateTransformer()
    result = transformer.transform_paths([ats_file, notes_file])

    assert result.profile.full_name == "Jane Smith"
    assert "jane.smith@acme.example" in result.profile.emails
    assert "+12125550119" in result.profile.phones
    assert "Acme Corp" in result.profile.companies
    assert "JavaScript" in result.profile.skills
    assert "PostgreSQL" in result.profile.skills
    assert "candidate_name" not in result.profile.provenance

    note_provenance = result.profile.provenance["emails"]
    assert note_provenance.reason
    assert note_provenance.evidence


def test_transform_csv_ats_fixture() -> None:
    transformer = CandidateTransformer()
    result = transformer.transform_paths([Path(__file__).parent / "fixtures" / "ats_sample.csv"])

    assert result.profile.full_name == "John Doe"
    assert "john.doe@example.com" in result.profile.emails
    assert "+12125550109" in result.profile.phones
    assert "Python" in result.profile.skills
    assert "AWS" in result.profile.skills
    assert "Kubernetes" in result.profile.skills
