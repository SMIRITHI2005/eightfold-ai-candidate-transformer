"""Schema-driven projection engine for runtime output reshaping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import CandidateProfile, ProjectionFieldResult, ProjectionOutput
from .normalization import normalize_date, normalize_phone, normalize_skill, normalize_text


class MissingValuePolicy(str, Enum):
    """How the projection layer handles missing values."""

    null = "null"
    omit = "omit"
    error = "error"


class ProjectionFieldSpec(BaseModel):
    """Projection rule for one output field."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: str
    from_: str | None = Field(default=None, alias="from")
    type: str = "any"
    required: bool = False
    normalize: str | None = None
    include_confidence: bool | None = None
    include_provenance: bool | None = None
    on_missing: MissingValuePolicy | None = None

    @property
    def source_path(self) -> str:
        return self.from_ or self.path


class ProjectionConfig(BaseModel):
    """Projection configuration loaded from YAML or JSON."""

    model_config = ConfigDict(extra="forbid")

    fields: list[ProjectionFieldSpec] = Field(default_factory=list)
    include_confidence: bool = False
    include_provenance: bool = False
    on_missing: MissingValuePolicy = MissingValuePolicy.null

    @model_validator(mode="before")
    @classmethod
    def _convert_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        fields = value.get("fields")
        if isinstance(fields, dict):
            converted_fields: list[dict[str, Any]] = []
            for path, rule in fields.items():
                rule_payload = dict(rule or {})
                rule_payload.setdefault("path", path)
                if "source" in rule_payload and "from" not in rule_payload:
                    rule_payload["from"] = rule_payload.pop("source")
                converted_fields.append(rule_payload)
            value = {**value, "fields": converted_fields}
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectionConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectionConfig":
        return cls.from_file(path)

    @classmethod
    def from_text(cls, text: str) -> "ProjectionConfig":
        raw = yaml.safe_load(text) if text.strip() else {}
        return cls.model_validate(raw or {})


@dataclass(slots=True)
class ProjectionEngine:
    """Project canonical profiles to a configurable output schema."""

    config: ProjectionConfig
    field_results: dict[str, ProjectionFieldResult] = field(default_factory=dict)

    def project(self, profile: CandidateProfile) -> ProjectionOutput:
        canonical = profile.model_dump(mode="json")
        data: dict[str, Any] = {}
        provenance_map: dict[str, ProjectionFieldResult] = {}
        confidence_map: dict[str, float | None] = {}
        missing_fields: list[str] = []

        for spec in self.config.fields:
            source_path = spec.source_path
            output_path = spec.path
            include_confidence = self._effective_flag(spec.include_confidence, self.config.include_confidence)
            include_provenance = self._effective_flag(spec.include_provenance, self.config.include_provenance)
            missing_policy = spec.on_missing or self.config.on_missing

            raw_value = self._extract_path(canonical, source_path)
            is_missing = self._is_missing(raw_value)
            if is_missing:
                missing_fields.append(output_path)
                if spec.required or missing_policy == MissingValuePolicy.error:
                    raise ValueError(f"Missing required field '{output_path}' from '{source_path}'")
                if missing_policy == MissingValuePolicy.omit:
                    continue
                projected_value: Any = None
            else:
                projected_value = self._normalize_value(raw_value, spec.normalize, spec.type, source_path)

            projected_value = self._validate_and_coerce_type(projected_value, spec.type, output_path)
            self._set_output_value(data, output_path, projected_value)

            provenance = self._build_provenance(profile, source_path, projected_value)
            if include_provenance:
                provenance_map[output_path] = provenance
            if include_confidence:
                confidence_map[output_path] = provenance.confidence
            self.field_results[output_path] = provenance

        self.field_results = provenance_map
        schema_payload = {
            "fields": [spec.model_dump(by_alias=True) for spec in self.config.fields],
            "include_confidence": self.config.include_confidence,
            "include_provenance": self.config.include_provenance,
            "on_missing": self.config.on_missing.value,
        }
        return ProjectionOutput(
            data=data,
            provenance=provenance_map,
            confidence=confidence_map,
            missing_fields=missing_fields,
            output_schema=schema_payload,
        )

    def _effective_flag(self, field_value: bool | None, default_value: bool) -> bool:
        return default_value if field_value is None else field_value

    def _build_provenance(self, profile: CandidateProfile, source_path: str, value: Any) -> ProjectionFieldResult:
        root_field = source_path.split(".", 1)[0].split("[", 1)[0]
        provenance = profile.provenance.get(root_field)
        return ProjectionFieldResult(
            selected=value,
            reason=list(provenance.reason) if provenance else [f"Projected from {source_path}"],
            confidence=provenance.score if provenance else None,
            provenance={
                "source": provenance.evidence[0].source if provenance and provenance.evidence else None,
                "source_type": provenance.evidence[0].source_type.value if provenance and provenance.evidence else None,
                "evidence_count": len(provenance.evidence) if provenance else 0,
                "canonical_path": root_field,
            },
        )

    def _extract_path(self, value: Any, path: str) -> Any:
        normalized_path = path.replace(" ", "")
        if not normalized_path:
            return value
        tokens = self._tokenize_path(normalized_path)
        current_values: list[Any] = [value]
        for token in tokens:
            next_values: list[Any] = []
            for current in current_values:
                next_values.extend(self._apply_token(current, token))
            current_values = next_values
        if not current_values:
            return None
        if len(current_values) == 1:
            return current_values[0]
        return current_values

    def _tokenize_path(self, path: str) -> list[str]:
        tokens: list[str] = []
        for part in path.split("."):
            if not part:
                continue
            remainder = part
            while remainder:
                if remainder.startswith("["):
                    closing_index = remainder.find("]")
                    if closing_index == -1:
                        tokens.append(remainder)
                        break
                    tokens.append(remainder[: closing_index + 1])
                    remainder = remainder[closing_index + 1 :]
                    continue
                bracket_index = remainder.find("[")
                if bracket_index == -1:
                    tokens.append(remainder)
                    break
                if bracket_index > 0:
                    tokens.append(remainder[:bracket_index])
                remainder = remainder[bracket_index:]
        return tokens

    def _apply_token(self, current: Any, token: str) -> list[Any]:
        if current is None:
            return []
        if token == "[]":
            if isinstance(current, list):
                return list(current)
            return [current]
        if token.startswith("[") and token.endswith("]"):
            index_text = token[1:-1].strip()
            if not index_text:
                return [current] if not isinstance(current, list) else list(current)
            if isinstance(current, list):
                index = int(index_text)
                if -len(current) <= index < len(current):
                    return [current[index]]
            return []

        if isinstance(current, dict):
            if token in current:
                return [current[token]]
            return []
        if isinstance(current, list):
            collected: list[Any] = []
            for item in current:
                if isinstance(item, dict) and token in item:
                    collected.append(item[token])
                elif hasattr(item, token):
                    collected.append(getattr(item, token))
                elif token in {"name", "value"}:
                    collected.append(item)
            return collected
        if hasattr(current, token):
            return [getattr(current, token)]
        if token in {"name", "value"}:
            return [current]
        return []

    def _normalize_value(self, value: Any, normalize: str | None, type_name: str, source_path: str) -> Any:
        if normalize is None:
            return self._dedupe_lists(value)

        if self._is_list_type(type_name):
            values = value if isinstance(value, list) else [value]
            return self._normalize_list(values, normalize, source_path)

        return self._normalize_scalar(value, normalize, source_path)

    def _normalize_list(self, values: list[Any], normalize: str, source_path: str) -> list[Any]:
        normalized_items = [self._normalize_scalar(item, normalize, source_path) for item in values]
        normalized_items = [item for item in normalized_items if item not in (None, "")]
        deduped: list[Any] = []
        seen: set[str] = set()
        for item in normalized_items:
            marker = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(item)
        return deduped

    def _normalize_scalar(self, value: Any, normalize: str, source_path: str) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            normalized_list = [self._normalize_scalar(item, normalize, source_path) for item in value]
            return [item for item in normalized_list if item not in (None, "")]

        normalize_key = normalize.lower()
        if normalize_key == "e164":
            return normalize_phone(value)
        if normalize_key == "canonical":
            if "skill" in source_path.lower():
                return normalize_skill(value)
            return normalize_text(value)
        if normalize_key in {"yyyy-mm", "year-month", "date"}:
            return normalize_date(value)
        if normalize_key in {"text", "string"}:
            return normalize_text(value)
        return value

    def _dedupe_lists(self, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        deduped: list[Any] = []
        seen: set[str] = set()
        for item in value:
            marker = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(item)
        return deduped

    def _validate_and_coerce_type(self, value: Any, type_name: str, field_path: str) -> Any:
        if value is None:
            return None

        normalized_type = type_name.strip().lower()
        if normalized_type == "any":
            return value
        if normalized_type in {"string", "str"}:
            if isinstance(value, list):
                if len(value) == 1:
                    return self._validate_scalar(value[0], field_path)
                raise ValueError(f"Field '{field_path}' expected a string but received a list")
            return self._validate_scalar(value, field_path)
        if normalized_type in {"string[]", "str[]"}:
            values = value if isinstance(value, list) else [value]
            return [self._validate_scalar(item, field_path) for item in values if item is not None]
        if normalized_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            raise ValueError(f"Field '{field_path}' expected a number")
        if normalized_type == "boolean":
            if isinstance(value, bool):
                return value
            raise ValueError(f"Field '{field_path}' expected a boolean")
        if normalized_type == "object":
            if isinstance(value, dict):
                return value
            raise ValueError(f"Field '{field_path}' expected an object")
        if normalized_type == "object[]":
            values = value if isinstance(value, list) else [value]
            if all(isinstance(item, dict) for item in values):
                return values
            raise ValueError(f"Field '{field_path}' expected an object array")
        raise ValueError(f"Unsupported projection type '{type_name}' for field '{field_path}'")

    def _validate_scalar(self, value: Any, field_path: str) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            raise ValueError(f"Field '{field_path}' expected a scalar but received an object")
        if isinstance(value, list):
            raise ValueError(f"Field '{field_path}' expected a scalar but received a list")
        return str(value)

    def _set_output_value(self, target: dict[str, Any], path: str, value: Any) -> None:
        tokens = path.split(".")
        current = target
        for token in tokens[:-1]:
            if token not in current or not isinstance(current[token], dict):
                current[token] = {}
            current = current[token]
        current[tokens[-1]] = value

    def _is_missing(self, value: Any) -> bool:
        return value in (None, "", [], {})

    def _is_list_type(self, type_name: str) -> bool:
        return type_name.strip().lower().endswith("[]")
