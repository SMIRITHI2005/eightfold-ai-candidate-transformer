"""Application configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .projection import ProjectionConfig


class AppSettings(BaseModel):
    """Runtime settings for the transformer."""

    model_config = ConfigDict(extra="forbid")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    spacy_model: str = "en_core_web_sm"
    default_region: str = "US"


@dataclass(slots=True)
class RuntimeContext:
    """Mutable runtime context for deterministic execution."""

    settings: AppSettings = field(default_factory=AppSettings)
    projection: ProjectionConfig | None = None
