"""Source adapters for candidate inputs."""

from .base import SourceAdapter, SourceDocument
from .files import auto_detect_adapter

__all__ = ["SourceAdapter", "SourceDocument", "auto_detect_adapter"]
