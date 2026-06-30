"""Local Ollama client for semantic extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class OllamaClient:
    """Small Ollama JSON client with deterministic failure handling."""

    host: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: float = 30.0

    def generate(self, prompt: str) -> str | None:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        return body.get("response") or body.get("message", {}).get("content")

    def generate_json(self, prompt: str) -> dict[str, Any] | None:
        response = self.generate(prompt)
        if not response:
            return None
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
