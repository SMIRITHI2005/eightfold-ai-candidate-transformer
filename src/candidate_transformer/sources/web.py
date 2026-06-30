"""Remote profile source adapters for GitHub and LinkedIn URLs."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ..models import EvidenceSourceType
from ..normalization import normalize_text
from .base import SourceAdapter, SourceDocument


class _ProfileHTMLParser(HTMLParser):
    """Collect lightweight profile metadata from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self.anchors: list[dict[str, str]] = []
        self._text_chunks: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() == "meta":
            key = attributes.get("property") or attributes.get("name")
            content = attributes.get("content")
            if key and content:
                self.meta[key.lower()] = content
        elif tag.lower() == "a":
            href = attributes.get("href") or ""
            self.anchors.append({"href": href, "text": ""})

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = normalize_text(data)
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        self._text_chunks.append(text)
        if self.anchors:
            last_anchor = self.anchors[-1]
            if last_anchor.get("text") == "":
                last_anchor["text"] = text

    @property
    def text(self) -> str:
        return "\n".join(self._text_chunks)


class WebProfileAdapter(SourceAdapter):
    """Adapter for GitHub and LinkedIn profile URLs."""

    def __init__(self, source_type: EvidenceSourceType):
        self.source_type = source_type

    def can_handle(self, source: Path | str) -> bool:
        parsed = self._parse_source(source)
        return self._is_supported_host(parsed.netloc)

    def load(self, source: Path | str) -> SourceDocument:
        url = self._source_to_url(source)
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 CandidateTransformer/1.0"}, timeout=20)
        response.raise_for_status()
        parser = _ProfileHTMLParser()
        parser.feed(response.text)
        raw = self._extract_structured_payload(url, parser)
        text = self._build_text(parser, raw)
        return SourceDocument(
            path=None,
            source=url,
            source_type=self.source_type,
            text=text,
            raw=raw,
            metadata={"url": url, "host": urlparse(url).netloc, "title": parser.title},
        )

    def _source_to_url(self, source: Path | str) -> str:
        if isinstance(source, Path):
            return source.as_uri()
        return str(source)

    def _parse_source(self, source: Path | str):
        return urlparse(self._source_to_url(source))

    def _is_supported_host(self, host: str) -> bool:
        normalized_host = host.lower()
        return normalized_host.endswith("github.com") or normalized_host.endswith("linkedin.com")

    def _extract_structured_payload(self, url: str, parser: _ProfileHTMLParser) -> dict[str, Any]:
        host = urlparse(url).netloc.lower()
        payload: dict[str, Any] = {
            "urls": [url],
            "source_url": url,
        }
        title = parser.title or parser.meta.get("og:title") or parser.meta.get("twitter:title")
        description = (
            parser.meta.get("description")
            or parser.meta.get("og:description")
            or parser.meta.get("twitter:description")
        )
        canonical = parser.meta.get("og:url") or parser.meta.get("twitter:url") or parser.meta.get("canonical")
        if canonical:
            payload.setdefault("urls", []).append(canonical)

        emails = self._extract_emails(parser.text)
        if emails:
            payload["emails"] = emails

        if "github" in host:
            payload.update(self._github_payload(url, title, description, parser))
        else:
            payload.update(self._linkedin_payload(url, title, description, parser))
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

    def _github_payload(self, url: str, title: str | None, description: str | None, parser: _ProfileHTMLParser) -> dict[str, Any]:
        name = self._name_from_title(title)
        headline = description or parser.meta.get("og:description")
        companies = self._extract_companies(parser.text)
        payload: dict[str, Any] = {
            "full_name": name,
            "headline": headline,
            "summary": description,
            "companies": companies,
            "urls": [url],
        }
        return payload

    def _linkedin_payload(self, url: str, title: str | None, description: str | None, parser: _ProfileHTMLParser) -> dict[str, Any]:
        name = self._name_from_title(title)
        headline = description or parser.meta.get("og:description")
        location = self._extract_location(parser.text)
        payload: dict[str, Any] = {
            "full_name": name,
            "headline": headline,
            "summary": description,
            "location": location,
            "urls": [url],
        }
        return payload

    def _build_text(self, parser: _ProfileHTMLParser, raw: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("full_name", "headline", "summary", "location"):
            value = raw.get(key)
            if value:
                parts.append(str(value))
        for key in ("emails", "urls", "companies", "skills"):
            value = raw.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
        parts.append(parser.text)
        return "\n".join(part for part in parts if part)

    def _name_from_title(self, title: str | None) -> str | None:
        if not title:
            return None
        cleaned = title.replace("| LinkedIn", "").replace("- GitHub", "").strip()
        match = re.split(r"\s+[\|\-]\s+", cleaned, maxsplit=1)
        if match:
            return normalize_text(match[0])
        return normalize_text(cleaned)

    def _extract_emails(self, text: str) -> list[str]:
        matches = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)
        deduped: list[str] = []
        seen: set[str] = set()
        for match in matches:
            lowered = match.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(lowered)
        return deduped

    def _extract_companies(self, text: str) -> list[str]:
        matches = re.findall(r"(?:works at|company|employer)\s*[:\-]\s*([^\n]+)", text, flags=re.IGNORECASE)
        return [normalize_text(match) for match in matches if normalize_text(match)]

    def _extract_location(self, text: str) -> str | None:
        match = re.search(r"(?:location|based in)\s*[:\-]\s*([^\n]+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        return normalize_text(match.group(1))
