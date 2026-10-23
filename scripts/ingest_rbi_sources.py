"""Fetch the allowlisted RBI pages and build a local, cited text index."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "rbi_sources" / "catalog.json"
INDEX_PATH = PROJECT_ROOT / "data" / "rbi_sources" / "index.jsonl"
ALLOWED_HOSTS = {"rbi.org.in", "www.rbi.org.in"}
MAX_PAGE_BYTES = 15 * 1024 * 1024
MAX_CHUNK_CHARS = 1_800
CHUNK_OVERLAP_CHARS = 180


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        _attrs: list[tuple[str, str | None]],
    ) -> None:
        _ = _attrs
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag.casefold() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag.casefold() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self.parts.append(cleaned)


def _fetch_source(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Source URL is not on the HTTPS RBI allowlist: {url}")

    with requests.get(
        url,
        headers={"User-Agent": "DriftingOracleSourceIndexer/1.0"},
        timeout=(8, 25),
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        final = urlparse(response.url)
        if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
            raise ValueError("RBI source redirected outside the official RBI domain.")
        content_type = response.headers.get("Content-Type", "").casefold()
        if "html" not in content_type:
            raise ValueError("The allowlisted RBI source did not return an HTML page.")
        body = bytearray()
        for part in response.iter_content(chunk_size=64 * 1024):
            body.extend(part)
            if len(body) > MAX_PAGE_BYTES:
                raise ValueError("RBI source page exceeded the configured size limit.")

    parser = _VisibleTextParser()
    parser.feed(body.decode(response.encoding or "utf-8", errors="replace"))
    text = re.sub(r"[ \t]+\n", "\n", "\n".join(parser.parts))
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text.strip()) < 300:
        raise ValueError("RBI source page did not contain enough readable text.")
    return text.strip()


def _split_chunks(text: str) -> list[tuple[str, str]]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[tuple[str, str]] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > MAX_CHUNK_CHARS:
            if current:
                chunks.append((current, "Official RBI page excerpt"))
                current = ""
            chunks.append((paragraph[:MAX_CHUNK_CHARS], "Official RBI page excerpt"))
            paragraph = paragraph[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS :]
        if current and len(current) + len(paragraph) + 1 > MAX_CHUNK_CHARS:
            chunks.append((current, "Official RBI page excerpt"))
            current = ""
        current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append((current, "Official RBI page excerpt"))
    return chunks


def build_index() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    records = []
    for source in catalog.get("sources", []):
        text = _fetch_source(source["url"])
        for index, (excerpt, section) in enumerate(_split_chunks(text)):
            records.append(
                {
                    "source_id": source["source_id"],
                    "title": source["title"],
                    "url": source["url"],
                    "issued_on": source.get("issued_on"),
                    "updated_as_of": source.get("updated_as_of"),
                    "currentness_status": source["currentness_status"],
                    "status_note": source["status_note"],
                    "topics": source.get("topics", []),
                    "keywords": source.get("keywords", []),
                    "section": f"{section} {index + 1}",
                    "text": excerpt,
                }
            )
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)


if __name__ == "__main__":
    print(f"Created local RBI index with {build_index()} excerpts.")
