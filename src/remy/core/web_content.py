"""Fail-soft HTTP and visible-text extraction for public web pages."""

from __future__ import annotations

from html.parser import HTMLParser
import re
import urllib.request


_MAX_HTML_BYTES = 4 * 1024 * 1024
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "canvas"})


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "div", "section", "article", "main", "header", "footer", "nav", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "section", "article", "main", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        text = data.strip()
        if self._in_title and not self.title:
            self.title = text
        self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def fetch_html(url: str, *, timeout: int = 20) -> str:
    """Fetch HTML with a browser-like but honest agent User-Agent."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Remy-Agent/1.0; +local-assistant)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "uk,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("Content-Type") or "")
        if "html" not in content_type.casefold():
            raise ValueError(f"Expected HTML response, got {content_type or 'unknown content type'}")
        raw = response.read(_MAX_HTML_BYTES + 1)
        if len(raw) > _MAX_HTML_BYTES:
            raise ValueError("HTML response exceeds 4 MiB safety limit")
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def extract_visible_text(html: str) -> tuple[str, str]:
    """Return server-rendered visible text and the page title."""
    parser = _VisibleTextParser()
    parser.feed(html or "")
    parser.close()
    return parser.text(), parser.title
