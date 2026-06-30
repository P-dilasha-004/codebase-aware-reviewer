"""
Phase 1 — Track B: Prose and documentation chunker (Markdown / plain text).

Strategy:
- Split at Markdown header boundaries (#, ##, ###, ####, etc.).
- Target chunk size: 300–600 tokens. Hard cap: 800 tokens.
- Sub-sections inherit the two immediate parent header titles so retrieved
  chunks carry enough context to locate themselves in the document.
- Every chunk is prefixed with [Source: <filename>] on the first line.
- Overlap: ~100 tokens, applied only when a section body exceeds 800 tokens
  to preserve narrative continuity across paragraph splits.
- Plain-text (.txt) files are treated as a single Markdown section with no
  headers and chunked by paragraph boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

# ── Token counting ─────────────────────────────────────────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENC.encode(text))


MAX_TOKENS     = 800
TARGET_MAX     = 600
OVERLAP_TOKENS = 100

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    file_path: str
    content: str
    file_type: str = "prose_doc"
    chunk_strategy: str = "markdown_header"
    target_module: Optional[str] = None
    commit_hash: str = ""
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = _count(self.content)


# ── Header parsing ────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)


def _source_label(file_path: str) -> str:
    import os
    return f"[Source: {os.path.basename(file_path)}]"


def _build_prefix(file_path: str, breadcrumb: str) -> str:
    label = _source_label(file_path)
    if breadcrumb:
        return f"{label}\n{breadcrumb}"
    return label


# ── Overlap helper ────────────────────────────────────────────────────────────

def _tail_tokens(text: str, n: int) -> str:
    words = text.split()
    tail: list[str] = []
    count = 0
    for word in reversed(words):
        wt = _count(word + " ")
        if count + wt > n:
            break
        tail.append(word)
        count += wt
    return " ".join(reversed(tail))


# ── Body splitting ────────────────────────────────────────────────────────────

def _split_body(body: str, prefix: str, file_path: str) -> list[str]:
    full = f"{prefix}\n\n{body}".strip()
    if _count(full) <= MAX_TOKENS:
        return [full]

    paragraphs = re.split(r'\n{2,}', body.strip())
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    overlap_tail = ""

    for para in paragraphs:
        para_tokens = _count(para)
        preamble_tokens = _count(prefix) + 2

        if buf_tokens + para_tokens + preamble_tokens > MAX_TOKENS and buf:
            body_text = "\n\n".join(buf)
            chunks.append(f"{prefix}\n\n{overlap_tail}{body_text}".strip())
            overlap_tail = _tail_tokens(body_text, OVERLAP_TOKENS)
            if overlap_tail:
                overlap_tail += "\n\n"
            buf = []
            buf_tokens = 0

        # A single paragraph that exceeds the cap on its own: split at words.
        if para_tokens + preamble_tokens > MAX_TOKENS:
            words = para.split()
            word_buf: list[str] = []
            word_buf_tokens = 0
            for word in words:
                wt = _count(word + " ")
                if word_buf_tokens + wt + preamble_tokens > MAX_TOKENS and word_buf:
                    sub = " ".join(word_buf)
                    chunks.append(f"{prefix}\n\n{overlap_tail}{sub}".strip())
                    overlap_tail = _tail_tokens(sub, OVERLAP_TOKENS)
                    if overlap_tail:
                        overlap_tail += "\n\n"
                    word_buf = []
                    word_buf_tokens = 0
                word_buf.append(word)
                word_buf_tokens += wt
            if word_buf:
                sub = " ".join(word_buf)
                buf.append(sub)
                buf_tokens += _count(sub) + 2
            continue

        buf.append(para)
        buf_tokens += para_tokens + 2

    if buf:
        body_text = "\n\n".join(buf)
        chunks.append(f"{prefix}\n\n{overlap_tail}{body_text}".strip())

    return chunks if chunks else [full]


# ── Markdown section extraction ───────────────────────────────────────────────

def _parse_sections(source: str) -> list[tuple[int, str, str]]:
    sections: list[tuple[int, str, str]] = []
    matches = list(_HEADER_RE.finditer(source))

    if not matches:
        return [(0, "", source.strip())]

    preamble = source[:matches[0].start()].strip()
    if preamble:
        sections.append((0, "", preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[body_start:body_end].strip()
        sections.append((level, title, body))

    return sections


def _breadcrumb(ancestors: list[tuple[int, str]], max_parents: int = 2) -> str:
    if not ancestors:
        return ""
    tail = ancestors[-max_parents:]
    return " > ".join(f"{'#' * lvl} {title}" for lvl, title in tail)


# ── Public entry point ────────────────────────────────────────────────────────

def chunk_prose_file(file_path: str, source_text: str) -> list[Chunk]:
    if not source_text.strip():
        return []

    sections = _parse_sections(source_text)
    chunks: list[Chunk] = []

    ancestor_stack: list[tuple[int, str]] = []

    for level, title, body in sections:
        if level > 0:
            while ancestor_stack and ancestor_stack[-1][0] >= level:
                ancestor_stack.pop()

        breadcrumb = _breadcrumb(ancestor_stack)
        if title:
            own_header = f"{'#' * level} {title}"
            if breadcrumb:
                prefix = f"{_source_label(file_path)}\n{breadcrumb}\n{own_header}"
            else:
                prefix = f"{_source_label(file_path)}\n{own_header}"
        else:
            prefix = _source_label(file_path)

        if body:
            for raw in _split_body(body, prefix, file_path):
                chunks.append(Chunk(file_path=file_path, content=raw))

        if level > 0:
            ancestor_stack.append((level, title))

    return chunks
