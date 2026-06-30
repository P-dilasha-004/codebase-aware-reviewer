"""
Phase 1 — Track C: Configuration and manifest chunker (.toml, .json, .yaml).

Strategy:
- Parse the file as a structured object (TOML / JSON / YAML).
- Flatten each top-level section into declarative English sentences so the
  content is searchable in natural language rather than raw syntax.
  e.g. {"debug": true} → "The debug flag is set to true."
- Massive-file guardrail: files exceeding 2 000 lines bypass full translation
  and emit only a single metadata chunk (file path + top-level keys).
- Target chunk size: 100–300 tokens. Hard cap: 800 tokens.
- Zero overlap: declarative key-value structures have no narrative adjacency.
- chunk_strategy: "llm_translation"
- file_type: "config"
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tiktoken
import yaml

# ── Token counting ─────────────────────────────────────────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENC.encode(text))


MAX_TOKENS    = 800
TARGET_MAX    = 300
MASSIVE_LINES = 2_000

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    file_path: str
    content: str
    file_type: str = "config"
    chunk_strategy: str = "llm_translation"
    target_module: Optional[str] = None
    commit_hash: str = ""
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = _count(self.content)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse(file_path: str, source_text: str) -> dict | list | None:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".toml":
            return tomllib.loads(source_text)
        if ext == ".json":
            return json.loads(source_text)
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(source_text)
    except Exception:
        return None
    return None


# ── Declarative sentence generation ──────────────────────────────────────────

def _value_repr(val: Any) -> str:
    if isinstance(val, bool):
        return "enabled" if val else "disabled"
    if val is None:
        return "not set"
    if isinstance(val, list):
        if not val:
            return "an empty list"
        if len(val) <= 5:
            return ", ".join(str(v) for v in val)
        return f"{', '.join(str(v) for v in val[:5])}, … ({len(val)} items)"
    return str(val)


def _key_to_phrase(key: str) -> str:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
    s = re.sub(r'[_\-]', ' ', s)
    return s.lower()


def _split_sentence(sentence: str, file_path: str) -> list[str]:
    """Truncate an oversized single sentence at word boundary with [truncated] marker."""
    words = sentence.split()
    buf: list[str] = []
    header = f"[Config: {file_path}]"
    cap = MAX_TOKENS - _count(header) - _count(" [truncated]") - 2
    tok = 0
    for word in words:
        wt = _count(word + " ")
        if tok + wt > cap:
            break
        buf.append(word)
        tok += wt
    return [f"{header}\n{' '.join(buf)} [truncated]"]


def _flatten_to_sentences(obj: Any, prefix: str = "") -> list[str]:
    sentences: list[str] = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            key = str(key)  # YAML 1.1 can parse bare keys like 'on'/'yes' as booleans
            full_key = f"{prefix}.{key}" if prefix else key
            phrase = _key_to_phrase(full_key)

            if isinstance(val, dict):
                sentences.append(f"The {phrase} section contains the following settings.")
                sentences.extend(_flatten_to_sentences(val, prefix=full_key))
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                sentences.append(
                    f"The {phrase} list contains {len(val)} item(s)."
                )
            else:
                sentences.append(f"The {phrase} is {_value_repr(val)}.")

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            sentences.extend(_flatten_to_sentences(item, prefix=f"{prefix}[{i}]"))

    return sentences


# ── Massive-file guardrail ────────────────────────────────────────────────────

def _metadata_only_chunk(file_path: str, obj: dict | list | None,
                          source_text: str) -> Chunk:
    if isinstance(obj, dict):
        keys = ", ".join(list(obj.keys())[:20])
        suffix = f" (and {len(obj) - 20} more)" if len(obj) > 20 else ""
        body = (
            f"[Config: {file_path}]\n"
            f"This file has more than {MASSIVE_LINES} lines and was not fully translated.\n"
            f"Top-level keys: {keys}{suffix}."
        )
    else:
        body = (
            f"[Config: {file_path}]\n"
            f"This file has more than {MASSIVE_LINES} lines and was not fully translated."
        )
    return Chunk(file_path=file_path, content=body)


# ── Chunk assembly ────────────────────────────────────────────────────────────

def _make_chunks(file_path: str, sentences: list[str]) -> list[Chunk]:
    header = f"[Config: {file_path}]"
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = _count(header) + 1

    for sentence in sentences:
        st = _count(sentence) + 1
        # Oversized single sentence — truncate it
        if _count(header) + 1 + st > MAX_TOKENS and not buf:
            chunks.extend(
                Chunk(file_path=file_path, content=c)
                for c in _split_sentence(sentence, file_path)
            )
            continue

        if buf_tokens + st > MAX_TOKENS and buf:
            chunks.append(Chunk(
                file_path=file_path,
                content=header + "\n" + "\n".join(buf),
            ))
            buf = []
            buf_tokens = _count(header) + 1

        buf.append(sentence)
        buf_tokens += st

    if buf:
        chunks.append(Chunk(
            file_path=file_path,
            content=header + "\n" + "\n".join(buf),
        ))

    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

def chunk_config_file(file_path: str, source_text: str) -> list[Chunk]:
    if not source_text.strip():
        return []

    line_count = source_text.count("\n")

    if line_count > MASSIVE_LINES:
        obj = _parse(file_path, source_text)
        return [_metadata_only_chunk(file_path, obj, source_text)]

    obj = _parse(file_path, source_text)

    if obj is None:
        header = f"[Config: {file_path}]"
        raw = source_text[:4000]
        content = f"{header}\n{raw}"
        if _count(content) <= MAX_TOKENS:
            return [Chunk(file_path=file_path, content=content)]
        words = raw.split()
        trunc: list[str] = []
        tok = _count(header) + 1
        for w in words:
            wt = _count(w + " ")
            if tok + wt > MAX_TOKENS:
                break
            trunc.append(w)
            tok += wt
        return [Chunk(file_path=file_path, content=f"{header}\n{' '.join(trunc)}")]

    sentences = _flatten_to_sentences(obj)
    if not sentences:
        return []

    return _make_chunks(file_path, sentences)
