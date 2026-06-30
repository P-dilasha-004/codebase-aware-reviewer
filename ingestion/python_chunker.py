"""
Phase 1 — Track A: Python source code chunker (AST-based).

Strategy:
- Primary: tree-sitter-python parses at function/class boundaries.
- Fallback: regex-based splitting on `def ` / `class ` lines if tree-sitter fails.
- Chunk size cap: ~800 tokens (tiktoken cl100k_base approximation).
- Oversized entities are split at internal logical junctures (try/except, for/while loops).
- Every chunk is prefixed with an immutable context header.
- File-level imports + globals are extracted and prepended to every functional chunk.
- Test file detection: heuristic filename match (src/auth.py → tests/test_auth.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENC.encode(text))


MAX_TOKENS = 800

_PY_LANG = Language(tspython.language())
_PARSER  = Parser(_PY_LANG)


@dataclass
class Chunk:
    file_path: str
    content: str
    file_type: str = "source_code"
    chunk_strategy: str = "ast_function"
    target_module: Optional[str] = None
    commit_hash: str = ""
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = _count(self.content)


def _header(file_path: str, class_name: str | None, func_name: str | None,
            block: str | None = None) -> str:
    parts = [f"File: {file_path}"]
    if class_name:
        parts.append(f"Class: {class_name}")
    if func_name:
        parts.append(f"Method: {func_name}")
    if block:
        parts.append(f"Block: {block}")
    return "// Context: " + " | ".join(parts)


def _infer_target_module(file_path: str) -> str | None:
    p = Path(file_path)
    name = p.stem
    if name.startswith("test_"):
        source_stem = name[len("test_"):]
        parts = list(p.parts)
        for i, part in enumerate(parts):
            if part in ("tests", "test"):
                parts[i] = "src"
                break
        parts[-1] = source_stem + ".py"
        return str(Path(*parts))
    return None


def _split_oversized(text: str, file_path: str, class_name: str | None,
                     func_name: str | None) -> list[str]:
    juncture = re.compile(
        r'^(\s{0,8})(try:|except\b|for\b|while\b|with\b|if\b|elif\b|else:)',
        re.MULTILINE,
    )
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush(block_hint: str | None = None) -> None:
        if not buf:
            return
        body = "".join(buf)
        hdr = _header(file_path, class_name, func_name, block_hint)
        chunks.append(f"{hdr}\n{body}")
        buf.clear()

    block_label: str | None = None
    for line in lines:
        m = juncture.match(line)
        if m and buf_tokens >= MAX_TOKENS // 2:
            flush(block_label)
            buf_tokens = 0
            block_label = m.group(2).rstrip(":")
        buf.append(line)
        buf_tokens += _count(line)
        if buf_tokens >= MAX_TOKENS - 20:
            flush(block_label)
            buf_tokens = 0
            block_label = None

    flush(block_label)
    return chunks if chunks else [f"{_header(file_path, class_name, func_name)}\n{text}"]


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_imports_and_globals(tree_root: Node, source: bytes) -> str:
    lines: list[str] = []
    for child in tree_root.children:
        if child.type in ("import_statement", "import_from_statement",
                          "expression_statement"):
            text = _node_text(child, source).strip()
            if child.type == "expression_statement":
                if "=" not in text or text.startswith("("):
                    continue
            lines.append(text)
    return "\n".join(lines)


def _walk_entities(tree_root: Node, source: bytes, file_path: str,
                   header_text: str = "",
                   class_name: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []

    for node in tree_root.children:
        inner = node
        if node.type == "decorated_definition":
            inner = next(
                (c for c in node.children
                 if c.type in ("function_definition", "class_definition")),
                None,
            )
            if inner is None:
                continue

        if inner.type == "class_definition":
            name_node = inner.child_by_field_name("name")
            cname = _node_text(name_node, source) if name_node else "UnknownClass"
            qualified = f"{class_name}.{cname}" if class_name else cname
            body = inner.child_by_field_name("body")
            if body:
                chunks.extend(_walk_entities(body, source, file_path,
                                             header_text=header_text,
                                             class_name=qualified))
            else:
                text = _node_text(node, source)
                hdr = _header(file_path, qualified, None)
                body_content = f"{header_text}\n{text}" if header_text else text
                chunks.append(Chunk(file_path=file_path,
                                    content=f"{hdr}\n{body_content}",
                                    chunk_strategy="ast_function"))

        elif inner.type == "function_definition":
            name_node = inner.child_by_field_name("name")
            fname = _node_text(name_node, source) if name_node else "unknown"
            text = _node_text(node, source)
            hdr = _header(file_path, class_name, fname)
            body_content = f"{header_text}\n{text}" if header_text else text

            full = f"{hdr}\n{body_content}"
            if _count(full) <= MAX_TOKENS:
                chunks.append(Chunk(file_path=file_path, content=full,
                                    chunk_strategy="ast_function"))
            else:
                for sub in _split_oversized(body_content, file_path, class_name, fname):
                    chunks.append(Chunk(file_path=file_path, content=sub,
                                        chunk_strategy="ast_function"))

    return chunks


def _regex_chunks(source: str, file_path: str) -> list[Chunk]:
    pattern = re.compile(r'^(?=def |class )', re.MULTILINE)
    parts = pattern.split(source)
    chunks: list[Chunk] = []

    for part in parts:
        if not part.strip():
            continue
        first = part.splitlines()[0] if part.splitlines() else ""
        m = re.match(r'(?:def|class)\s+(\w+)', first)
        fname = m.group(1) if m else None
        hdr = _header(file_path, None, fname)
        full = f"{hdr}\n{part}"

        if _count(full) <= MAX_TOKENS:
            chunks.append(Chunk(file_path=file_path, content=full,
                                chunk_strategy="ast_function"))
        else:
            for sub in _split_oversized(part, file_path, None, fname):
                chunks.append(Chunk(file_path=file_path, content=sub,
                                    chunk_strategy="ast_function"))

    return chunks


def chunk_python_file(file_path: str, source_text: str) -> list[Chunk]:
    target_module = _infer_target_module(file_path)
    chunks: list[Chunk] = []

    try:
        source_bytes = source_text.encode("utf-8")
        tree = _PARSER.parse(source_bytes)
        root = tree.root_node

        header_text = _extract_imports_and_globals(root, source_bytes)
        functional = _walk_entities(root, source_bytes, file_path,
                                    header_text=header_text)

        for c in functional:
            c.target_module = target_module
            chunks.append(c)

    except Exception:
        for c in _regex_chunks(source_text, file_path):
            c.target_module = target_module
            chunks.append(c)

    if not chunks:
        hdr = _header(file_path, None, None)
        content = f"{hdr}\n{source_text}"
        if _count(content) <= MAX_TOKENS:
            chunks.append(Chunk(file_path=file_path, content=content,
                                chunk_strategy="ast_function",
                                target_module=target_module))
        else:
            for sub in _split_oversized(source_text, file_path, None, None):
                c = Chunk(file_path=file_path, content=sub,
                          chunk_strategy="ast_function",
                          target_module=target_module)
                chunks.append(c)

    return chunks
