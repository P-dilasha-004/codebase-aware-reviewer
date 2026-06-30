"""
Tests for ingestion/prose_chunker.py — Markdown / plain-text chunker.

Run with: python3 -m pytest ingestion/tests/test_prose_chunker.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from ingestion.prose_chunker import (
    chunk_prose_file, MAX_TOKENS, TARGET_MAX, OVERLAP_TOKENS,
    _parse_sections, _breadcrumb, _tail_tokens, _source_label, _count,
    Chunk,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_MD = """\
# Introduction

This is the introduction section.

## Background

Some background information here.

### Details

Very specific details about the background.
"""

FLAT_MD = """\
# Section One

Content of section one.

# Section Two

Content of section two.

# Section Three

Content of section three.
"""

PREAMBLE_MD = """\
This text appears before any header.

It has multiple paragraphs.

# First Header

Content under first header.
"""

PLAIN_TEXT = """\
This is a plain text file.

It has no headers at all.

Just paragraphs separated by blank lines.
"""

EMPTY_MD = ""
WHITESPACE_MD = "   \n\n   \n"


# ── Source label ──────────────────────────────────────────────────────────────

class TestSourceLabel:
    def test_every_chunk_starts_with_source_label(self):
        chunks = chunk_prose_file("docs/README.md", SIMPLE_MD)
        for c in chunks:
            assert c.content.startswith("[Source:"), (
                f"Chunk missing [Source:] label:\n{c.content[:80]}"
            )

    def test_source_label_contains_basename_only(self):
        chunks = chunk_prose_file("docs/nested/guide.md", SIMPLE_MD)
        first_line = chunks[0].content.splitlines()[0]
        assert "guide.md" in first_line
        assert "docs/nested" not in first_line

    def test_source_label_format(self):
        label = _source_label("path/to/file.md")
        assert label == "[Source: file.md]"

    def test_txt_file_source_label(self):
        chunks = chunk_prose_file("notes.txt", PLAIN_TEXT)
        for c in chunks:
            assert "[Source: notes.txt]" in c.content


# ── Token cap ─────────────────────────────────────────────────────────────────

class TestTokenCap:
    def test_all_chunks_within_max_tokens(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over, f"{len(over)} chunks exceed {MAX_TOKENS} tokens"

    def test_max_tokens_constant_is_800(self):
        assert MAX_TOKENS == 800

    def test_overlap_tokens_constant_is_100(self):
        assert OVERLAP_TOKENS == 100

    def test_large_section_split_within_cap(self):
        # Section body >> 800 tokens — must be split
        big_body = "# Big Section\n\n" + ("word " * 200 + "\n\n") * 10
        chunks = chunk_prose_file("big.md", big_body)
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over

    def test_single_oversized_paragraph_split_within_cap(self):
        # One paragraph with no blank-line breaks, > 800 tokens
        giant_para = "# Section\n\n" + ("word " * 900)
        chunks = chunk_prose_file("big.md", giant_para)
        assert chunks, "Expected at least one chunk"
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over, (
            f"{len(over)} chunk(s) exceed {MAX_TOKENS} tokens: "
            + str([c.token_count for c in over])
        )

    def test_token_count_on_chunk_is_accurate(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.token_count == _count(c.content)


# ── Header-based splitting ────────────────────────────────────────────────────

class TestHeaderSplitting:
    def test_each_h1_section_produces_chunk(self):
        chunks = chunk_prose_file("README.md", FLAT_MD)
        full = "\n".join(c.content for c in chunks)
        assert "Section One" in full
        assert "Section Two" in full
        assert "Section Three" in full

    def test_nested_headers_produce_chunks(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        full = "\n".join(c.content for c in chunks)
        assert "Introduction" in full
        assert "Background" in full
        assert "Details" in full

    def test_header_levels_h1_through_h4(self):
        src = "# H1\n\nbody1\n\n## H2\n\nbody2\n\n### H3\n\nbody3\n\n#### H4\n\nbody4\n"
        chunks = chunk_prose_file("doc.md", src)
        full = "\n".join(c.content for c in chunks)
        assert "H1" in full
        assert "H2" in full
        assert "H3" in full
        assert "H4" in full


# ── Breadcrumb inheritance ────────────────────────────────────────────────────

class TestBreadcrumbInheritance:
    def test_subsection_inherits_parent_header(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        details_chunks = [c for c in chunks if "Details" in c.content]
        assert details_chunks
        # "Background" should appear as ancestor context
        assert any("Background" in c.content for c in details_chunks)

    def test_breadcrumb_uses_at_most_two_ancestors(self):
        src = "# L1\n\n## L2\n\n### L3\n\n#### L4\n\nbody\n"
        chunks = chunk_prose_file("deep.md", src)
        l4_chunks = [c for c in chunks if "L4" in c.content]
        assert l4_chunks
        # Should show at most 2 ancestors (L2 > L3), not all the way to L1
        for c in l4_chunks:
            lines = c.content.splitlines()
            ancestor_lines = [l for l in lines if ">" in l or l.startswith("#")]
            # At most 2 ancestor headers plus own header = 3 header lines
            assert len(ancestor_lines) <= 3

    def test_breadcrumb_format_uses_arrow_separator(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        details_chunks = [c for c in chunks if "Details" in c.content
                          and "Background" in c.content]
        assert details_chunks
        assert any(">" in c.content for c in details_chunks)

    def test_top_level_section_has_no_breadcrumb_ancestors(self):
        chunks = chunk_prose_file("README.md", FLAT_MD)
        intro_chunks = [c for c in chunks if "Section One" in c.content]
        assert intro_chunks
        # No ">" separator — no ancestors at level 1
        for c in intro_chunks:
            lines = c.content.splitlines()
            # Only the source label and own header — no ancestor line with ">"
            non_label_non_header = [
                l for l in lines[1:]
                if l.strip() and not l.startswith("#") and ">" not in l
            ]
            # Content lines OK, just no breadcrumb ancestor
            breadcrumb_lines = [l for l in lines if " > " in l]
            assert not breadcrumb_lines

    def test_breadcrumb_helper_returns_empty_for_no_ancestors(self):
        assert _breadcrumb([]) == ""

    def test_breadcrumb_helper_single_ancestor(self):
        result = _breadcrumb([(1, "Introduction")])
        assert "Introduction" in result

    def test_breadcrumb_helper_two_ancestors(self):
        result = _breadcrumb([(1, "Parent"), (2, "Child")])
        assert "Parent" in result
        assert "Child" in result
        assert " > " in result

    def test_breadcrumb_helper_truncates_to_two(self):
        result = _breadcrumb([(1, "A"), (2, "B"), (3, "C")])
        assert "A" not in result  # only last 2
        assert "B" in result
        assert "C" in result


# ── Preamble handling ─────────────────────────────────────────────────────────

class TestPreambleHandling:
    def test_preamble_text_before_first_header_is_included(self):
        chunks = chunk_prose_file("README.md", PREAMBLE_MD)
        full = "\n".join(c.content for c in chunks)
        assert "This text appears before any header" in full

    def test_preamble_chunk_has_source_label(self):
        chunks = chunk_prose_file("README.md", PREAMBLE_MD)
        preamble_chunks = [c for c in chunks
                           if "This text appears before any header" in c.content]
        assert preamble_chunks
        assert all(c.content.startswith("[Source:") for c in preamble_chunks)


# ── Plain text (.txt) handling ────────────────────────────────────────────────

class TestPlainTextHandling:
    def test_plain_text_produces_chunks(self):
        chunks = chunk_prose_file("notes.txt", PLAIN_TEXT)
        assert len(chunks) >= 1

    def test_plain_text_content_present(self):
        chunks = chunk_prose_file("notes.txt", PLAIN_TEXT)
        full = "\n".join(c.content for c in chunks)
        assert "plain text file" in full

    def test_plain_text_within_token_cap(self):
        chunks = chunk_prose_file("notes.txt", PLAIN_TEXT)
        assert all(c.token_count <= MAX_TOKENS for c in chunks)

    def test_large_plain_text_split_at_paragraph_boundaries(self):
        # Build a large plain text with clear paragraph breaks
        big = "\n\n".join(["word " * 60] * 20)
        chunks = chunk_prose_file("big.txt", big)
        assert len(chunks) > 1
        assert all(c.token_count <= MAX_TOKENS for c in chunks)


# ── Empty / whitespace files ──────────────────────────────────────────────────

class TestEmptyFiles:
    def test_empty_string_returns_no_chunks(self):
        assert chunk_prose_file("empty.md", EMPTY_MD) == []

    def test_whitespace_only_returns_no_chunks(self):
        assert chunk_prose_file("empty.md", WHITESPACE_MD) == []

    def test_header_with_no_body_produces_no_chunk(self):
        # A header with no body text should not emit a chunk
        src = "# Title With No Body\n\n## Also Empty\n"
        chunks = chunk_prose_file("empty_headers.md", src)
        assert chunks == []


# ── Overlap ───────────────────────────────────────────────────────────────────

class TestOverlap:
    def test_overlap_applied_on_large_section(self):
        # Create a section large enough to be split into 2+ chunks
        para = "word " * 90  # ~90 tokens per paragraph
        src = "# Big\n\n" + "\n\n".join([para] * 12)
        chunks = chunk_prose_file("big.md", src)
        assert len(chunks) >= 2
        # The second chunk should contain words from the end of the first
        first_words = set(chunks[0].content.split()[-20:])
        second_words = set(chunks[1].content.split()[:30])
        overlap = first_words & second_words
        assert overlap, "Expected overlap tokens between consecutive chunks"

    def test_tail_tokens_helper_returns_last_n_tokens(self):
        text = "alpha beta gamma delta epsilon"
        tail = _tail_tokens(text, 10)
        assert "epsilon" in tail
        assert len(tail.split()) <= len(text.split())

    def test_tail_tokens_empty_text(self):
        assert _tail_tokens("", 100) == ""


# ── Section parsing ───────────────────────────────────────────────────────────

class TestSectionParsing:
    def test_parse_sections_returns_level_title_body_tuples(self):
        sections = _parse_sections("# Title\n\nBody text.\n")
        assert len(sections) == 1
        level, title, body = sections[0]
        assert level == 1
        assert title == "Title"
        assert "Body text" in body

    def test_no_headers_returns_single_section_with_level_zero(self):
        sections = _parse_sections("Just plain text.")
        assert len(sections) == 1
        assert sections[0][0] == 0
        assert sections[0][1] == ""

    def test_preamble_gets_level_zero(self):
        sections = _parse_sections("Preamble text.\n\n# Header\n\nBody.\n")
        assert sections[0][0] == 0
        assert "Preamble" in sections[0][2]

    def test_multiple_headers_parsed_correctly(self):
        sections = _parse_sections(FLAT_MD)
        titles = [s[1] for s in sections]
        assert "Section One" in titles
        assert "Section Two" in titles
        assert "Section Three" in titles

    def test_header_level_detected_correctly(self):
        src = "# H1\n\nbody1\n\n## H2\n\nbody2\n\n### H3\n\nbody3\n"
        sections = _parse_sections(src)
        levels = [s[0] for s in sections]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels


# ── Qdrant payload schema fields ─────────────────────────────────────────────

class TestQdrantPayloadFields:
    def test_file_type_is_prose_doc(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.file_type == "prose_doc"

    def test_chunk_strategy_is_markdown_header(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.chunk_strategy == "markdown_header"

    def test_file_path_preserved(self):
        chunks = chunk_prose_file("docs/guide.md", SIMPLE_MD)
        for c in chunks:
            assert c.file_path == "docs/guide.md"

    def test_commit_hash_defaults_empty(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.commit_hash == ""

    def test_target_module_defaults_none(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.target_module is None

    def test_token_count_positive(self):
        chunks = chunk_prose_file("README.md", SIMPLE_MD)
        for c in chunks:
            assert c.token_count > 0

    def test_chunk_dataclass_token_count_computed_on_init(self):
        c = Chunk(file_path="README.md", content="# Hello\n\nContent.")
        assert c.token_count > 0
        assert c.token_count == _count(c.content)
