"""
Tests for ingestion/config_chunker.py — config/manifest chunker.

Run with: python3 -m pytest ingestion/tests/test_config_chunker.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import pytest
from ingestion.config_chunker import (
    chunk_config_file, MAX_TOKENS, TARGET_MAX, MASSIVE_LINES,
    _flatten_to_sentences, _parse,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_JSON = json.dumps({
    "debug": True,
    "port": 8080,
    "host": "localhost",
})

NESTED_JSON = json.dumps({
    "server": {"host": "0.0.0.0", "port": 8080},
    "database": {"url": "postgres://localhost/db", "pool_size": 5},
    "feature_flags": {"enable_cache": True, "enable_metrics": False},
})

SIMPLE_TOML = """\
[server]
host = "0.0.0.0"
port = 8080

[database]
url = "postgres://localhost/db"
pool_size = 5
"""

SIMPLE_YAML = """\
name: my-app
version: 1.0.0
debug: false
dependencies:
  - requests
  - fastapi
  - uvicorn
"""

BROKEN_JSON = '{"key": "value", "unclosed": '
EMPTY_JSON  = "{}"


# ── Config label ──────────────────────────────────────────────────────────────

class TestConfigLabel:
    def test_every_chunk_starts_with_config_label(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        for c in chunks:
            assert c.content.startswith("[Config:"), (
                f"Chunk missing [Config:] label:\n{c.content[:80]}"
            )

    def test_config_label_contains_file_path(self):
        chunks = chunk_config_file("config/settings.json", SIMPLE_JSON)
        for c in chunks:
            assert "config/settings.json" in c.content.splitlines()[0]


# ── Declarative sentence flattening ──────────────────────────────────────────

class TestDeclarativeSentences:
    def test_boolean_true_becomes_enabled(self):
        sentences = _flatten_to_sentences({"debug": True})
        assert any("enabled" in s for s in sentences)

    def test_boolean_false_becomes_disabled(self):
        sentences = _flatten_to_sentences({"cache": False})
        assert any("disabled" in s for s in sentences)

    def test_scalar_value_in_sentence(self):
        sentences = _flatten_to_sentences({"port": 8080})
        assert any("8080" in s for s in sentences)

    def test_nested_dict_emits_section_sentence(self):
        sentences = _flatten_to_sentences({"database": {"host": "localhost"}})
        assert any("database" in s.lower() and "section" in s.lower() for s in sentences)

    def test_nested_values_flattened_with_dotted_key(self):
        sentences = _flatten_to_sentences({"database": {"host": "localhost"}})
        assert any("database.host" in s.lower() for s in sentences)

    def test_list_of_scalars_summarised_inline(self):
        sentences = _flatten_to_sentences({"deps": ["requests", "fastapi"]})
        joined = " ".join(sentences)
        assert "requests" in joined and "fastapi" in joined

    def test_list_of_objects_emits_count_sentence(self):
        sentences = _flatten_to_sentences({"items": [{"id": 1}, {"id": 2}]})
        assert any("2" in s and "item" in s.lower() for s in sentences)

    def test_none_value_represented(self):
        sentences = _flatten_to_sentences({"key": None})
        assert any("not set" in s for s in sentences)


# ── JSON ──────────────────────────────────────────────────────────────────────

class TestJsonFiles:
    def test_simple_json_produces_chunk(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        assert len(chunks) >= 1

    def test_nested_json_all_keys_present(self):
        chunks = chunk_config_file("settings.json", NESTED_JSON)
        full = "\n".join(c.content for c in chunks)
        assert "server" in full.lower()
        assert "database" in full.lower()
        assert "8080" in full

    def test_json_all_chunks_within_token_limit(self):
        chunks = chunk_config_file("settings.json", NESTED_JSON)
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over


# ── TOML ──────────────────────────────────────────────────────────────────────

class TestTomlFiles:
    def test_toml_produces_chunk(self):
        chunks = chunk_config_file("pyproject.toml", SIMPLE_TOML)
        assert len(chunks) >= 1

    def test_toml_values_in_chunk(self):
        chunks = chunk_config_file("pyproject.toml", SIMPLE_TOML)
        full = "\n".join(c.content for c in chunks)
        assert "8080" in full
        assert "pool" in full.lower()

    def test_toml_chunks_within_token_limit(self):
        chunks = chunk_config_file("pyproject.toml", SIMPLE_TOML)
        assert all(c.token_count <= MAX_TOKENS for c in chunks)


# ── YAML ──────────────────────────────────────────────────────────────────────

class TestYamlFiles:
    def test_yaml_produces_chunk(self):
        chunks = chunk_config_file("config.yaml", SIMPLE_YAML)
        assert len(chunks) >= 1

    def test_yaml_values_present(self):
        chunks = chunk_config_file("config.yaml", SIMPLE_YAML)
        full = "\n".join(c.content for c in chunks)
        assert "my-app" in full
        assert "1.0.0" in full

    def test_yaml_chunks_within_token_limit(self):
        chunks = chunk_config_file("config.yaml", SIMPLE_YAML)
        assert all(c.token_count <= MAX_TOKENS for c in chunks)


# ── YAML 1.1 boolean key coercion ────────────────────────────────────────────

class TestYaml11BooleanKeys:
    def test_on_key_does_not_crash(self):
        # PyYAML parses bare 'on:' as Python True — str(key) must be called
        src = "on:\n  push:\n    branches: [main]\n"
        chunks = chunk_config_file(".github/workflows/ci.yml", src)
        assert isinstance(chunks, list)

    def test_yes_key_does_not_crash(self):
        src = "yes: enabled\nno: disabled\n"
        chunks = chunk_config_file("flags.yaml", src)
        assert isinstance(chunks, list)

    def test_github_actions_yaml_produces_chunks(self):
        src = (
            "on:\n  pull_request:\n    types: [opened]\n"
            "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        )
        chunks = chunk_config_file(".github/workflows/pr.yml", src)
        assert len(chunks) >= 1


# ── Massive-file guardrail ────────────────────────────────────────────────────

class TestMassiveFileGuardrail:
    def _make_massive_src(self) -> str:
        src = "\n".join([f'  "k{i}": "v{i}",' for i in range(MASSIVE_LINES + 10)])
        return "{\n" + src + '\n  "last": "val"\n}'

    def test_massive_file_returns_exactly_one_chunk(self):
        chunks = chunk_config_file("huge.json", self._make_massive_src())
        assert len(chunks) == 1

    def test_massive_chunk_mentions_not_fully_translated(self):
        chunks = chunk_config_file("huge.json", self._make_massive_src())
        assert "not fully translated" in chunks[0].content.lower()

    def test_massive_chunk_within_token_limit(self):
        chunks = chunk_config_file("huge.json", self._make_massive_src())
        assert chunks[0].token_count <= MAX_TOKENS


# ── Parse failure fallback ────────────────────────────────────────────────────

class TestParseFailureFallback:
    def test_broken_json_does_not_crash(self):
        chunks = chunk_config_file("bad.json", BROKEN_JSON)
        assert isinstance(chunks, list)

    def test_broken_json_produces_chunk(self):
        chunks = chunk_config_file("bad.json", BROKEN_JSON)
        assert len(chunks) >= 1

    def test_broken_json_chunk_within_token_limit(self):
        chunks = chunk_config_file("bad.json", BROKEN_JSON)
        assert all(c.token_count <= MAX_TOKENS for c in chunks)

    def test_broken_json_chunk_has_config_label(self):
        chunks = chunk_config_file("bad.json", BROKEN_JSON)
        assert chunks[0].content.startswith("[Config:")


# ── Empty file ────────────────────────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_string_returns_no_chunks(self):
        assert chunk_config_file("empty.json", "") == []

    def test_empty_object_returns_no_chunks(self):
        assert chunk_config_file("empty.json", EMPTY_JSON) == []

    def test_whitespace_only_returns_no_chunks(self):
        assert chunk_config_file("empty.yaml", "   \n\n") == []


# ── Token cap and zero overlap ────────────────────────────────────────────────

class TestTokenCapAndOverlap:
    def _make_large_config(self, n_keys: int = 300) -> str:
        obj = {f"setting_key_{i}": f"some_value_for_setting_{i}" for i in range(n_keys)}
        return json.dumps(obj)

    def test_large_config_all_within_cap(self):
        chunks = chunk_config_file("big.json", self._make_large_config(300))
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over, f"{len(over)} chunks exceed {MAX_TOKENS} tokens"

    def test_large_config_is_split(self):
        chunks = chunk_config_file("big.json", self._make_large_config(300))
        assert len(chunks) > 1

    def test_zero_overlap_no_sentence_duplicated(self):
        chunks = chunk_config_file("big.json", self._make_large_config(300))
        seen: set[str] = set()
        for c in chunks:
            for line in c.content.splitlines()[1:]:  # skip [Config:] header
                if line.strip():
                    assert line not in seen, f"Sentence duplicated: {line!r}"
                    seen.add(line)


# ── Oversized single sentence fallback ───────────────────────────────────────

class TestOversizedSentenceFallback:
    def test_very_long_value_does_not_exceed_token_cap(self):
        # A single value that would produce a sentence > 800 tokens
        long_val = "word " * 1000
        src = json.dumps({"description": long_val})
        chunks = chunk_config_file("settings.json", src)
        assert all(c.token_count <= MAX_TOKENS for c in chunks)

    def test_truncated_marker_present_on_oversized_sentence(self):
        long_val = "word " * 1000
        src = json.dumps({"description": long_val})
        chunks = chunk_config_file("settings.json", src)
        full = "\n".join(c.content for c in chunks)
        assert "[truncated]" in full


# ── Qdrant payload schema fields ─────────────────────────────────────────────

class TestQdrantPayloadFields:
    def test_file_type_is_config(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        for c in chunks:
            assert c.file_type == "config"

    def test_chunk_strategy_is_llm_translation(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        for c in chunks:
            assert c.chunk_strategy == "llm_translation"

    def test_file_path_preserved(self):
        chunks = chunk_config_file("config/app.toml", SIMPLE_TOML)
        for c in chunks:
            assert c.file_path == "config/app.toml"

    def test_commit_hash_defaults_empty(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        for c in chunks:
            assert c.commit_hash == ""

    def test_token_count_positive(self):
        chunks = chunk_config_file("settings.json", SIMPLE_JSON)
        for c in chunks:
            assert c.token_count > 0
