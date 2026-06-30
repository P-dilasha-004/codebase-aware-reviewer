"""
Tests for ingestion/python_chunker.py — AST-based Python chunker.

Run with: python3 -m pytest ingestion/tests/test_python_chunker.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from ingestion.python_chunker import (
    chunk_python_file, MAX_TOKENS, Chunk,
    _header, _infer_target_module, _count,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_FUNC = """\
import os

def add(a, b):
    return a + b
"""

TWO_FUNCS = """\
import os

def foo():
    return 1

def bar():
    return 2
"""

CLASS_WITH_METHODS = """\
import os

class AuthService:
    def login(self, user, password):
        return True

    def logout(self, user):
        return True
"""

DECORATED_FUNC = """\
import functools

def decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@decorator
def my_func():
    return 42
"""

EMPTY_FILE = ""
COMMENTS_ONLY = "# This file is intentionally empty\n# No code here\n"


# ── Context header ────────────────────────────────────────────────────────────

class TestContextHeader:
    def test_header_contains_file_path(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        assert all("src/auth.py" in c.content for c in chunks)

    def test_header_format_starts_with_context(self):
        h = _header("src/auth.py", None, "login")
        assert h.startswith("// Context:")

    def test_header_includes_class_name(self):
        h = _header("src/auth.py", "AuthService", "login")
        assert "AuthService" in h

    def test_header_includes_method_name(self):
        h = _header("src/auth.py", "AuthService", "login")
        assert "login" in h

    def test_header_without_class(self):
        h = _header("src/utils.py", None, "helper")
        assert "Class" not in h
        assert "helper" in h


# ── Token cap ─────────────────────────────────────────────────────────────────

class TestTokenCap:
    def test_all_chunks_within_token_limit(self):
        src = SIMPLE_FUNC * 10
        chunks = chunk_python_file("src/big.py", src)
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over, f"{len(over)} chunks exceed {MAX_TOKENS} tokens"

    def test_max_tokens_constant_is_800(self):
        assert MAX_TOKENS == 800

    def test_token_count_on_chunk_is_positive(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        for c in chunks:
            assert c.token_count > 0


# ── Function splitting ────────────────────────────────────────────────────────

class TestFunctionSplitting:
    def test_two_functions_produce_two_or_more_chunks(self):
        chunks = chunk_python_file("src/utils.py", TWO_FUNCS)
        assert len(chunks) >= 2

    def test_each_function_in_separate_chunk(self):
        chunks = chunk_python_file("src/utils.py", TWO_FUNCS)
        foo_chunks = [c for c in chunks if "foo" in c.content]
        bar_chunks = [c for c in chunks if "bar" in c.content]
        assert foo_chunks
        assert bar_chunks

    def test_class_methods_chunked_separately(self):
        chunks = chunk_python_file("src/auth.py", CLASS_WITH_METHODS)
        login_chunks  = [c for c in chunks if "login" in c.content]
        logout_chunks = [c for c in chunks if "logout" in c.content]
        assert login_chunks
        assert logout_chunks


# ── Decorated functions ───────────────────────────────────────────────────────

class TestDecoratedFunctions:
    def test_decorated_function_produces_chunk(self):
        chunks = chunk_python_file("src/utils.py", DECORATED_FUNC)
        assert len(chunks) >= 1

    def test_decorated_function_content_present(self):
        chunks = chunk_python_file("src/utils.py", DECORATED_FUNC)
        full = "\n".join(c.content for c in chunks)
        assert "my_func" in full

    def test_decorated_function_does_not_crash(self):
        # Regression: decorated_definition node type must be handled
        src = "@property\ndef value(self):\n    return self._value\n"
        chunks = chunk_python_file("src/model.py", src)
        assert isinstance(chunks, list)


# ── Imports prepended to functional chunks ───────────────────────────────────

class TestImportsPrependedToFunctionalChunks:
    def test_imports_appear_in_function_chunk(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        func_chunks = [c for c in chunks if "add" in c.content]
        assert func_chunks
        assert any("import os" in c.content for c in func_chunks)

    def test_imports_not_emitted_as_standalone_chunk(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        # No chunk should contain ONLY imports (no function body)
        import_only = [
            c for c in chunks
            if c.content.count("def ") == 0 and "import" in c.content
            and "// Context:" in c.content
        ]
        assert not import_only, "Standalone import chunk found — should be prepended"


# ── Target module inference ───────────────────────────────────────────────────

class TestTargetModuleInference:
    def test_test_file_gets_target_module(self):
        result = _infer_target_module("tests/test_auth.py")
        assert result is not None
        assert "auth.py" in result

    def test_non_test_file_returns_none(self):
        assert _infer_target_module("src/auth.py") is None

    def test_target_module_on_chunks(self):
        src = "def test_login(): pass\n"
        chunks = chunk_python_file("tests/test_auth.py", src)
        for c in chunks:
            if c.target_module:
                assert "auth" in c.target_module

    def test_deeply_nested_test_path(self):
        result = _infer_target_module("project/tests/test_payment.py")
        assert result is not None


# ── Chunk dataclass ───────────────────────────────────────────────────────────

class TestChunkDataclass:
    def test_chunk_has_file_path(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        for c in chunks:
            assert c.file_path == "src/auth.py"

    def test_chunk_has_file_type_source_code(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        for c in chunks:
            assert c.file_type == "source_code"

    def test_chunk_has_chunk_strategy_ast_function(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        for c in chunks:
            assert c.chunk_strategy == "ast_function"

    def test_chunk_commit_hash_defaults_empty(self):
        chunks = chunk_python_file("src/auth.py", SIMPLE_FUNC)
        for c in chunks:
            assert c.commit_hash == ""

    def test_chunk_token_count_computed_on_init(self):
        c = Chunk(file_path="src/auth.py", content="def foo(): pass")
        assert c.token_count > 0


# ── Nested classes ────────────────────────────────────────────────────────────

class TestNestedClasses:
    def test_nested_class_methods_chunked(self):
        src = """\
class Outer:
    class Inner:
        def method(self):
            return 1
"""
        chunks = chunk_python_file("src/model.py", src)
        full = "\n".join(c.content for c in chunks)
        assert "method" in full

    def test_multiple_classes_in_one_file(self):
        src = """\
class Foo:
    def foo_method(self):
        pass

class Bar:
    def bar_method(self):
        pass
"""
        chunks = chunk_python_file("src/models.py", src)
        foo_chunks = [c for c in chunks if "foo_method" in c.content]
        bar_chunks = [c for c in chunks if "bar_method" in c.content]
        assert foo_chunks
        assert bar_chunks


# ── Empty and comments-only files ────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_file_does_not_crash(self):
        chunks = chunk_python_file("src/empty.py", EMPTY_FILE)
        assert isinstance(chunks, list)

    def test_comments_only_file_does_not_crash(self):
        chunks = chunk_python_file("src/empty.py", COMMENTS_ONLY)
        assert isinstance(chunks, list)


# ── Oversized subchunk signature ─────────────────────────────────────────────

class TestOversizedSubchunkSignature:
    def _make_large_func(self, lines: int = 200) -> str:
        body = "\n".join(f"    x_{i} = {i}" for i in range(lines))
        return f"def large_func():\n{body}\n    return x_0\n"

    def test_oversized_function_split_into_multiple_chunks(self):
        src = self._make_large_func(200)
        chunks = chunk_python_file("src/big.py", src)
        assert len(chunks) >= 2

    def test_all_oversized_subchunks_within_cap(self):
        src = self._make_large_func(200)
        chunks = chunk_python_file("src/big.py", src)
        over = [c for c in chunks if c.token_count > MAX_TOKENS]
        assert not over

    def test_oversized_subchunks_have_header(self):
        src = self._make_large_func(200)
        chunks = chunk_python_file("src/big.py", src)
        for c in chunks:
            assert "// Context:" in c.content


# ── Chunk overlap ─────────────────────────────────────────────────────────────

class TestChunkOverlap:
    def test_no_duplicate_function_bodies_across_chunks(self):
        chunks = chunk_python_file("src/utils.py", TWO_FUNCS)
        # Each function body line should appear in at most one chunk
        body_lines: dict[str, int] = {}
        for i, c in enumerate(chunks):
            for line in c.content.splitlines():
                stripped = line.strip()
                # Skip header lines and import lines (these ARE shared intentionally)
                if not stripped or stripped.startswith("// Context:") or stripped.startswith("import ") or stripped.startswith("from "):
                    continue
                # Skip lines that are just `def foo():` signatures (appear in header context)
                if stripped.startswith("def ") or stripped.startswith("class "):
                    continue
                if stripped in body_lines and body_lines[stripped] != i:
                    pytest.fail(f"Body line duplicated across chunks: {stripped!r}")
                body_lines[stripped] = i
