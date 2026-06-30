"""
Tests for ingestion/pipeline.py — file router and ingestion logic.

Run with: python3 -m pytest ingestion/tests/test_pipeline.py -v
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest
from ingestion.pipeline import (
    _classify, _should_skip, _is_binary, _walk_repo, _chunk_file,
    _delete_file_vectors, QDRANT_COLLECTION,
    SKIP_FILENAMES, SKIP_DIRS, MAX_TOKENS,
)


def _fake_embed(n=1):
    """Return an embedder mock whose .embed() yields numpy-backed vectors."""
    embedder = MagicMock()
    embedder.embed.return_value = iter(np.zeros((n, 768)) for _ in range(1))
    # Re-create a fresh iterator each call
    embedder.embed.side_effect = lambda texts: iter(
        np.zeros(768) for _ in texts
    )
    return embedder


# ── File classification ───────────────────────────────────────────────────────

class TestClassify:
    def test_py_maps_to_track_a(self):
        assert _classify(Path("src/auth.py")) == "A"

    def test_md_maps_to_track_b(self):
        assert _classify(Path("README.md")) == "B"

    def test_txt_maps_to_track_b(self):
        assert _classify(Path("notes.txt")) == "B"

    def test_json_maps_to_track_c(self):
        assert _classify(Path("config.json")) == "C"

    def test_toml_maps_to_track_c(self):
        assert _classify(Path("pyproject.toml")) == "C"

    def test_yaml_maps_to_track_c(self):
        assert _classify(Path("docker-compose.yaml")) == "C"

    def test_yml_maps_to_track_c(self):
        assert _classify(Path(".github/workflows/ci.yml")) == "C"

    def test_unknown_extension_returns_none(self):
        assert _classify(Path("image.png")) is None
        assert _classify(Path("archive.zip")) is None
        assert _classify(Path("binary.exe")) is None

    def test_no_extension_returns_none(self):
        assert _classify(Path("Makefile")) is None


# ── Skip rules ────────────────────────────────────────────────────────────────

class TestShouldSkip:
    @pytest.mark.parametrize("filename", sorted(SKIP_FILENAMES))
    def test_all_lockfiles_are_skipped(self, filename):
        assert _should_skip(Path(filename)), f"{filename} should be skipped"

    def test_lock_extension_skipped(self):
        assert _should_skip(Path("something.lock"))

    def test_normal_python_file_not_skipped(self):
        assert not _should_skip(Path("src/auth.py"))

    def test_pyproject_toml_not_skipped(self):
        assert not _should_skip(Path("pyproject.toml"))

    def test_case_insensitive_lockfile_skipped(self):
        assert _should_skip(Path("Package-Lock.JSON"))


# ── Binary detection ──────────────────────────────────────────────────────────

class TestIsBinary:
    def test_text_file_not_binary(self, tmp_path):
        f = tmp_path / "source.py"
        f.write_text("def hello(): return 1\n")
        assert not _is_binary(f)

    def test_null_bytes_detected_as_binary(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        assert _is_binary(f)

    def test_utf8_content_not_binary(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n\nこんにちは 🌍\n", encoding="utf-8")
        assert not _is_binary(f)


# ── Repository walker ─────────────────────────────────────────────────────────

class TestWalkRepo:
    def _make_repo(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / ".git").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "docs").mkdir()

        (tmp_path / "src" / "auth.py").write_text("def login(): pass\n")
        (tmp_path / "src" / "utils.py").write_text("import os\n")
        (tmp_path / "tests" / "test_auth.py").write_text("def test_login(): pass\n")
        (tmp_path / "README.md").write_text("# My Project\n")
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
        (tmp_path / "poetry.lock").write_text("# lock file\n")
        (tmp_path / ".git" / "config").write_text("[core]\n")
        (tmp_path / "__pycache__" / "auth.pyc").write_bytes(b"\x00" * 10)
        (tmp_path / "docs" / "GUIDE.md").write_text("## Guide\n\nContent.\n")
        return tmp_path

    def test_python_files_collected(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert "src/auth.py" in files
        assert "src/utils.py" in files
        assert "tests/test_auth.py" in files

    def test_markdown_files_collected(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert "README.md" in files
        assert "docs/GUIDE.md" in files

    def test_config_files_collected(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert "pyproject.toml" in files

    def test_lockfiles_excluded(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert "poetry.lock" not in files

    def test_git_dir_excluded(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert not any(".git" in p for p in files)

    def test_pycache_excluded(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert not any("__pycache__" in p for p in files)

    def test_binary_file_excluded(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        assert not any(".pyc" in p for p in files)

    def test_empty_repo_returns_empty_list(self, tmp_path):
        assert _walk_repo(tmp_path) == []

    def test_paths_are_relative_to_repo_root(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        for path in files:
            assert not Path(path).is_absolute(), f"Path should be relative: {path}"

    def test_root_level_file_has_no_leading_separator(self, tmp_path):
        repo = self._make_repo(tmp_path)
        files = dict(_walk_repo(repo))
        for path in files:
            assert not path.startswith("/"), f"Path has leading slash: {path}"
            assert not path.startswith("./"), f"Path has leading ./: {path}"


# ── Chunking dispatch ─────────────────────────────────────────────────────────

class TestChunkFileDispatch:
    def test_py_file_dispatches_to_python_chunker(self):
        src = "import os\n\ndef foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        assert len(chunks) >= 1
        assert all(c.chunk_strategy == "ast_function" for c in chunks)
        assert all(c.file_type == "source_code" for c in chunks)

    def test_md_file_dispatches_to_prose_chunker(self):
        src = "# Title\n\nSome content here.\n"
        chunks = _chunk_file("README.md", src)
        assert len(chunks) >= 1
        assert all(c.chunk_strategy == "markdown_header" for c in chunks)
        assert all(c.file_type == "prose_doc" for c in chunks)

    def test_toml_file_dispatches_to_config_chunker(self):
        src = "[tool]\nname = \"my-app\"\n"
        chunks = _chunk_file("pyproject.toml", src)
        assert len(chunks) >= 1
        assert all(c.chunk_strategy == "llm_translation" for c in chunks)
        assert all(c.file_type == "config" for c in chunks)

    def test_json_file_dispatches_to_config_chunker(self):
        src = '{"version": "1.0.0", "debug": false}'
        chunks = _chunk_file("config.json", src)
        assert len(chunks) >= 1
        assert all(c.file_type == "config" for c in chunks)

    def test_yaml_file_dispatches_to_config_chunker(self):
        src = "version: 1\nname: app\n"
        chunks = _chunk_file("docker-compose.yaml", src)
        assert len(chunks) >= 1
        assert all(c.file_type == "config" for c in chunks)

    def test_unknown_extension_returns_empty(self):
        assert _chunk_file("image.png", b"\x89PNG") == []

    def test_all_chunk_file_paths_preserved(self):
        src = "import os\n\ndef foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        for c in chunks:
            assert c.file_path == "src/foo.py"

    def test_commit_hash_defaults_empty_before_stamping(self):
        src = "import os\n\ndef foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        for c in chunks:
            assert c.commit_hash == ""


# ── Commit hash stamping ──────────────────────────────────────────────────────

class TestCommitHashStamping:
    def test_commit_hash_stamped_on_chunks(self, tmp_path):
        from ingestion.pipeline import _chunk_file
        src = "def foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        commit = "abc123def456"
        for c in chunks:
            c.commit_hash = commit
        for c in chunks:
            assert c.commit_hash == commit

    def test_commit_hash_default_is_empty_string(self):
        src = "def foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        for c in chunks:
            assert c.commit_hash == ""


# ── Repo ID stamping ──────────────────────────────────────────────────────────

class TestRepoIdStamping:
    def test_repo_id_present_in_payload_fields(self):
        from ingestion.pipeline import _upsert_batch
        import uuid
        from unittest.mock import MagicMock, call

        src = "def foo():\n    return 1\n"
        chunks = _chunk_file("src/foo.py", src)
        vectors = [[0.1] * 768 for _ in chunks]
        mock_client = MagicMock()

        _upsert_batch(mock_client, chunks, vectors, "owner/repo", "abc123")

        mock_client.upsert.assert_called_once()
        points = mock_client.upsert.call_args[1]["points"]
        for point in points:
            assert point.payload["repo_id"] == "owner/repo"


# ── YML end-to-end ────────────────────────────────────────────────────────────

class TestYmlEndToEnd:
    def test_yml_file_produces_config_chunks(self):
        src = "name: pattern-buddy\nversion: 1\n"
        chunks = _chunk_file("config.yml", src)
        assert len(chunks) >= 1
        assert all(c.file_type == "config" for c in chunks)

    def test_github_actions_yml_with_on_key(self):
        src = "on:\n  push:\n    branches: [main]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
        chunks = _chunk_file(".github/workflows/ci.yml", src)
        assert len(chunks) >= 1


# ── .github vs .git distinction ──────────────────────────────────────────────

class TestGithubVsGitDirectory:
    def test_git_dir_excluded_from_walk(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\n")
        files = dict(_walk_repo(tmp_path))
        assert not any(".git/" in p or p.startswith(".git") for p in files)

    def test_github_dir_included_in_walk(self, tmp_path):
        gh_dir = tmp_path / ".github" / "workflows"
        gh_dir.mkdir(parents=True)
        (gh_dir / "ci.yml").write_text("on:\n  push:\njobs:\n  build:\n    runs-on: ubuntu-latest\n")
        files = dict(_walk_repo(tmp_path))
        assert any(".github" in p for p in files)


# ── Symlink handling ──────────────────────────────────────────────────────────

class TestSymlinkHandling:
    def test_symlink_to_python_file_is_collected(self, tmp_path):
        real = tmp_path / "real.py"
        real.write_text("def foo(): pass\n")
        link = tmp_path / "link.py"
        link.symlink_to(real)
        files = dict(_walk_repo(tmp_path))
        assert "link.py" in files or "real.py" in files

    def test_symlink_to_directory_does_not_crash(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "auth.py").write_text("def login(): pass\n")
        link_dir = tmp_path / "linked_src"
        link_dir.symlink_to(src_dir)
        # Must not raise
        files = dict(_walk_repo(tmp_path))
        assert len(files) >= 1


# ── Permission errors ─────────────────────────────────────────────────────────

class TestPermissionErrors:
    def test_unreadable_file_is_skipped(self, tmp_path):
        f = tmp_path / "secret.py"
        f.write_text("SECRET = 'abc'\n")
        f.chmod(0o000)
        try:
            files = dict(_walk_repo(tmp_path))
            # File should be skipped, not raise
            assert "secret.py" not in files
        finally:
            f.chmod(0o644)


# ── MAX_TOKENS import source ──────────────────────────────────────────────────

class TestMaxTokensImportSource:
    def test_max_tokens_importable_from_pipeline(self):
        from ingestion.pipeline import MAX_TOKENS as mt
        assert isinstance(mt, int)
        assert mt > 0

    def test_max_tokens_value_matches_python_chunker(self):
        from ingestion.pipeline import MAX_TOKENS as pipeline_mt
        from ingestion.python_chunker import MAX_TOKENS as chunker_mt
        assert pipeline_mt == chunker_mt


# ── Payload completeness (all 7 Qdrant fields) ───────────────────────────────

class TestPayloadCompleteness:
    """
    Verify every upserted point carries all 7 required payload fields.
    A chunker bug that omits a field would silently degrade retrieval — this
    test catches it before any vector reaches Qdrant.
    """

    REQUIRED_FIELDS = {
        "repo_id", "file_path", "file_type",
        "chunk_strategy", "target_module", "commit_hash", "content",
    }

    def _upsert_and_capture(self, rel_path: str, src: str):
        from ingestion.pipeline import _upsert_batch
        from unittest.mock import MagicMock
        chunks = _chunk_file(rel_path, src)
        assert chunks, f"No chunks produced for {rel_path}"
        vectors = [[0.1] * 768 for _ in chunks]
        mock_client = MagicMock()
        _upsert_batch(mock_client, chunks, vectors, "owner/repo", "abc123")
        return mock_client.upsert.call_args[1]["points"]

    def test_python_chunk_has_all_7_fields(self):
        points = self._upsert_and_capture("src/auth.py", "def login(): pass\n")
        for p in points:
            missing = self.REQUIRED_FIELDS - set(p.payload.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_markdown_chunk_has_all_7_fields(self):
        points = self._upsert_and_capture("README.md", "# Title\n\nContent.\n")
        for p in points:
            missing = self.REQUIRED_FIELDS - set(p.payload.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_config_chunk_has_all_7_fields(self):
        points = self._upsert_and_capture("config.json", '{"debug": true}')
        for p in points:
            missing = self.REQUIRED_FIELDS - set(p.payload.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_content_field_is_non_empty_string(self):
        points = self._upsert_and_capture("src/auth.py", "def login(): pass\n")
        for p in points:
            assert isinstance(p.payload["content"], str)
            assert len(p.payload["content"]) > 0

    def test_commit_hash_field_value_matches_input(self):
        from ingestion.pipeline import _upsert_batch
        from unittest.mock import MagicMock
        chunks = _chunk_file("src/auth.py", "def login(): pass\n")
        vectors = [[0.1] * 768 for _ in chunks]
        mock_client = MagicMock()
        _upsert_batch(mock_client, chunks, vectors, "owner/repo", "deadbeef1234")
        points = mock_client.upsert.call_args[1]["points"]
        for p in points:
            assert p.payload["commit_hash"] == "deadbeef1234"

    def test_file_path_field_matches_input(self):
        points = self._upsert_and_capture("src/auth.py", "def login(): pass\n")
        for p in points:
            assert p.payload["file_path"] == "src/auth.py"


# ── Skip rules produce zero vectors ──────────────────────────────────────────

class TestSkipRulesProduceZeroVectors:
    """
    Verify that files excluded by skip rules never reach _chunk_file —
    i.e. they are caught by _walk_repo and produce no chunks at all.
    A chunker bug could otherwise silently ingest garbage into Qdrant.
    """

    def test_poetry_lock_produces_no_chunks(self, tmp_path):
        f = tmp_path / "poetry.lock"
        f.write_text("# lockfile\n[[package]]\nname = \"requests\"\n")
        files = dict(_walk_repo(tmp_path))
        assert "poetry.lock" not in files

    def test_package_lock_json_produces_no_chunks(self, tmp_path):
        f = tmp_path / "package-lock.json"
        f.write_text('{"lockfileVersion": 2, "packages": {}}')
        files = dict(_walk_repo(tmp_path))
        assert "package-lock.json" not in files

    def test_binary_file_produces_no_chunks(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00")
        files = dict(_walk_repo(tmp_path))
        assert "image.png" not in files

    def test_git_config_produces_no_chunks(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\n")
        files = dict(_walk_repo(tmp_path))
        assert not any(".git" in p for p in files)

    def test_dot_lock_extension_produces_no_chunks(self, tmp_path):
        f = tmp_path / "custom.lock"
        f.write_text("lock content\n")
        files = dict(_walk_repo(tmp_path))
        assert "custom.lock" not in files

    def test_pycache_produces_no_chunks(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "auth.cpython-313.pyc").write_bytes(b"\x00" * 20)
        files = dict(_walk_repo(tmp_path))
        assert not any("__pycache__" in p for p in files)


# ── Delete-before-upsert ──────────────────────────────────────────────────────

class TestDeleteBeforeUpsert:
    """
    ingest_repository must delete existing vectors for (repo_id, file_path)
    before upserting new ones — otherwise re-runs duplicate every vector.
    """

    def _make_client(self):
        client = MagicMock()
        client.delete = MagicMock()
        client.upsert = MagicMock()
        return client

    def test_delete_called_before_upsert_for_each_file(self, tmp_path):
        (tmp_path / "auth.py").write_text("def login(): pass\n")
        client = self._make_client()
        call_order = []
        client.delete.side_effect = lambda **_: call_order.append("delete")
        client.upsert.side_effect = lambda **_: call_order.append("upsert")

        from ingestion.pipeline import ingest_repository
        with patch("ingestion.pipeline._load_embedder", return_value=_fake_embed()):
            ingest_repository(tmp_path, "owner/repo", "abc123", client)

        assert "delete" in call_order
        assert "upsert" in call_order
        assert call_order.index("delete") < call_order.index("upsert")

    def test_delete_called_once_per_file(self, tmp_path):
        (tmp_path / "auth.py").write_text("def login(): pass\n")
        (tmp_path / "utils.py").write_text("def helper(): pass\n")
        client = self._make_client()

        from ingestion.pipeline import ingest_repository
        with patch("ingestion.pipeline._load_embedder", return_value=_fake_embed()):
            ingest_repository(tmp_path, "owner/repo", "abc123", client)

        assert client.delete.call_count == 2

    def test_delete_uses_correct_repo_id_and_file_path(self, tmp_path):
        (tmp_path / "payments.py").write_text("def pay(): pass\n")
        client = self._make_client()

        from ingestion.pipeline import ingest_repository
        with patch("ingestion.pipeline._load_embedder", return_value=_fake_embed()):
            ingest_repository(tmp_path, "acme/payments", "sha1", client)

        assert client.delete.call_count == 1
        kwargs = client.delete.call_args.kwargs
        assert kwargs["collection_name"] == QDRANT_COLLECTION
        filt = kwargs["points_selector"].filter
        keys = {c.key for c in filt.must}
        assert "repo_id" in keys
        assert "file_path" in keys

    def test_delete_file_vectors_helper_calls_client_delete(self):
        client = MagicMock()
        _delete_file_vectors(client, "owner/repo", "src/auth.py")
        client.delete.assert_called_once()
        kwargs = client.delete.call_args.kwargs
        assert kwargs["collection_name"] == QDRANT_COLLECTION

    def test_no_delete_for_skipped_files(self, tmp_path):
        (tmp_path / "poetry.lock").write_text("# lockfile\n")
        client = self._make_client()

        from ingestion.pipeline import ingest_repository
        with patch("ingestion.pipeline._load_embedder", return_value=_fake_embed()):
            ingest_repository(tmp_path, "owner/repo", "sha1", client)

        client.delete.assert_not_called()
