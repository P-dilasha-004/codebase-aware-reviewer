"""
Tests for Task 4.2 — _process_sync delete-then-upsert logic.

Covers gaps identified in review of the Task 4.1 stub:
- _process_sync actual logic (end-to-end delete -> fetch -> chunk -> embed -> upsert)
- repo_id-scoped delete (critical: must never touch another tenant's vectors)
- commit_hash stamping on upserted points
- Idempotency on duplicate/redelivered merge events
- Empty changed_files
- Malformed payload fields (missing repo_id, non-dict entries, missing path,
  missing commit_hash)

A lightweight in-memory FakeQdrantClient stands in for real Qdrant so delete
and upsert semantics are exercised for real (not just call-counted), which is
what makes the idempotency and repo_id-scoping tests meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from api.diff import FetchResult

SECRET = "test-webhook-secret-abc123"


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    import importlib
    import api.main as m
    importlib.reload(m)
    monkeypatch.setattr("api.main._WEBHOOK_SECRET", SECRET)


# ── Test doubles ──────────────────────────────────────────────────────────────

@dataclass
class FakeChunk:
    file_path: str
    content: str
    file_type: str = "source_code"
    chunk_strategy: str = "ast_function"
    target_module: str | None = None


class FakeQdrantClient:
    """
    In-memory stand-in exercising the real FilterSelector/PointStruct shapes
    that _delete_file_vectors / _upsert_batch build, so delete-before-upsert
    semantics are actually validated rather than just mock-call-counted.
    """

    def __init__(self):
        self.points: dict[str, dict] = {}   # point_id -> payload
        self.delete_calls: list[dict] = []  # one dict of matched field->value per call
        self.upsert_calls: list[list] = []  # one list of PointStruct per call

    def delete(self, collection_name, points_selector):
        conditions = {
            cond.key: cond.match.value for cond in points_selector.filter.must
        }
        self.delete_calls.append(conditions)
        to_remove = [
            pid for pid, payload in self.points.items()
            if all(payload.get(k) == v for k, v in conditions.items())
        ]
        for pid in to_remove:
            del self.points[pid]

    def upsert(self, collection_name, points):
        self.upsert_calls.append(list(points))
        for p in points:
            self.points[p.id] = p.payload

    def seed(self, point_id: str, payload: dict) -> None:
        self.points[point_id] = payload


def _fake_chunk_file(path: str, content: str) -> list[FakeChunk]:
    return [FakeChunk(file_path=path, content=content)]


def _fake_embed_chunks(chunks, embedder) -> list[list[float]]:
    return [[0.1, 0.2, 0.3] for _ in chunks]


BASE_PAYLOAD = {
    "repo_id": "owner/repo",
    "commit_hash": "sha-new",
    "changed_files": [
        {
            "path": "src/auth.py",
            "status": "modified",
            "raw_url": "https://raw.githubusercontent.com/owner/repo/sha-new/src/auth.py",
        }
    ],
}


def _patched(fake_client, fetch_result):
    """Context manager stack patching every external boundary of _process_sync."""
    return (
        patch("api.main._get_qdrant", return_value=fake_client),
        patch("api.main._get_embedder", return_value=object()),
        patch("api.main._chunk_file", side_effect=_fake_chunk_file),
        patch("api.main._embed_chunks", side_effect=_fake_embed_chunks),
        patch("api.main.extract_and_fetch", new_callable=AsyncMock, return_value=fetch_result),
    )


async def _run_sync(payload, fake_client, fetch_result):
    import api.main as m
    patches = _patched(fake_client, fetch_result)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await m._process_sync(payload)


# ── 1. _process_sync actual logic ────────────────────────────────────────────

class TestActualSyncLogic:
    @pytest.mark.asyncio
    async def test_full_flow_deletes_then_upserts(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        assert len(client.delete_calls) == 1
        assert len(client.upsert_calls) == 1
        assert len(client.points) == 1

    @pytest.mark.asyncio
    async def test_delete_happens_before_upsert(self):
        client = FakeQdrantClient()
        # Pre-seed a stale point for this exact path.
        client.seed("stale-id", {"repo_id": "owner/repo", "file_path": "src/auth.py",
                                  "commit_hash": "sha-old"})
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        # Stale point is gone; only the freshly upserted one remains.
        assert "stale-id" not in client.points
        assert len(client.points) == 1

    @pytest.mark.asyncio
    async def test_removed_file_only_deleted_never_upserted(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                {"path": "src/gone.py", "status": "removed"},
            ],
        }
        # extract_and_fetch filters "removed" entries out of .files by design.
        fetch_result = FetchResult(files={})

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == [{"repo_id": "owner/repo", "file_path": "src/gone.py"}]
        assert client.upsert_calls == []
        assert client.points == {}

    @pytest.mark.asyncio
    async def test_fallback_result_skips_upsert_but_delete_already_ran(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(fallback=True)

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        assert len(client.delete_calls) == 1   # deletion is unconditional
        assert client.upsert_calls == []       # fallback aborts before fetch/upsert


# ── Renamed files (planning.md Phase 4 exit criterion: "Renames correctly swap paths") ──

class TestRenamedFiles:
    @pytest.mark.asyncio
    async def test_rename_deletes_old_path_and_upserts_new_path(self):
        client = FakeQdrantClient()
        client.seed("old-point", {"repo_id": "owner/repo", "file_path": "src/old_name.py",
                                   "commit_hash": "sha-old"})
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                {"path": "src/new_name.py", "previous_path": "src/old_name.py",
                 "status": "renamed", "raw_url": "https://x/new_name.py"},
            ],
        }
        fetch_result = FetchResult(files={"src/new_name.py": "def foo(): pass"})

        await _run_sync(payload, client, fetch_result)

        assert {"repo_id": "owner/repo", "file_path": "src/old_name.py"} in client.delete_calls
        assert {"repo_id": "owner/repo", "file_path": "src/new_name.py"} in client.delete_calls
        assert "old-point" not in client.points
        remaining = list(client.points.values())
        assert len(remaining) == 1
        assert remaining[0]["file_path"] == "src/new_name.py"

    @pytest.mark.asyncio
    async def test_rename_missing_previous_path_does_not_crash(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                {"path": "src/new_name.py", "status": "renamed",
                 "raw_url": "https://x/new_name.py"},
            ],
        }
        fetch_result = FetchResult(files={"src/new_name.py": "def foo(): pass"})

        await _run_sync(payload, client, fetch_result)

        # Only the new path is deleted/upserted; no crash from the missing previous_path.
        assert client.delete_calls == [{"repo_id": "owner/repo", "file_path": "src/new_name.py"}]
        assert len(client.points) == 1

    @pytest.mark.asyncio
    async def test_non_renamed_entries_do_not_trigger_previous_path_lookup(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                {"path": "src/auth.py", "status": "modified", "raw_url": "https://x/auth.py"},
            ],
        }
        fetch_result = FetchResult(files={"src/auth.py": "content"})

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == [{"repo_id": "owner/repo", "file_path": "src/auth.py"}]


# ── 2. repo_id-scoped delete (critical) ──────────────────────────────────────

class TestRepoIdScopedDelete:
    @pytest.mark.asyncio
    async def test_delete_filter_includes_repo_id_and_path(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        assert client.delete_calls == [{"repo_id": "owner/repo", "file_path": "src/auth.py"}]

    @pytest.mark.asyncio
    async def test_delete_never_matches_other_tenants_vectors(self):
        client = FakeQdrantClient()
        # A point at the same file_path but a *different* repo_id — must survive.
        client.seed("other-tenant", {"repo_id": "someone-else/repo", "file_path": "src/auth.py"})
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        assert "other-tenant" in client.points
        assert client.points["other-tenant"]["repo_id"] == "someone-else/repo"

    @pytest.mark.asyncio
    async def test_multiple_changed_files_each_scoped_individually(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                {"path": "src/a.py", "status": "modified", "raw_url": "https://x/a.py"},
                {"path": "src/b.py", "status": "added", "raw_url": "https://x/b.py"},
            ],
        }
        fetch_result = FetchResult(files={"src/a.py": "a", "src/b.py": "b"})

        await _run_sync(payload, client, fetch_result)

        assert {"repo_id": "owner/repo", "file_path": "src/a.py"} in client.delete_calls
        assert {"repo_id": "owner/repo", "file_path": "src/b.py"} in client.delete_calls
        assert len(client.delete_calls) == 2


# ── 3. commit_hash stamping ───────────────────────────────────────────────────

class TestCommitHashStamping:
    @pytest.mark.asyncio
    async def test_upserted_point_stamped_with_new_commit_hash(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        payloads = list(client.points.values())
        assert len(payloads) == 1
        assert payloads[0]["commit_hash"] == "sha-new"

    @pytest.mark.asyncio
    async def test_stale_commit_hash_replaced_not_merged(self):
        client = FakeQdrantClient()
        client.seed("stale-id", {"repo_id": "owner/repo", "file_path": "src/auth.py",
                                  "commit_hash": "sha-old"})
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)

        commit_hashes = {p["commit_hash"] for p in client.points.values()}
        assert commit_hashes == {"sha-new"}


# ── 4. Idempotency on duplicate events ───────────────────────────────────────

class TestIdempotency:
    @pytest.mark.asyncio
    async def test_processing_same_event_twice_does_not_duplicate_vectors(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(BASE_PAYLOAD, client, fetch_result)
        first_run_count = len(client.points)

        await _run_sync(BASE_PAYLOAD, client, fetch_result)
        second_run_count = len(client.points)

        assert first_run_count == 1
        assert second_run_count == 1  # not 2 — delete-before-upsert prevents accumulation

    @pytest.mark.asyncio
    async def test_processing_same_event_three_times_stays_stable(self):
        client = FakeQdrantClient()
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        for _ in range(3):
            await _run_sync(BASE_PAYLOAD, client, fetch_result)

        assert len(client.points) == 1


# ── 5. Empty changed_files ────────────────────────────────────────────────────

class TestEmptyChangedFiles:
    @pytest.mark.asyncio
    async def test_empty_list_short_circuits_before_delete(self):
        client = FakeQdrantClient()
        payload = {**BASE_PAYLOAD, "changed_files": []}
        fetch_result = FetchResult()

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock) as mock_fetch:
            await _run_sync(payload, client, fetch_result)
            mock_fetch.assert_not_called()

        assert client.delete_calls == []
        assert client.upsert_calls == []

    @pytest.mark.asyncio
    async def test_missing_changed_files_key_treated_as_empty(self):
        client = FakeQdrantClient()
        payload = {"repo_id": "owner/repo", "commit_hash": "sha-new"}
        fetch_result = FetchResult()

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock) as mock_fetch:
            await _run_sync(payload, client, fetch_result)
            mock_fetch.assert_not_called()

        assert client.delete_calls == []

    @pytest.mark.asyncio
    async def test_null_changed_files_treated_as_empty(self):
        client = FakeQdrantClient()
        payload = {"repo_id": "owner/repo", "commit_hash": "sha-new", "changed_files": None}
        fetch_result = FetchResult()

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == []
        assert client.upsert_calls == []


# ── 6. Malformed payload fields ───────────────────────────────────────────────

class TestMalformedPayloadFields:
    @pytest.mark.asyncio
    async def test_missing_repo_id_aborts_before_any_qdrant_call(self):
        client = FakeQdrantClient()
        payload = {"commit_hash": "sha-new", "changed_files": [{"path": "src/a.py"}]}
        fetch_result = FetchResult()

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock) as mock_fetch:
            await _run_sync(payload, client, fetch_result)
            mock_fetch.assert_not_called()

        assert client.delete_calls == []
        assert client.upsert_calls == []

    @pytest.mark.asyncio
    async def test_empty_string_repo_id_aborts(self):
        client = FakeQdrantClient()
        payload = {"repo_id": "", "commit_hash": "sha-new",
                   "changed_files": [{"path": "src/a.py"}]}
        fetch_result = FetchResult()

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == []

    @pytest.mark.asyncio
    async def test_non_dict_entries_in_changed_files_are_skipped(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": ["not-a-dict", 42, None],
        }
        fetch_result = FetchResult()

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == []  # nothing valid to delete
        assert client.upsert_calls == []

    @pytest.mark.asyncio
    async def test_entry_missing_path_is_skipped(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [{"status": "modified", "raw_url": "https://x/a.py"}],
        }
        fetch_result = FetchResult()

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == []

    @pytest.mark.asyncio
    async def test_mixed_valid_and_malformed_entries_processes_only_valid(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "commit_hash": "sha-new",
            "changed_files": [
                "not-a-dict",
                {"path": "src/good.py", "status": "modified", "raw_url": "https://x/good.py"},
                {"status": "modified"},  # missing path
            ],
        }
        fetch_result = FetchResult(files={"src/good.py": "content"})

        await _run_sync(payload, client, fetch_result)

        assert client.delete_calls == [{"repo_id": "owner/repo", "file_path": "src/good.py"}]

    @pytest.mark.asyncio
    async def test_missing_commit_hash_does_not_crash_and_stamps_empty(self):
        client = FakeQdrantClient()
        payload = {
            "repo_id": "owner/repo",
            "changed_files": [
                {"path": "src/auth.py", "status": "modified", "raw_url": "https://x/auth.py"},
            ],
        }
        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})

        await _run_sync(payload, client, fetch_result)

        payloads = list(client.points.values())
        assert len(payloads) == 1
        assert payloads[0]["commit_hash"] == ""
