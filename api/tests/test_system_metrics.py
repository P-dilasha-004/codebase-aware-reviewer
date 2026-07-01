"""
Tests for Task 5.2 — system metrics collection (RAM utilization,
end_to_end_latency_ms, inference_latency_ms), planning.md section 9.5.

Covers:
- _current_ram_mb: normalizes ru_maxrss units by platform (KB on Linux,
  bytes on macOS) and returns a positive MB value
- _emit_system_metrics: JSON schema, appends JSONL to logs/system_metrics.json
- _stage_timer: exposes latency_ms on ctx after the block exits, so callers
  can reuse the measured value (e.g. for the end-to-end metrics record)
- _process_review: emits exactly one system_metrics record per run, with
  inference_latency_ms populated on the happy path
- _process_review: still emits a system_metrics record (status=failed,
  inference_latency_ms=None) when the pipeline fails before inference runs
- _process_review: end_to_end_latency_ms covers the whole pipeline, not
  just one stage
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from api.diff import FetchResult
from api.retrieval import PromptResult

REVIEW_PAYLOAD = {
    "repo_id": "owner/repo",
    "pr_number": 42,
    "base_sha": "aaa111",
    "head_sha": "bbb222",
    "changed_files": [
        {"path": "src/auth.py", "status": "modified",
         "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/auth.py"}
    ],
}


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "test-webhook-secret-abc123")
    import importlib
    import api.main as m
    importlib.reload(m)


# ── _current_ram_mb ───────────────────────────────────────────────────────────

class TestCurrentRamMb:
    def test_returns_positive_value(self):
        from api.main import _current_ram_mb
        assert _current_ram_mb() > 0

    def test_linux_units_are_kb_divided_by_1024(self, monkeypatch):
        import api.main as m
        monkeypatch.setattr("sys.platform", "linux")
        with patch("resource.getrusage") as mock_rusage:
            mock_rusage.return_value.ru_maxrss = 512_000  # KB on Linux
            assert m._current_ram_mb() == pytest.approx(500.0)

    def test_macos_units_are_bytes_divided_by_1024_squared(self, monkeypatch):
        import api.main as m
        monkeypatch.setattr("sys.platform", "darwin")
        with patch("resource.getrusage") as mock_rusage:
            mock_rusage.return_value.ru_maxrss = 500 * 1024 * 1024  # bytes on macOS
            assert m._current_ram_mb() == pytest.approx(500.0)


# ── _emit_system_metrics ──────────────────────────────────────────────────────

class TestEmitSystemMetrics:
    def test_writes_expected_schema(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)
        monkeypatch.setattr(m, "_current_ram_mb", lambda: 321.0)

        m._emit_system_metrics("trace-1", "success", 1234.567, 890.123)

        entry = json.loads((tmp_path / "system_metrics.json").read_text().strip())
        assert entry == {
            "trace_id": "trace-1",
            "status": "success",
            "end_to_end_latency_ms": 1234.57,
            "inference_latency_ms": 890.12,
            "ram_mb": 321.0,
        }

    def test_none_inference_latency_serializes_as_null(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_system_metrics("trace-1", "failed", 50.0, None)

        entry = json.loads((tmp_path / "system_metrics.json").read_text().strip())
        assert entry["inference_latency_ms"] is None
        assert entry["status"] == "failed"

    def test_appends_across_multiple_reviews(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_system_metrics("trace-1", "success", 100.0, 50.0)
        m._emit_system_metrics("trace-2", "success", 200.0, 60.0)

        lines = (tmp_path / "system_metrics.json").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_disk_write_failure_does_not_raise(self, monkeypatch):
        import api.main as m
        from pathlib import Path
        monkeypatch.setattr(m, "LOGS_DIR", Path("/nonexistent_root_only/logs"))
        m._emit_system_metrics("trace-1", "success", 1.0, 1.0)  # must not raise


# ── _stage_timer exposes latency_ms for reuse ────────────────────────────────

class TestStageTimerExposesLatency:
    def test_latency_ms_available_on_ctx_after_success(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with m._stage_timer("trace-1", "inference_handoff") as ctx:
            pass

        assert "latency_ms" in ctx
        assert ctx["latency_ms"] >= 0

    def test_latency_ms_available_on_ctx_after_failure(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with pytest.raises(ValueError):
            with m._stage_timer("trace-1", "inference_handoff") as ctx:
                raise ValueError("boom")

        assert "latency_ms" in ctx


# ── _process_review end-to-end metrics ───────────────────────────────────────

class TestProcessReviewSystemMetrics:
    @pytest.mark.asyncio
    async def test_happy_path_emits_one_metrics_record_with_inference_latency(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})
        prompt_result = PromptResult(prompt="PROMPT", chunks_used=1, chunks_retrieved=1)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock, return_value=fetch_result), \
             patch("api.main.retrieve_and_assemble", return_value=prompt_result), \
             patch("api.main.call_inference", new_callable=AsyncMock, return_value="REVIEW TEXT"), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):
            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-happy"})

        lines = (tmp_path / "system_metrics.json").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["trace_id"] == "trace-happy"
        assert entry["status"] == "success"
        assert entry["inference_latency_ms"] is not None
        assert entry["end_to_end_latency_ms"] >= entry["inference_latency_ms"]

    @pytest.mark.asyncio
    async def test_failure_before_inference_emits_metrics_with_null_inference_latency(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock,
                   side_effect=RuntimeError("fetch boom")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):
            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-fail"})

        entry = json.loads((tmp_path / "system_metrics.json").read_text().strip())
        assert entry["trace_id"] == "trace-fail"
        assert entry["status"] == "failed"
        assert entry["inference_latency_ms"] is None

    @pytest.mark.asyncio
    async def test_failure_after_inference_still_captures_inference_latency(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})
        prompt_result = PromptResult(prompt="PROMPT", chunks_used=1, chunks_retrieved=1)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock, return_value=fetch_result), \
             patch("api.main.retrieve_and_assemble", return_value=prompt_result), \
             patch("api.main.call_inference", new_callable=AsyncMock, return_value="REVIEW TEXT"), \
             patch("api.main.post_github_comment", new_callable=AsyncMock,
                   side_effect=[RuntimeError("github boom"), None]):
            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-partial"})

        entry = json.loads((tmp_path / "system_metrics.json").read_text().strip())
        assert entry["status"] == "failed"
        assert entry["inference_latency_ms"] is not None  # inference succeeded before the github call failed
