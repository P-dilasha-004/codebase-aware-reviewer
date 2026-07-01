"""
Phase 2 — Backend API Orchestrator.

Entry point for the Azure App Service.
Run locally: uvicorn api.main:app --reload
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from api.diff import extract_and_fetch
from api.retrieval import retrieve_and_assemble, build_macro_prompt
from api.inference import call_inference, post_github_comment, FALLBACK_COMMENT

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Pattern Buddy", version="2.0.0")

# ── Security ──────────────────────────────────────────────────────────────────

_WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")
_PAT:            str = os.environ.get("PAT", "")
_PHI_ENDPOINT:   str = os.environ.get("PHI_ENDPOINT", "")
_PHI_KEY:        str = os.environ.get("PHI_API_KEY", "")
_PHI_DEPLOYMENT: str = os.environ.get("PHI_DEPLOYMENT_NAME", "Phi-4-mini-reasoning")

def _get_qdrant() -> "QdrantClient":
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", ""),
        api_key=os.environ.get("QDRANT_API_KEY", ""),
    )

# Loaded once at startup — reloading the ONNX model per request causes OOM on App Service.
def _load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _load_embedder()
    return _embedder


def _verify_signature(body: bytes, signature_header: str | None) -> None:
    """
    Validate the GitHub HMAC-SHA256 webhook signature.

    GitHub sends: x-hub-signature-256: sha256=<hex>
    Raises HTTP 401 if missing or invalid.
    """
    if not _WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET not configured")

    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing or malformed signature")

    expected = "sha256=" + hmac.new(
        _WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Signature mismatch")


# ── Background task (stub — filled in by Task 2.2) ───────────────────────────

async def _process_review(payload: dict[str, Any]) -> None:
    """
    Full review pipeline — runs in the background after 202 is sent to GitHub.

    On any exception, posts a fallback PR comment so the developer is never
    silently left without feedback (closes the gap described in Task 2.1 review).
    """
    import asyncio

    repo_id   = payload.get("repo_id", "")
    pr_number = payload.get("pr_number", 0)

    logger.info(
        "review_started repo=%s pr=%s head=%s",
        repo_id, pr_number, payload.get("head_sha"),
    )

    try:
        # Task 2.2 — diff extraction + concurrent file fetch
        fetch_result = await extract_and_fetch(payload, pat=_PAT)

        # Task 2.3 — Qdrant retrieval + prompt assembly
        if fetch_result.fallback:
            logger.info("large_pr_macro_path repo=%s pr=%s", repo_id, pr_number)
            prompt = build_macro_prompt(payload)
        else:
            def _retrieve_in_thread():
                return retrieve_and_assemble(
                    fetch_result.files, payload, _get_qdrant(), _get_embedder()
                )
            pr_obj = await asyncio.get_event_loop().run_in_executor(None, _retrieve_in_thread)
            prompt = pr_obj.prompt

        # Task 2.4 — inference handoff
        logger.info("inference_start repo=%s pr=%s", repo_id, pr_number)
        review = await call_inference(prompt, _PHI_ENDPOINT, _PHI_KEY, phi_model=_PHI_DEPLOYMENT)
        logger.info("inference_complete repo=%s pr=%s chars=%d", repo_id, pr_number, len(review))

        # Task 2.4 — GitHub callback
        await post_github_comment(repo_id, pr_number, review, _PAT)
        logger.info("review_complete repo=%s pr=%s", repo_id, pr_number)

    except Exception as exc:
        logger.error(
            "review_failed repo=%s pr=%s error=%s",
            repo_id, pr_number, exc, exc_info=True,
        )
        try:
            await post_github_comment(repo_id, pr_number, FALLBACK_COMMENT, _PAT)
        except Exception as cb_exc:
            logger.error("fallback_comment_failed repo=%s pr=%s error=%s", repo_id, pr_number, cb_exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/review", status_code=202)
async def post_review(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
) -> JSONResponse:
    """
    Receive a PR trigger from the GitHub Actions workflow.

    Security gate: validates HMAC-SHA256 signature before any processing.
    Returns 202 immediately; all orchestration runs in the background.
    """
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    payload: dict[str, Any] = await request.json()
    background_tasks.add_task(_process_review, payload)

    return JSONResponse(status_code=202, content={"status": "accepted"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
