"""
Phase 2, Task 2.4 — Inference handoff and GitHub callback.

Public API
----------
call_inference(prompt, phi_endpoint, phi_key, http_timeout) -> str
post_github_comment(repo_id, pr_number, body, pat, http_timeout) -> None

The Azure AI Foundry endpoint uses the OpenAI-compatible chat completions API.
GitHub comment posting uses the Issues Comments endpoint (PRs share the
issues namespace in the GitHub API).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_PHI_CONTEXT_WINDOW: int = 8_192   # Phi-4-mini-reasoning hard limit

from api.retrieval import PROMPT_TOKEN_CAP  # noqa: E402 — avoids circular at call time

# Generation headroom = context window − input cap; must stay in sync with retrieval.py.
MAX_NEW_TOKENS: int = _PHI_CONTEXT_WINDOW - PROMPT_TOKEN_CAP

INFERENCE_TIMEOUT_S: float = 120.0   # Phi-4-mini-reasoning can be slow; cap at 2 min
GITHUB_TIMEOUT_S:    float = 30.0

GITHUB_API_BASE = "https://api.github.com"

FALLBACK_COMMENT = (
    "_Pattern Buddy encountered an error during review. "
    "Please check the App Service logs for details._"
)


# ── Inference ─────────────────────────────────────────────────────────────────

async def call_inference(
    prompt: str,
    phi_endpoint: str,
    phi_key: str,
    http_timeout: float = INFERENCE_TIMEOUT_S,
) -> str:
    """
    POST the assembled prompt to the Azure AI Foundry Phi-4-mini-reasoning
    serverless endpoint and return the generated review text.

    The endpoint uses the OpenAI-compatible chat completions API format.
    Raises httpx.HTTPStatusError on non-2xx or httpx.TimeoutException on timeout.
    """
    headers = {
        "Authorization": f"Bearer {phi_key}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
    }

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(phi_endpoint, json=body, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected inference response shape: {data!r}") from exc


# ── GitHub comment ────────────────────────────────────────────────────────────

def _comment_url(repo_id: str, pr_number: int) -> str:
    """
    PRs share the GitHub Issues namespace.
    POST /repos/{owner}/{repo}/issues/{number}/comments
    """
    return f"{GITHUB_API_BASE}/repos/{repo_id}/issues/{pr_number}/comments"


async def post_github_comment(
    repo_id: str,
    pr_number: int,
    body: str,
    pat: str,
    http_timeout: float = GITHUB_TIMEOUT_S,
) -> None:
    """
    Post a Markdown comment to the GitHub pull request.
    Raises httpx.HTTPStatusError on non-2xx.
    """
    url = _comment_url(repo_id, pr_number)
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(url, json={"body": body}, headers=headers)
        resp.raise_for_status()

    logger.info(
        "comment_posted repo=%s pr=%s chars=%d",
        repo_id, pr_number, len(body),
    )
