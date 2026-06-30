#!/usr/bin/env python3
"""
Phase 0.3 — Pattern Buddy Infrastructure Smoke Test

Verifies all three infrastructure layers are wired correctly before any
application code is written.

Steps:
  1  nomic-embed-text ONNX embedding (via fastembed)        → assert dim == 768
  2  Qdrant collection setup                                 → create if absent,
                                                              delete legacy collection,
                                                              create repo_id tenant index
  3  Qdrant upsert                                           → full payload schema
  4  Qdrant retrieval + tenant isolation                     → score ≥ 0.99, wrong
                                                              repo_id returns 0 hits
  5  Inference endpoint                                      → HTTP 200 + Markdown body
  6  Cleanup                                                 → delete test vector

Env vars required (set in .env):
  QDRANT_URL, QDRANT_API_KEY
  PHI_ENDPOINT, PHI_API_KEY
  PHI_DEPLOYMENT_NAME   (default: Phi-4-mini-reasoning)

Exit code 0 = all steps passed. Exit code 1 = one or more failures.
"""

import os
import sys
import uuid
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.environ["QDRANT_URL"]
QDRANT_API_KEY  = os.environ["QDRANT_API_KEY"]
PHI_ENDPOINT    = os.environ["PHI_ENDPOINT"]
PHI_API_KEY     = os.environ["PHI_API_KEY"]
PHI_MODEL       = os.getenv("PHI_DEPLOYMENT_NAME", "Phi-4-mini-reasoning")

COLLECTION      = "global_codebase_memory"
LEGACY_COLL     = "pattern_buddy_rules"
NOMIC_DIM       = 768
TEST_REPO_ID    = "smoke-test-repo"
TEST_POINT_ID   = str(uuid.uuid4())
# Querying with the exact stored vector → cosine similarity should be ~1.0
SCORE_FLOOR     = 0.99

# ── Helpers ───────────────────────────────────────────────────────────────────
_G = "\033[92m"   # green
_R = "\033[91m"   # red
_X = "\033[0m"    # reset

results: dict[str, bool] = {}

def _log(name: str, ok: bool, detail: str = "") -> bool:
    tag = f"{_G}✓ PASS{_X}" if ok else f"{_R}✗ FAIL{_X}"
    suffix = f"  ({detail})" if detail else ""
    print(f"    {tag}  {name}{suffix}")
    results[name] = ok
    return ok


# ── Step 1: ONNX Embedding ────────────────────────────────────────────────────
def step_embed() -> list[float] | None:
    print("\n  [1/6] nomic-embed-text via fastembed (ONNX Runtime)")
    print("        Note: first run downloads ~130 MB model to ~/.cache/fastembed/")
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        embeddings = list(model.embed(["Pattern Buddy smoke test"]))
        vec = embeddings[0].tolist()
        ok = _log("embedding dimension", len(vec) == NOMIC_DIM,
                  f"got {len(vec)}, expected {NOMIC_DIM}")
        return vec if ok else None
    except Exception as exc:
        _log("embedding dimension", False, str(exc))
        return None


# ── Step 2: Qdrant Collection Setup ───────────────────────────────────────────
def step_qdrant_setup(client: QdrantClient) -> None:
    print("\n  [2/6] Qdrant collection setup")

    # Delete the legacy collection that has the wrong schema (3072-dim, no repo_id index)
    if client.collection_exists(LEGACY_COLL):
        client.delete_collection(LEGACY_COLL)
        _log(f"deleted legacy '{LEGACY_COLL}'", True)
    else:
        _log(f"legacy '{LEGACY_COLL}' absent", True, "nothing to delete")

    # Create or verify global_codebase_memory
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        size = info.config.params.vectors.size
        _log(f"collection '{COLLECTION}' exists", size == NOMIC_DIM,
             f"vector size {size}, expected {NOMIC_DIM}")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=NOMIC_DIM, distance=models.Distance.COSINE),
        )
        _log(f"created '{COLLECTION}'", True, "cosine, dim 768")

    # Create repo_id tenant index — O(1) routing per the spec
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="repo_id",
            field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
        )
        _log("repo_id tenant index (is_tenant=True)", True)
    except Exception as exc:
        if "already exists" in str(exc).lower():
            _log("repo_id tenant index (is_tenant=True)", True, "already present")
        else:
            _log("repo_id tenant index (is_tenant=True)", False, str(exc))


# ── Step 3: Upsert ────────────────────────────────────────────────────────────
def step_upsert(client: QdrantClient, vec: list[float]) -> None:
    print("\n  [3/6] Qdrant upsert with full payload schema")
    payload = {
        "repo_id":        TEST_REPO_ID,
        "file_path":      "tests/test_smoke.py",
        "file_type":      "source_code",
        "chunk_strategy": "ast_function",
        "target_module":  None,
        "commit_hash":    "000000000000000000000000000000000000dead",
        "content":        "def test_smoke(): assert True",
    }
    try:
        client.upsert(
            collection_name=COLLECTION,
            points=[models.PointStruct(id=TEST_POINT_ID, vector=vec, payload=payload)],
        )
        _log("upsert to global_codebase_memory", True)
    except Exception as exc:
        _log("upsert to global_codebase_memory", False, str(exc))


# ── Step 4: Retrieval + Tenant Isolation ─────────────────────────────────────
def step_retrieve(client: QdrantClient, vec: list[float]) -> None:
    print("\n  [4/6] Qdrant retrieval + tenant isolation")

    # 4a: retrieve own repo — score should be near 1.0
    try:
        result = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="repo_id",
                                      match=models.MatchValue(value=TEST_REPO_ID))
            ]),
            limit=1,
            score_threshold=0.75,
        )
        hits = result.points
        ok = len(hits) == 1 and hits[0].score >= SCORE_FLOOR
        detail = f"score {hits[0].score:.4f}" if hits else "0 results returned"
        _log("retrieval by correct repo_id", ok, detail)
    except Exception as exc:
        _log("retrieval by correct repo_id", False, str(exc))

    # 4b: tenant isolation — a different repo_id must return zero results
    try:
        result = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="repo_id",
                                      match=models.MatchValue(value="other-tenant-repo"))
            ]),
            limit=1,
            score_threshold=0.75,
        )
        hits = result.points
        _log("tenant isolation (wrong repo_id → 0 results)", len(hits) == 0,
             f"got {len(hits)} result(s)")
    except Exception as exc:
        _log("tenant isolation (wrong repo_id → 0 results)", False, str(exc))


# ── Step 5: Inference Endpoint ────────────────────────────────────────────────
def step_inference() -> None:
    # Interim: tests the Azure AI Services Phi deployment.
    # Replace PHI_ENDPOINT + PHI_MODEL with the internal ACI URL once provisioned.
    print(f"\n  [5/6] Inference endpoint")
    print(f"        {PHI_ENDPOINT}")
    print(f"        model: {PHI_MODEL}")

    headers = {
        "Authorization": f"Bearer {PHI_API_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": PHI_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are a code reviewer. Output only Markdown."},
            {"role": "user",
             "content": 'Respond with exactly two lines:\n# Review\n\nNo violations found.'},
        ],
        "max_tokens": 64,
        "temperature": 0,
    }

    try:
        with httpx.Client(timeout=30) as http:
            resp = http.post(PHI_ENDPOINT, headers=headers, json=body)

        if resp.status_code != 200:
            _log("inference HTTP 200", False,
                 f"status {resp.status_code}: {resp.text[:200]}")
            return

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        _log("inference HTTP 200", True, f"{len(content)} chars")
        _log("Markdown content non-empty", bool(content.strip()),
             repr(content[:80]) if content else "empty")

    except httpx.ConnectError as exc:
        _log("inference HTTP 200", False,
             f"connection refused — is the endpoint reachable? {exc}")
    except Exception as exc:
        _log("inference HTTP 200", False, str(exc))


# ── Step 6: Cleanup ───────────────────────────────────────────────────────────
def step_cleanup(client: QdrantClient) -> None:
    print("\n  [6/6] Cleanup")
    try:
        client.delete(
            collection_name=COLLECTION,
            points_selector=models.PointIdsList(points=[TEST_POINT_ID]),
        )
        _log("deleted smoke-test vector", True)
    except Exception as exc:
        _log("deleted smoke-test vector", False, str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  Pattern Buddy — Phase 0.3 Smoke Test           ║")
    print("  ╚══════════════════════════════════════════════════╝")

    vec = step_embed()

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    step_qdrant_setup(qdrant)

    if vec is not None:
        step_upsert(qdrant, vec)
        step_retrieve(qdrant, vec)
    else:
        print("\n  (skipping upsert/retrieval — embedding step failed)")
        results["upsert to global_codebase_memory"] = False
        results["retrieval by correct repo_id"] = False
        results["tenant isolation (wrong repo_id → 0 results)"] = False

    step_inference()
    step_cleanup(qdrant)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results.values())
    total  = len(results)
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  {passed}/{total} passed", end="")
    print(" " * (47 - len(f"{passed}/{total} passed")) + "║")
    print("  ╚══════════════════════════════════════════════════╝")

    if all(results.values()):
        print("\n  Infrastructure ready — proceed to Phase 1.\n")
        sys.exit(0)
    else:
        failed = [name for name, ok in results.items() if not ok]
        print("\n  Failed steps:")
        for name in failed:
            print(f"    • {name}")
        print("\n  Resolve failures before proceeding to Phase 1.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
