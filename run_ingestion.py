#!/usr/bin/env python3
"""
Run the Phase 1 ingestion pipeline against a local repo clone.

Usage:
    python run_ingestion.py <repo_path> <repo_id> [--commit <sha>]

Examples:
    python run_ingestion.py ../my-repo owner/my-repo
    python run_ingestion.py ../my-repo owner/my-repo --commit abc1234

Env vars required (set in .env or export):
    QDRANT_URL
    QDRANT_API_KEY
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from ingestion.pipeline import ingest_repository, QDRANT_COLLECTION

load_dotenv()

QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
NOMIC_DIM      = 768


def _ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        info = client.get_collection(QDRANT_COLLECTION)
        size = info.config.params.vectors.size
        if size != NOMIC_DIM:
            print(f"  WARNING: collection exists but vector size is {size}, expected {NOMIC_DIM}")
        else:
            print(f"  Collection '{QDRANT_COLLECTION}' already exists (dim={size})")
    else:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=NOMIC_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        print(f"  Created collection '{QDRANT_COLLECTION}' (cosine, dim={NOMIC_DIM})")

<<<<<<< HEAD
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="repo_id",
            field_schema=qdrant_models.KeywordIndexParams(type="keyword", is_tenant=True),
        )
        print("  repo_id tenant index created")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print("  repo_id tenant index already present")
        else:
            raise
=======
    for field, is_tenant in [("repo_id", True), ("file_path", False)]:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=qdrant_models.KeywordIndexParams(type="keyword", is_tenant=is_tenant),
            )
            print(f"  {field} index created")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                print(f"  {field} index already present")
            else:
                raise
>>>>>>> all-phases


def _head_sha(repo_path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a repository into Qdrant.")
    parser.add_argument("repo_path", help="Path to the local repo clone")
    parser.add_argument("repo_id",   help="Repo identifier, e.g. owner/repo-name")
    parser.add_argument("--commit",  default=None, help="Commit SHA (defaults to HEAD)")
    args = parser.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env or environment.")
        sys.exit(1)

    commit_hash = args.commit or _head_sha(args.repo_path)

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  Pattern Buddy — Phase 1 Ingestion              ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print(f"  repo_path  : {args.repo_path}")
    print(f"  repo_id    : {args.repo_id}")
    print(f"  commit_hash: {commit_hash}")
    print(f"  qdrant_url : {QDRANT_URL}")
    print()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print("  [1/3] Ensuring Qdrant collection...")
    _ensure_collection(client)

    print("\n  [2/3] Running ingestion pipeline...")
    print("        (first run downloads ~130 MB nomic-embed model to ~/.cache/fastembed/)\n")
    result = ingest_repository(
        repo_path=args.repo_path,
        repo_id=args.repo_id,
        commit_hash=commit_hash,
        qdrant_client=client,
    )

    print("\n  [3/3] Done.")
    print(f"        files processed : {result['files_processed']}")
    print(f"        files skipped   : {result['files_skipped']}")
    print(f"        chunks upserted : {result['chunks_upserted']}")
    print()
    print("  Ingestion complete. Qdrant is populated and ready for Phase 2 queries.")
    print()


if __name__ == "__main__":
    main()
