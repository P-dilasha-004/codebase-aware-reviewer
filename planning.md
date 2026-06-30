# Engineering Specification: Codebase RAG & PR Review Agent

> **Instructions for the Engineer**
>
> This document is your architectural contract. Define the exact mathematical, infrastructural, and logical boundaries of the system before writing ingestion code.
>
> Vague answers will result in hallucinations, broken context windows, and data leaks.
>
> Fill in every bracketed `[ ]` section explicitly.

---

# 1. System Objectives & Scope Boundaries

## Primary Objective

The primary objective of the Automated Code Review Agent is to deliver highly targeted, context-aware pull request reviews without the prohibitive costs and token limitations associated with passing an entire codebase to an LLM.

It accomplishes this by leveraging a Retrieval-Augmented Generation (RAG) architecture and a Qdrant-powered vector database. When a new Pull Request is submitted, the agent executes the following core workflow:

###Diff Isolation###: Extracts the specific code changes introduced by the developer.

Intelligent Retrieval: Converts the diff into a vector embedding and queries the Qdrant database to fetch a localized, highly relevant bundle of repository context, including surrounding source code, behavioral tests, architectural decision records (ADRs), and internal style guidelines.

Contextual Generation: Hands this precise, filtered bundle to a generative LLM to produce an accurate, high-quality code review that understands the project's specific engineering culture, structural boundaries, and historical decisions.

Multi-Tenant Isolation: The agent is architected to serve multiple repositories simultaneously within a single Qdrant collection. Each repository's vectors are physically co-located and logically isolated via a strict repo_id payload index. A middleware enforcement layer permanently appends this identifier to every retrieval query, ensuring that context fetched for one repository can never bleed into the review of another.

## Explicit Non-Goals
To maintain precision, minimize hallucination, and optimize retrieval latency, the agent is strictly designed not to perform the following:

###Code Execution & CI/CD Validation###: The agent does not compile code, run test suites, or execute build pipelines. It is a static, semantic reviewer that relies on ingested behavioral tests to understand intent, not execution.

###Global Architecture Refactoring###: The agent's analytical scope is constrained to the PR diff and its immediate contextual dependencies. It will not suggest sweeping, out-of-scope codebase rewrites or unrelated optimizations.

###Deep Dependency & Lockfile Analysis###: As outlined in the ingestion blueprint, the agent does not parse deep dependency trees, exact hash lockfiles (e.g., poetry.lock, uv.lock, or eventually package-lock.json), or the raw source code of external third-party libraries. It relies solely on top-level manifests (like pyproject.toml) for environmental boundaries.

###Cross-Repository Knowledge Transfer###: Due to the strict multitenancy middleware lock, the agent will never cross-pollinate data. It cannot leverage coding patterns, snippets, or historical PR context from one tenant's repository to inform the review of another.

###Replacing Deterministic Security Scanners###: While the agent can enforce team-specific security guidelines ingested from the prose track (e.g., catching hardcoded secrets based on a SECURITY.md), it is not a replacement for dedicated Static Application Security Testing (SAST) tools or vulnerability scanners.

## Target Audience
The output of the Automated Code Review Agent is designed to serve two primary groups within the engineering team:

###Pull Request Authors (Developers)###: The engineers submitting the code changes. They consume the agent's reviews to get immediate, actionable feedback on their diffs. The agent acts as an initial guardrail, helping them catch localized architectural violations, missing test coverage, and deviations from internal style guides before a human reviewer ever looks at the code.

###Pull Request Reviewers###: The developers responsible for ultimately approving the PR. They use the agent's output as an automated "first pass" or pre-filter. By relying on the agent to enforce Architectural Decision Records (ADRs) and flag contextual inconsistencies, human reviewers can focus their cognitive effort on complex business logic and broader systemic design rather than policing boilerplate, formatting, or forgotten edge cases.

---

# 2. Trigger Infrastructure & Execution Environment

## Compute Environment

This architecture requires three strictly segregated compute workloads to balance web traffic, latency, and heavy machine learning inference:

The PR Review Trigger (Light/Event-Driven): Executed via GitHub Actions (ubuntu-latest runner). This workflow is purely a stateless trigger on pull_request events. It does not download code or execute LLMs. It simply extracts the diff metadata and forwards the pointer payload to the Backend API.

The Backend API & Orchestrator (I/O & Embedding Bound): Executed on a standard Azure App Service. This acts as the central nervous system. It handles the GitHub webhook ingress and performs concurrent asynchronous file fetching. Crucially, it holds the lightweight nomic-embed-text model in memory. This allows it to instantly generate vectors for the PR diff, query Qdrant, and construct the final prompt. It also handles the background ingestion syncing for the repository.

The Model Inference Engine (Heavy/Compute-Bound): Executed on a dedicated, segregated Azure Container Instance (ACI) or a separate compute-optimized App Service plan. This environment is strictly an internal generation server. It holds only the Phi-4-mini-instruct model in memory. It has no public internet ingress and only accepts pre-packaged prompt requests from the Backend API.

## Trigger & Diff Extraction Method

Method: Pointer-Based Payload Routing & Asynchronous Webhook.

Architectural Execution:

The Trigger & Handoff: The GitHub Action queries the Git API strictly to identify which files changed. It constructs a lightweight JSON payload containing only the metadata and the raw blob URLs of the modified files, and POSTs it to the Azure App Service.

The Asynchronous Acknowledgment (Critical): To prevent the GitHub Action from timing out while waiting for the LLM inference, the Azure App Service must instantly return an HTTP 202 Accepted to close the CI connection. All subsequent processing moves to a background thread.

Concurrent Native Fetch: In the background, the Azure App Service utilizes native asynchronous I/O (asyncio) to concurrently fetch the raw file blobs directly from GitHub. Crucially, it must attach the injected GitHub PAT to these requests to prevent the App Service IP from hitting unauthenticated API rate limits.

The Embedding & Context Phase: The Backend API processes the fetched source code through its local nomic-embed-text model to generate numerical vectors. It immediately uses these vectors to query Qdrant for matching architectural guidelines, tests, and documentation.

The Inference Handoff & Callback: The Backend API packages the diff and Qdrant context into a structured prompt and sends an HTTP POST request to the isolated Model Inference Engine. Once the Inference Engine returns the generated review, the Backend API makes a final outbound call to the GitHub API to post the review as a PR comment.

## API Segregation & Fallback: 

The Backend API strictly limits fetching to modified source code files. It must never request repository documentation (/contents) or historical PR comments (/issues/comments) during the active PR review loop. Those resources are extracted asynchronously during the background ingestion phase.

Defensive Threshold: If a PR modifies >50 files, the Backend API will bypass raw blob fetching, embedding generation, and Qdrant queries entirely. To protect the Phi-4-mini context window from collapse, it falls back to a "Macro-Level Summary" based strictly on commit messages and file names, sending a simplified prompt to the Inference Engine to abort line-by-line analysis.

## Authentication & Secrets

Method: CI-Native Vaults (GitHub Actions Secrets) & Azure App Settings.

Injection Strategy: * GitHub Actions: Injected strictly with the webhook URL of the Azure App Service and a verification token.

Backend API (App Service): Injected with the GitHub PAT (for fetching code and posting comments), the Qdrant URL/API Key, and the internal routing URL/auth token for the Model Inference Engine.

Model Inference Engine: Requires no external credentials, as it strictly serves internal requests from the Backend API.

Because the architecture uses open-weights models locally for both embeddings (nomic-embed-text) and generation (Phi-4-mini), no external LLM provider keys (e.g., Azure OpenAI, Anthropic) are required or injected.

Security Constraint: Secrets must never be written to .env files on the CI runner's disk, the App Service disk, or the Inference container disk. All GitHub Actions secrets must utilize the ::add-mask:: directive to prevent log bleeding.

---

## 3. Polymorphic Ingestion & Token Math

The ingestion pipeline delivers context to an in-process Phi-4-mini-instruct model. Because redundant or misaligned context reduces retrieval precision and increases hallucination risk, this pipeline utilizes a Polymorphic Router rather than a naive sliding-window text splitter. The token math below establishes targeted boundaries to optimize retrieval relevance and protect compute resources.

**Embedding Model:** `nomic-embed-text` (Open-Weights, ~137M parameters).
* **Deployment Note:** Executed locally within the Azure App Service (Backend API) to generate vectors instantly at zero API cost.

**Model Context Limits:** * `8,192` tokens. Both `nomic-embed-text` (truncation threshold) and `Phi-4-mini` (generation threshold) share this architectural constraint. 
* **Guardrail:** The prompt construction logic must enforce a target upper bound of ≤7,000 tokens for combined context and diff pointers, reserving an ample buffer for the generation output.

**Target Chunk Size:** Dynamic, capped at `~800` tokens.
* **Code Track (AST):** Chunk boundaries target Python function or class lengths. Methods exceeding 800 tokens are split at internal logical junctures. *Note: As AST parsing can fail on syntactically invalid PR code, the pipeline must implement a fallback to regex-based logical splitting.*
* **Prose Track (Markdown):** Bound by header levels (`##`, `###`). Chunks range from 300–600 tokens depending on density.
* **Config Track:** Capped at 100–300 tokens via a plain-English translation strategy. 
* **Config Defensive Rule:** For massive generated configurations (e.g., `package-lock.json`), the pipeline bypasses full translation and extracts only the top-level metadata and file path to maintain repository awareness without blowing out the token budget.

**Chunk Overlap:** Dynamic (`0` – `100` tokens).
* **Code Track (AST):** `0` tokens. Arbitrary overlap creates redundant code fragments. Structure is preserved by prepending an immutable metadata header to the top of every split chunk: `// Context: File: [path] | Class: [name] | Method: [name]`.
* **Prose Track (Markdown):** `~100` tokens. Applied strictly when a section breaches the 800-token target, ensuring semantic narrative continuity.
* **Config Track:** `0` tokens. Declarative schemas do not require narrative adjacency.

---

## Track A — Source Code (.py)

### Parser Tool

Primary Recommendation: Tree-sitter.

Rationale: While native AST modules (like ast in Python) require perfectly compilable code, Tree-sitter is a fault-tolerant incremental parser. This allows the pipeline to extract structural information from "dirty" PR diffs or incomplete code fragments without triggering fatal ingestion crashes.

### Boundary Logic

Target Chunk Size: ~800 tokens.

Primary Split: Code is chunked strictly at functional boundaries (Class/Function).

Oversized Entity Fallback: If a single entity exceeds 800 tokens, it is split at internal logical junctures (e.g., control flow blocks like try/except or top-level for loops).

Global Header Injection: To provide essential context, the file's top-level imports and global constants are extracted as a "Header Chunk" and are prepended to all functional chunks derived from that file.

Scope Preservation: To maintain semantic identity, every chunk is prepended with its full lexical call-stack hierarchy.

Format: // Context: File: [path] | Class: [name] | Method: [name] | Block: [e.g., try-block]

### Test File Mapping

Strategy: Tiered Hybrid Retrieval.

Primary (Heuristic): $O(1)$ filename matching (e.g., src/auth.py -> tests/test_auth.py). This is the default to optimize performance in constrained Azure App Service tiers.

Fallback (Symbol Resolution): If heuristic matching yields low-confidence scores, the pipeline triggers a secondary AST-based import analysis to map the test file to the specific modules it validates.

## Track B — Prose & Documentation (.md, .txt)

### Target Size
300–600 tokens.

### Boundary Logic

Hierarchical Splitting: Split strictly by Markdown header levels (#, ##, ###).

**Fallback (no headers present):** Recursive boundary descent — `\n\n` paragraph breaks → sentence boundaries (`. `, `! `, `? `) → fixed 800-token hard cut. Step down only when the current level produces a chunk exceeding the size target. 100-token overlap applies at paragraph level and below only.

Inheritance: Sub-sections inherit the title strings of their two immediate parent headers to preserve narrative context.

Authoritative Labeling: Every chunk is prepended with the source document name (e.g., [Source: security_policy.md]) to signal authoritative weight to the LLM.

### Chunk Overlap
~100 tokens. Applied only when a single logical section exceeds the 800-token limit, ensuring continuity across paragraph boundaries.

## Track C — Configuration & Manifests (.toml, .json, .yaml)

### Target Size
100–300 tokens.

### Boundary Logic

Declarative Summarization: Configurations are parsed as objects and flattened into declarative English sentences (e.g., {"debug": true} -> "The debug environment flag is enabled.").

Massive File Guardrail: To prevent context window collapse, any configuration or lockfile exceeding 2,000 lines is filtered to ingest only top-level metadata and the file path, bypassing full translation.

### Chunk Overlap
0 tokens. Key-value structures do not possess narrative adjacency and require no overlap.

---

# 4. Vector Database & Multitenancy Security

## Database Provider

**Qdrant**

## Collection Name

```text
global_codebase_memory
```

## Tenant Isolation Strategy

Payload-based partitioning utilizing Qdrant's native multitenancy features.

### Critical Indexing Requirement

The `repo_id` field must be explicitly indexed with the `is_tenant: true` parameter upon collection creation. This forces Qdrant to physically co-locate all vectors for a specific repository on disk, ensuring that filtering by `repo_id` is a $O(1)$ routing operation rather than a full-collection scan.

### Middleware Retrieval Lock

The LLM agent is strictly denied direct access to the Qdrant query interface.

A middleware enforcement layer must permanently append a strict filter to all incoming retrieval requests:

```json
{
  "must": [
    {
      "key": "repo_id",
      "match": {
        "value": "<CURRENT_REPO>"
      }
    }
  ]
}
```

This guarantees that context fetched for one repository can never bleed into the review of another, even if the LLM hallucinates a malicious query.

## Required Payload Schema

```json
{
  "repo_id": "[Extracted from webhook payload]",
  "file_path": "[Relative path from repository root]",
  "file_type": "[source_code | prose_doc | config]",
  "chunk_strategy": "[ast_function | markdown_header | llm_translation]",
  "target_module": "[Optional source file if test file]",
  "commit_hash": "[SHA of the commit triggering the ingestion]",
  "content": "[Raw text chunk]"
}
```

## Field Definitions

| Field | Description | Execution Rule |
|---------|------------|----------------|
| `repo_id` | Repository identifier. | Mandatory. Used for strict multitenancy isolation. |
| `file_path` | Relative path from repository root. | Mandatory. Used to provide spatial context to the LLM. |
| `file_type` | Classification of the file being embedded. | Mandatory. Enables filtering (e.g., query only `prose_doc`). |
| `chunk_strategy` | Method used to create the chunk. | Mandatory. Tracks provenance of the text chunk. |
| `target_module` | Associated source module for test files. | Optional. Exists only on test tracks to link behaviors to source code. |
| `commit_hash` | SHA of the current commit. | Mandatory. Used for cache invalidation. Stale vectors with older hashes for the same `file_path` must be pruned. |
| `content` | Raw chunk text used for embedding. | Mandatory. The text supplied to the generation model. |

## Retrieval Strategy

### Similarity Metric

```yaml
cosine
```

### Top-K

```yaml
5
```

### Score Threshold

```yaml
0.75
```

### Mandatory Filter

```yaml
repo_id
```

### Optional Filters

```yaml
- file_type
- target_module
```

---

# 5. The Update Loop (Continuous Sync)

## Merge Trigger & Cache Invalidation

To maintain a synchronized vector state without doing a full repository re-ingestion, the system listens for merge events and applies a strict Delete-Then-Upsert pattern.

## Architectural Execution

### The Event Trigger & Verification

The Azure App Service webhook receiver validates the incoming GitHub payload.

It explicitly verifies that:

```json
{
  "action": "closed",
  "merged": true
}
```

to prevent triggering ingestion on closed-but-unmerged (abandoned) PRs.

### The Extraction & Handoff

The service extracts:

- `repo_id`
- `commit_hash` (merge commit SHA)
- Array of `file_path` values representing files that were modified, added, or deleted

### Asynchronous Acknowledgment (Critical)

The App Service immediately returns an HTTP `202 Accepted` response to close the GitHub connection.

All synchronization work is delegated to a background task to prevent webhook timeout failures.

### The Deletion Phase (Cache Invalidation)

To prevent orphaned or stale chunks from accumulating, the background task issues a DELETE operation against Qdrant.

The deletion filter must match:

- `repo_id`
- Any file contained in the changed file list

Example logic:

```json
{
  "must": [
    {
      "key": "repo_id",
      "match": {
        "value": "<CURRENT_REPO>"
      }
    },
    {
      "key": "file_path",
      "match": {
        "any": [
          "src/auth.py",
          "src/user.py"
        ]
      }
    }
  ]
}
```

All existing vectors associated with the changed files are removed before re-ingestion begins.

### The Upsert Phase

#### Modified or Added Files

For files that were modified or added:

1. Fetch the latest raw blob from GitHub.
2. Process the file through the ingestion pipeline.
3. Generate embeddings using `nomic-embed-text`.
4. Create updated chunks.
5. Upsert the new vectors into Qdrant.
6. Stamp each vector with the new `commit_hash`.

#### Deleted Files

For files explicitly deleted in the merge:

- No re-ingestion occurs.
- The process ends after the Deletion Phase.
- All associated vectors remain permanently removed from Qdrant.

---

## State Reconciliation

### Recommended Strategy

**Incremental Diff-Based Reindexing**

> Do **not** re-ingest the entire repository after every merge.

### Workflow

#### Modified Files

1. Extract changed files from the merged PR.
2. For each modified file:

```json
{
  "file_path": "src/services/auth.py"
}
```

3. Query Qdrant for existing chunks.
4. Delete existing chunks.
5. Re-chunk updated file.
6. Generate embeddings.
7. Upsert new chunks.

#### Deleted Files

- Remove all associated chunks from Qdrant.

#### Renamed Files

- Delete chunks under the old path.
- Insert chunks under the new path.

### Benefits

- Lower embedding cost
- Faster indexing
- Reduced webhook latency
- Consistent repository synchronization

---

## Cron Fallback

### Weekly Full Repository Rebuild

**Enabled:** Yes

### Schedule

```yaml
frequency: weekly
```

### Purpose

Acts as a reconciliation mechanism to detect:

- Missed webhooks
- Failed ingestion jobs
- Corrupted vector entries
- Metadata drift
- Partial indexing failures

### Workflow

1. Clone the latest `main` branch.
2. Re-process the entire repository.
3. Compare chunk counts.
4. Replace repository namespace if inconsistencies are detected.

### Why Weekly?

- Full ingestion is expensive.
- Incremental updates handle normal operations.
- Weekly rebuild provides operational safety.

---

# 6. Prompt Engineering Layer

## Retrieval K Value

### Recommended Setting

```yaml
retrieval_k: 5
```

### Guidelines

| Repository Size | K Value |
|-----------------|---------|
| Small–Medium Repository | 5 |
| Large Monorepo | 8–10 |

---

## Generative Model

### Recommended

**Claude 3.5 Sonnet**

#### Reasons

- Strong code review quality
- Excellent instruction following
- Large context window
- Consistent repository reasoning

### Alternative

- GPT-4o

---

## System Prompt Architecture

### System Persona

```text
You are a Senior Principal Engineer responsible for enforcing
repository standards and architectural consistency.

Your job is NOT to approve code.

Your job is to identify:
- Rule violations
- Missing tests
- Security risks
- Architectural inconsistencies
- Documentation violations

Only enforce rules appearing in retrieved repository context.

Do not invent policies.

If no rule applies, state:

"No repository rule violation detected."
```

---

### Prompt Template

```text
=========================
SYSTEM
=========================

You are a Senior Principal Engineer responsible for enforcing
repository standards.

Only use repository rules contained in supplied context.

Do not hallucinate rules.

Output valid Markdown.

=========================
REPOSITORY CONTEXT
=========================

{retrieved_qdrant_chunks}

=========================
PULL REQUEST DIFF
=========================

{pr_diff}

=========================
TASK
=========================

Review the pull request.

Identify:
1. Violated repository rules
2. Security concerns
3. Missing tests
4. Documentation gaps

For every finding:

- Severity
- Explanation
- Supporting repository rule
- Suggested fix

If no violation exists, explicitly state that.

=========================
OUTPUT FORMAT
=========================

# Review Summary

## Finding 1

Severity: High

Rule Source:
<file>

Issue:
<description>

Suggested Fix:
<description>
```

---

## Context Injection

### Retrieved Repository Context

Inserted into:

```text
REPOSITORY CONTEXT
```

Example:

```text
CONTRIBUTING.md

All new utility functions require
corresponding unit tests.
```

---

## Diff Injection

Inserted into:

```text
PULL REQUEST DIFF
```

Example:

```diff
+ def calculate_tax(amount):
+     return amount * 0.08
```

---

## Output Constraints

```text
- Output valid Markdown only
- Do not compliment authors
- Do not speculate
- Do not invent repository policies
- Cite exact repository file
- Include severity for every issue
- Focus on actionable feedback
```

---

# 7. Evaluation Matrix

| Scenario | PR Change | Retrieved Rule | Expected Output |
|-----------|-----------|----------------|-----------------|
| Missing Tests | New utility without tests | CONTRIBUTING.md | Reject |
| Missing Documentation | New endpoint without docs | API_GUIDELINES.md | Reject |
| Forbidden Dependency | Unapproved package | DEPENDENCIES.md | Reject |
| Missing Authentication | Protected route lacks auth | SECURITY.md | Reject |
| Layer Violation | Controller accesses DB directly | ARCHITECTURE.md | Reject |

---

## Pass Criteria

The agent succeeds only if:

1. The correct rule file is retrieved.
2. The rule is cited in the output.
3. The violation is correctly identified.
4. No hallucinated policies are introduced.

---

# 8. Failure Modes & Fallbacks

## Failure Mode #1 — PR Diff Exceeds Context Window

### Example

```text
15,000-line PR
```

### Fallback

1. Split diff by file.
2. Review files independently.
3. Aggregate findings.
4. If still too large:

```text
PR exceeds automated review limits.
Please split into smaller pull requests.
```

---

## Failure Mode #2 — AST Parser Error

### Example

```python
def hello(
```

### Fallback

1. Catch parser exception.
2. Log error.
3. Mark file as unparsed.
4. Continue processing remaining files.
5. Emit warning:

```text
Unable to parse:

src/example.py

File skipped due to syntax error.
```

---

## Failure Mode #3 — Webhook Delivery Failure

### Retry Policy

```yaml
attempt_1: immediate
attempt_2: 1 minute
attempt_3: 5 minutes
attempt_4: 15 minutes
attempt_5: 1 hour
```

### If Retries Fail

- Send alert
- Push to dead-letter queue
- Allow manual re-trigger:

```http
POST /admin/reindex/repository
```

---

## Failure Mode #4 — Qdrant Unavailable

### Fallback

1. Retry connection.
2. Queue indexing job.
3. Preserve webhook payload.
4. Reprocess when Qdrant becomes available.

> No PR review should run against partial repository context.

---

## Failure Mode #5 — Embedding API Failure

### Fallback

1. Retry embedding generation.
2. Queue failed files.
3. Continue processing unaffected files.
4. Alert if failure threshold exceeds configured limits.

This prevents a single file from blocking repository synchronization.

# 9. Observability & Performance Instrumentation

## 9.1 Core Design Principle

Every PR review is treated as a **traceable execution pipeline**.

Each stage emits structured telemetry so the system can answer:

> “Where did time go?”  
> “What is slow?”  
> “What breaks under load?”

---

## 9.2 Global Trace Model

Each PR receives a unique trace identifier:

```text
trace_id = pr_{repo}_{pr_number}_{timestamp}

This trace ID is propagated across all system components:

- GitHub Action  
- Backend API  
- Qdrant retrieval layer  
- Embedding service  
- LLM inference engine  
- GitHub callback handler  
```

---

## 9.3 Pipeline 

Each stage records:

```text
start_time
end_time
latency_ms
status
metadata
```
## Defined Stages

| Stage | Description |
|------|-------------|
| webhook_ingestion | GitHub → Azure entry point |
| diff_extraction | extraction of modified files and patches |
| file_fetch | retrieval of raw file content from GitHub |
| chunking | AST / markdown / config segmentation |
| embedding | vector generation via nomic-embed-text |
| vector_search | Qdrant similarity retrieval |
| prompt_building | assembly of structured LLM input |
| llm_inference | generation via Phi-4-mini-instruct |
| post_processing | formatting of review output |
| github_callback | posting review to pull request |

---

## 9.5 System Metrics

### End-to-End Latency
- `end_to_end_latency_ms`

### Stage-Level Latency
- `ingestion_latency_ms`
- `retrieval_latency_ms`
- `embedding_latency_ms`
- `inference_latency_ms`

### Retrieval Metrics
- `top_k_relevance_score`
- `context_token_utilization_ratio`

### System Health Metrics
- `qdrant_hit_rate`
- `embedding_failure_rate`
- `llm_timeout_rate`
- `retry_count_per_stage`

### Operational Metrics
- `api_calls_per_pr`
- `tokens_processed`
- `embedding_calls`
- `llm_tokens_generated`

---

## 9.6 Logging Format Standard

All system logs conform to a structured JSON schema:

```json
{
  "trace_id": "pr_repo_42_1700000000",
  "stage": "embedding",
  "status": "success",
  "latency_ms": 142,
  "metadata": {
    "chunks_processed": 12,
    "model": "nomic-embed-text"
  }
}
```

## 9.7 Storage Strategy

All logs are persisted using local filesystem-based structured logging.

### Directory Structure
/logs/
webhook_ingestion.json
diff_extraction.json
file_fetch.json
chunking.json
embedding.json
vector_search.json
prompt_building.json
llm_inference.json
post_processing.json
github_callback.json

## 9.8 Performance Evaluation

The system supports deterministic performance evaluation across PR executions.

### System-Level Outputs

- average end-to-end latency  
- p95 latency distribution  
- stage-wise latency contribution  
- system bottleneck identification  
- throughput per PR size category  

---

## 9.9 Latency Optimization Loop

The system operates as a continuous optimization loop:

1. PR review execution is triggered  
2. Telemetry is recorded for all pipeline stages  
3. Latency distribution is computed per stage  
4. Bottleneck stage is identified  
5. Optimizations are applied at the identified stage  
6. Benchmark execution is repeated  
7. Performance deltas are recorded  

---

## 9.10 Benchmark Dataset

A fixed evaluation dataset is used for system consistency:

- 10 small pull requests (< 5 files)  
- 5 medium pull requests (10–20 files)  
- 2 large pull requests (> 50 files)  

### Usage

This dataset is used for:

- latency comparison  
- regression detection  
- retrieval consistency validation

---

# 10. Execution Phases

## Overview

The system is built in six sequential phases. No phase begins until the previous one is verified working end-to-end. Each phase produces a testable, runnable artifact. Human sign-off is required before advancing.

---

## Pre-Phase: Architecture Decisions Required Before Any Code Is Written

The following conflicts between the current deployed infrastructure and the specification must be resolved before development begins:

| Conflict | Spec Says | Current State | Decision Needed |
|----------|-----------|---------------|-----------------|
| Embedding model | `nomic-embed-text` running locally inside App Service memory | No embedding model deployed anywhere | Run nomic-embed-text locally on App Service, OR deploy an Azure-hosted embedding model |
| Generation model | `Phi-4-mini-instruct` running locally on Azure Container Instance | `Phi-4-mini-reasoning` deployed on Azure AI Services | Use the deployed reasoning model via API, OR provision ACI with instruct model |
| Compute for Backend API | Standard Azure App Service (holds ML model in memory) | Azure Function App (FlexConsumption) — cannot hold models in memory | Upgrade to App Service plan, OR use API-based approach for embeddings and drop local model requirement |
| Qdrant collection | Single collection `global_codebase_memory` with `repo_id` index | `pattern_buddy_rules` collection (3 old test vectors, 3072-dim, no repo_id index) | Create new collection with correct schema and delete or ignore old one |
| PHI_ENDPOINT in `.env` | `cognitiveservices.azure.com` format | Wrong URL — returns 404 | Fix to: `https://dpant26-4965-resource.cognitiveservices.azure.com/openai/deployments/Phi-4-mini-reasoning/chat/completions?api-version=2024-12-01-preview` |

---

# Project Implementation Roadmap

## Phase 0 — Infrastructure Provisioning & Environment Setup
**Goal:** All services connected, credentials correct, and Qdrant schema strictly aligned with the locked architecture. Nothing moves to Phase 1 until a smoke-test script passes.

### Tasks
**0.1 — Provision the Segregated Compute Architecture**
* **Backend API (Azure App Service — B2 plan):** 2 vCores, 3.5 GB RAM. `nomic-embed-text` loaded via **ONNX Runtime** — confirmed loader. Do not substitute PyTorch or sentence-transformers; their framework overhead alone reaches 900 MB–1.2 GB, leaving insufficient headroom on a 3.5 GB ceiling under concurrent load. ONNX Runtime keeps total process footprint (model + Python + FastAPI workers) at ~800 MB–1 GB.
* **Critical Configuration:** Enable "Always On" in App Service settings. Without this, the container idles after inactivity, `nomic-embed-text` is evicted from memory, and the next webhook hits a cold start long enough to exceed GitHub's 10-second timeout before receiving the required 202.
* **Inference Engine — BUDGET DECISION (2026-06-28):** ACI dropped due to $100 Azure for Students constraint (ACI 2 vCPU/6 GB = ~$166/month alone). Instead, use the **Azure AI Foundry serverless endpoint** (`PHI_ENDPOINT` in `.env`) for Phi-4-mini-reasoning. Pay-per-token cost is ~$0.001/PR review. Smoke test confirmed the endpoint is live and returns valid output. The Backend API POSTs directly to this endpoint instead of an internal ACI.
* Add `GITHUB_PAT` (personal access token with `repo` scope) to the Backend API App Settings for file fetching and PR comment posting.
* **Mandatory Security Constraint:** Add a securely generated `WEBHOOK_SECRET` to Azure App Settings and GitHub Actions Secrets to prevent unauthenticated execution of the backend API.

**0.2 — Provision Qdrant Collection (`global_codebase_memory`)**
* Create a single collection, using Cosine distance.
* Vector size must match `nomic-embed-text` output dimensions.
* **Mandatory:** Index `repo_id` with `is_tenant: true` for O(1) multitenancy routing.
* Hard delete the legacy `pattern_buddy_rules` collection to prevent confusion.

**0.3 — End-to-End Smoke Test**
Write and run a local script that:
1. Embeds a dummy string locally via `nomic-embed-text` and confirms the dimension matches the collection.
2. Upserts one vector to `global_codebase_memory` with the full payload schema.
3. Queries back by `repo_id` and confirms retrieval.
4. Sends a POST request to the isolated ACI Inference Engine with a prompt and confirms a Markdown response.
5. Prints pass/fail for each step.

**Exit Criteria:** Smoke test passes all checks, proving the App Service can successfully route to the ACI.

---

## Phase 1 — Polymorphic Ingestion Pipeline
**Goal:** Given a local clone of any repository, produce a fully populated `global_codebase_memory` collection correctly partitioned by `repo_id`.

### Tasks
**1.1 — Polymorphic File Router**
* Classify every file by extension into Track A (`.py`), Track B (`.md`, `.txt`), or Track C (`.toml`, `.json`, `.yaml`).
* Skip: binary files, lockfiles (`poetry.lock`, `package-lock.json`), generated files, and the `.git/` directory.

**1.2 — Track A: AST Chunker (Source Code)**
* **Parser:** `tree-sitter-python`.
* **Chunk boundary:** Function/Class definition, max ~800 tokens. **Overlap:** 0.
* **Oversized entity:** Split at internal logical junctures; prepend parent signature to all sub-chunks.
* **Test file mapping:** Heuristic filename match (e.g., `src/auth.py` → `tests/test_auth.py`).

**1.3 — Track B & C: Prose & Config Chunker**
* **Prose:** Split at markdown headers. Target size: 300–600 tokens. **Overlap:** ~100 tokens (only if section > 800 tokens).
* **Config:** Flatten to declarative English sentences.

**1.4 — Embedding & Upsert**
* Embed each chunk using `nomic-embed-text` (running locally via fastembed/ONNX — Phase 1 is a standalone ingestion script, not an API).
* Build full payload per schema (`repo_id`, `file_path`, `commit_hash`, etc.) and upsert to Qdrant.

**Exit Criteria:** Running the ingestion script on a 50-file Python repository populates Qdrant. A test query returns relevant chunks with a score ≥ 0.75.

---

## Phase 2 — Backend API (Orchestrator)
**Goal:** A running Azure App Service that receives a PR trigger, fetches files, queries Qdrant, calls the Azure AI Foundry inference endpoint, and posts a review comment.

### Tasks
**2.1 — Webhook Ingestion Endpoint (`POST /review`)**
* Accepts trigger payload from GitHub Actions.
* **Security Gate:** Validate the `x-hub-signature-256` header using the `WEBHOOK_SECRET`. Reject unauthorized requests with `401 Unauthorized` before proceeding.
* **Critical:** Immediately returns `HTTP 202 Accepted` to close CI connection.
* Spawns a background asynchronous task for processing.

**2.2 — Diff Extraction & Concurrent File Fetching**
* Parse changed files. Exclude skipped extensions.
* **Defensive Threshold:** If changed files > 50, skip file fetching and Qdrant retrieval; fall back to commit-message-only summary.
* Concurrently fetch raw file blobs via `asyncio` + `PAT`.

**2.3 — Retrieval & Prompt Assembly**
* Enforce `must: [{ key: "repo_id", ... }]` filter on Qdrant query.
* **Top-K: 5 chunks** (fixed from earlier conflict — at 800 tokens/chunk, 10–15 chunks alone exceeds Phi-4-mini's 8,192-token context limit before the diff is included).
* Assemble `<system>`, `<repository_context>`, and `<pull_request_diff>` XML prompt.
* **Truncation Priority:** Enforce a **≤7,000 token cap** (fixed from earlier conflict — Phi-4-mini context limit is 8,192 tokens; 7,000 reserves headroom for generation output). If breached, strictly preserve the System and Diff blocks; drop Qdrant chunks starting from the lowest cosine similarity score.

**2.4 — Inference Handoff & GitHub Callback**
* **Network Handoff:** POST the assembled prompt to `PHI_ENDPOINT` (Azure AI Foundry serverless — Phi-4-mini-reasoning). No ACI involved. Budget decision recorded in Phase 0.1.
* Await the response, then POST the structured Markdown review as a PR comment via GitHub API using `PAT`.

**Exit Criteria:** Submitting a real PR causes a structured review comment to appear within 90 seconds. Unauthorized POST requests are rejected instantly.

---

## Phase 3 — GitHub Actions Trigger
**Goal:** Every new PR automatically fires the review pipeline statelessly.

### Tasks
**3.1 — Trigger Workflow**
* Create `.github/workflows/pr_review.yml` (Triggers: `pull_request` -> `opened`, `synchronize`).
* **Concurrency Lock:** Must include `concurrency: group: ${{ github.ref }}` to cancel previous runs and prevent webhook spam on rapid pushes.
* **Runner:** `ubuntu-latest`. No code checkout.

**3.2 — Payload Construction & POST**
* Query GitHub API for changed files.
* Generate an HMAC SHA-256 signature of the payload using the `WEBHOOK_SECRET` stored in GitHub Actions Secrets.
* POST lightweight pointer payload (`repo_id`, `base_sha`, `head_sha`, `changed_files[]`) to `BACKEND_API_URL` with the generated signature in the headers.
* Assert `HTTP 202` response.

**Exit Criteria:** Opening a PR triggers the webhook. The GitHub Action completes in under 10 seconds. Rapidly pushing 3 commits only results in 1 final review.

---

## Phase 4 — Continuous Sync (Update Loop)
**Goal:** Maintain vector state synchronization on PR merges without full re-ingestion.

### Tasks
**4.1 — Merge Webhook (`POST /sync`)**
* **Security Gate:** Validate the `x-hub-signature-256` header using the `WEBHOOK_SECRET`. Reject unauthorized requests.
* Validates `action == "closed"` AND `merged == true`.
* Returns `HTTP 202 Accepted` immediately.

**4.2 — Delete-Then-Upsert Execution**
* **Blind Deletion:** Issue `DELETE` to Qdrant matching `repo_id` + `file_path` of all changed/deleted files.
* **Fetch & Upsert:** For modified/added files, fetch blob, embed via Backend API, and upsert with new `commit_hash`.

**4.3 — Cron Fallback (Weekly Rebuild)**
* **Schedule:** Sundays at 3:00 AM UTC (Off-peak).
* **Execution:** Issue a global `DELETE` for all vectors matching the `repo_id`. Instead of using brittle git clone commands, fetch the repository via the GitHub API as a ZIP archive (`GET /repos/{owner}/{repo}/zipball/main`), extract into memory/temp directory, re-ingest, and aggressively wipe the local disk to prevent App Service storage exhaustion.

**Exit Criteria:** Merging a PR updates affected vectors. Renames correctly swap paths.

---

## Phase 5 — Observability
**Goal:** Ensure PR reviews are traceable and bottlenecks are measurable.

### Tasks
**5.1 — Trace ID & Structured Logging**
* Generate `trace_id` at webhook ingress.
* Emit JSON logs for: `diff_fetch`, `embedding`, `qdrant_retrieval`, and `inference_handoff`.

**5.2 — System Metrics Collection**
* Track App Service RAM utilization.
* Track `end_to_end_latency_ms` and ACI `inference_latency_ms`.

**Exit Criteria:** Every review produces a complete trace log.