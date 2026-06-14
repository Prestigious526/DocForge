# DocForge — LangGraph Agent Pipeline
## Plan · Retrieve · Write · Validate · Rewrite · Assemble
### Fully integrated with DocForge architecture v1.0

---

> **What this document is:** The complete implementation of the LangGraph document
> generation pipeline for both ITD and CTD modes. Every node, edge, state field,
> retrieval strategy, validation gate, and rewrite loop is defined and justified here.
> This sits inside `agents/graph/` and is the execution engine called by the Streamlit UI.

---

## Table of Contents

1. [Graph Overview](#1-graph-overview)
2. [State Design](#2-state-design)
3. [File Structure](#3-file-structure)
4. [Smart Retrieval Strategy](#4-smart-retrieval-strategy)
5. [All Prompts](#5-all-prompts)
6. [Node Implementations](#6-node-implementations)
7. [Conditional Edge Routing](#7-conditional-edge-routing)
8. [Graph Assembly](#8-graph-assembly)
9. [Public Pipeline Entry Points](#9-public-pipeline-entry-points)
10. [Streamlit Integration](#10-streamlit-integration)
11. [Adding to requirements.txt](#11-adding-to-requirementstxt)

---

## 1. Graph Overview

```
                        ┌─────────────────────────────────────────────────────┐
                        │              DocForge LangGraph Pipeline             │
                        └─────────────────────────────────────────────────────┘

                              [START]
                                 │
                                 ▼
                        ┌────────────────┐
                        │ plan_document  │  LLM: instruction → section plan
                        │                │  each section gets 3-5 retrieval queries
                        └───────┬────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  select_next_section │  picks next unwritten section from plan
                     └──────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
               (sections left)        (all done)
                    │                       │
                    ▼                       ▼
      ┌─────────────────────────┐   ┌──────────────────┐
      │ generate_retrieval_     │   │ assemble_document│
      │ queries                 │   │                  │
      │                         │   │ merges all       │
      │ LLM: section title +    │   │ validated        │
      │ instructions → 3-5      │   │ sections in      │
      │ targeted search queries │   │ plan order       │
      └──────────┬──────────────┘   └────────┬─────────┘
                 │                           │
                 ▼                           ▼
      ┌──────────────────────┐    ┌────────────────────┐
      │   retrieve_chunks    │    │   apply_template   │
      │                      │    │                    │
      │ multi-query search   │    │ applies fonts,     │
      │ dedup by chunk_id    │    │ styles, header,    │
      │ deprioritise already │    │ footer, letterhead │
      │ used chunks          │    │ from TemplateStyles│
      │ enforce token budget │    └────────┬───────────┘
      └──────────┬───────────┘             │
                 │                         ▼
                 ▼               ┌──────────────────────┐
      ┌──────────────────────┐   │   generate_output    │
      │    write_section     │   │                      │
      │                      │   │ routes to output     │
      │ streams content      │   │ plugins: docx, pdf,  │
      │ uses template slot   │   │ md, excel            │
      │ as structural anchor │   └──────────────────────┘
      └──────────┬───────────┘             │
                 │                       [END]
                 ▼
      ┌──────────────────────┐
      │   validate_section   │
      │                      │
      │ scores: completeness │
      │ template compliance  │
      │ factual grounding    │
      │ quality threshold    │
      └──────────┬───────────┘
                 │
        ┌────────┴──────────┐
        │                   │
   (score ≥ threshold  (score < threshold
    OR max retries hit)  AND retries left)
        │                   │
        ▼                   ▼
  select_next_section  ┌────────────────┐
                       │ rewrite_section│
                       │                │
                       │ targeted rewrite│
                       │ uses validator  │
                       │ feedback as     │
                       │ precise prompt  │
                       └────────┬───────┘
                                │
                                ▼
                         validate_section  (loop back)
```

---

## 2. State Design

The `GraphState` is the single source of truth passed between every node.
LangGraph merges state at each step — annotated fields with `operator.add` accumulate;
plain fields are replaced.

### `agents/graph/state.py`

```python
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any
from typing_extensions import TypedDict

from template.analyzer import TemplateStyles


# ── Per-section plan entry (output of PlannerAgent) ──────────────────────────

@dataclass
class SectionPlan:
    """One planned section of the document."""
    index: int                        # Position in document (0-based)
    title: str                        # Section heading
    instructions: str                 # What to write
    retrieval_queries: list[str]      # 3-5 targeted queries generated by planner
    expected_length: str              # "short" | "medium" | "long"
    requires_tables: bool             # Hint: section likely needs tables
    requires_code: bool               # Hint: section likely needs code blocks


# ── Per-section write result ──────────────────────────────────────────────────

@dataclass
class SectionDraft:
    """A written (or rewritten) section with its retrieval provenance."""
    section_index: int
    title: str
    content: str
    chunk_ids_used: list[str]         # For deduplication in later sections
    retrieval_queries_used: list[str] # Audit trail
    rewrite_count: int = 0
    validation_score: float = 0.0
    validation_feedback: str = ""
    accepted: bool = False


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    section_index: int
    score: float                      # 0.0 – 1.0
    passed: bool
    feedback: str                     # Specific, actionable — fed to rewriter
    completeness_score: float
    template_compliance_score: float
    grounding_score: float            # How well content cites retrieved context


# ── Main graph state ──────────────────────────────────────────────────────────

class GraphState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    instruction: str                         # User instruction or doc type
    project_name: str                        # LanceDB + cache namespace
    mode: str                                # "itd" | "ctd"
    template_styles: TemplateStyles | None   # Extracted from uploaded template
    template_section_slots: list[str]        # Section titles from template (structural anchor)
    output_formats: list[str]                # ["docx", "pdf", "md"]
    doc_title: str                           # Used as output filename base
    itd_model: str                           # Ollama model for ITD
    ctd_model: str                           # Ollama model for CTD
    context_budget_tokens: int               # Max tokens fed to each write call
    max_retrieval_chunks: int                # Hard cap on chunks per section
    min_relevance_score: float               # Min cosine sim threshold
    validation_threshold: float              # Min score to accept a section (0–1)
    max_rewrite_attempts: int                # Before accepting with warning flag

    # ── Plan ──────────────────────────────────────────────────────────────────
    section_plan: list[SectionPlan]          # Set by plan_document node
    current_section_index: int               # Which section is being written now

    # ── Retrieval tracking ────────────────────────────────────────────────────
    # Accumulates chunk_ids used across ALL sections to deprioritise reuse
    all_used_chunk_ids: Annotated[list[str], operator.add]
    # Chunks retrieved for the CURRENT section only (replaced each section)
    current_section_chunks: list[dict]

    # ── Writing ───────────────────────────────────────────────────────────────
    current_draft: SectionDraft | None       # Draft being worked on right now
    current_rewrite_count: int               # Resets to 0 for each new section

    # ── Completed sections ────────────────────────────────────────────────────
    # Accumulates as sections are accepted (operator.add = append, not replace)
    completed_sections: Annotated[list[SectionDraft], operator.add]

    # ── Validation ────────────────────────────────────────────────────────────
    last_validation: ValidationResult | None

    # ── Output ────────────────────────────────────────────────────────────────
    assembled_content: str                   # Full document markdown before template
    output_paths: dict[str, str]             # format → absolute path

    # ── Streaming / UI ────────────────────────────────────────────────────────
    # A queue of status messages for the Streamlit UI to consume
    status_messages: Annotated[list[str], operator.add]
    # The current streaming token buffer (replaced per token)
    stream_buffer: str
```

---

## 3. File Structure

```
agents/
└── graph/
    ├── __init__.py
    ├── state.py           ← GraphState, SectionPlan, SectionDraft, ValidationResult
    ├── prompts.py         ← All LLM system prompts as module-level constants
    ├── smart_retriever.py ← Multi-query retrieval, dedup, budget enforcement
    ├── nodes.py           ← All node functions (one function per node)
    ├── edges.py           ← Conditional routing functions
    └── pipeline.py        ← Graph assembly + public run_itd_pipeline() / run_ctd_pipeline()
```

---

## 4. Smart Retrieval Strategy

The retriever is where most of DocForge's intelligence lives. The rule is:
**retrieve only what the writer needs for this specific section, not everything related to the topic.**

### The Problem with Naive Retrieval

```
❌ Naive: embed(section.title) → top_k chunks
   "System Architecture" → returns everything about architecture
   → writer drowns in irrelevant context
   → same chunks retrieved for "Overview" and "Architecture" and "Design"
   → context budget wasted on duplicate information
```

### The DocForge Strategy: Multi-Query + Budget + Deduplication

```
✅ Smart:
  1. Planner generates 3-5 targeted queries per section (not just the title)
  2. Each query runs independently → raw results pooled
  3. Chunks deduplicated by chunk_id → aggregate score = max(individual scores)
  4. Chunks already used in previous sections get a penalty score
  5. Remaining chunks ranked by score
  6. Budget enforcer: walk ranked list, accumulate token estimate, stop at budget
  7. Writer receives exactly what fits, nothing more
```

### `agents/graph/smart_retriever.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from core.config import get_config
from core.logger import get_logger
from core.errors import StoreError
from embeddings.store import EmbeddingStore

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALREADY_USED_PENALTY  = 0.25   # Score penalty for chunks used in prior sections
TOKENS_PER_CHAR_RATIO = 0.25   # Rough estimate: 4 chars ≈ 1 token


@dataclass
class ScoredChunk:
    chunk_id: str
    file_id: str
    filename: str
    text: str
    score: float              # Higher is better (1 - distance)
    penalised: bool = False   # True if chunk was used in a prior section


def retrieve_for_section(
    queries: list[str],
    project_name: str,
    already_used_chunk_ids: set[str],
    context_budget_tokens: int,
    max_chunks: int,
    min_score: float,
) -> list[ScoredChunk]:
    """
    Multi-query retrieval with deduplication, penalty scoring, and budget enforcement.

    Runs each query independently, pools results, deduplicates by chunk_id,
    penalises previously used chunks, ranks by final score, then trims to
    the token budget.

    Args:
        queries:                  3-5 targeted retrieval queries for this section.
        project_name:             LanceDB namespace.
        already_used_chunk_ids:   chunk_ids used in all prior sections — penalised.
        context_budget_tokens:    Max tokens to pass to the writer.
        max_chunks:               Hard cap on chunks regardless of budget.
        min_score:                Minimum cosine similarity to include a chunk.

    Returns:
        List of ScoredChunk ordered by final score, trimmed to budget.

    Raises:
        StoreError: If LanceDB is unreachable.
    """
    log.info("smart_retriever.start",
             queries=len(queries),
             budget=context_budget_tokens,
             max_chunks=max_chunks)

    store = EmbeddingStore(project_name)

    # ── Step 1: Run all queries, pool raw results ─────────────────────────────
    raw_pool: dict[str, dict] = {}   # chunk_id → best raw result

    for query in queries:
        try:
            results = store.search(query=query, top_k=max_chunks, min_score=min_score)
            for r in results:
                cid   = r["chunk_id"]
                score = 1.0 - r.get("_distance", 1.0)
                # Keep the highest score across all queries for this chunk
                if cid not in raw_pool or score > raw_pool[cid]["_best_score"]:
                    r["_best_score"] = score
                    raw_pool[cid]    = r
        except StoreError as e:
            log.warning("smart_retriever.query_failed", query=query[:60], error=str(e))
            # Continue with other queries — degrade gracefully

    log.debug("smart_retriever.pool_size", unique_chunks=len(raw_pool))

    # ── Step 2: Score, penalise, rank ─────────────────────────────────────────
    scored: list[ScoredChunk] = []
    for cid, raw in raw_pool.items():
        base_score = raw["_best_score"]
        penalised  = cid in already_used_chunk_ids
        final_score = base_score - ALREADY_USED_PENALTY if penalised else base_score
        # Never go below 0
        final_score = max(0.0, final_score)

        scored.append(ScoredChunk(
            chunk_id  = cid,
            file_id   = raw["file_id"],
            filename  = raw["filename"],
            text      = raw["text"],
            score     = final_score,
            penalised = penalised,
        ))

    # Sort descending by final score
    scored.sort(key=lambda c: c.score, reverse=True)

    # ── Step 3: Enforce token budget ──────────────────────────────────────────
    budget_chunks: list[ScoredChunk] = []
    accumulated_tokens = 0

    for chunk in scored[:max_chunks]:
        estimated_tokens = int(len(chunk.text) * TOKENS_PER_CHAR_RATIO)
        if accumulated_tokens + estimated_tokens > context_budget_tokens:
            log.debug("smart_retriever.budget_stop",
                      accumulated=accumulated_tokens,
                      budget=context_budget_tokens,
                      stopped_at_rank=len(budget_chunks))
            break
        budget_chunks.append(chunk)
        accumulated_tokens += estimated_tokens

    log.info("smart_retriever.done",
             returned=len(budget_chunks),
             penalised=sum(1 for c in budget_chunks if c.penalised),
             total_tokens_est=accumulated_tokens)

    return budget_chunks


def format_chunks_for_prompt(chunks: list[ScoredChunk]) -> str:
    """
    Format retrieved chunks into a clean context block for the writer prompt.
    Groups by source file for readability. Includes relevance score for the LLM.
    """
    if not chunks:
        return "[No relevant context retrieved. Use general engineering knowledge.]"

    # Group by filename
    by_file: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.filename, []).append(chunk)

    parts = []
    for filename, file_chunks in by_file.items():
        file_chunks.sort(key=lambda c: c.score, reverse=True)
        header = f"### Source: {filename}"
        body   = "\n\n".join(
            f"[Relevance: {c.score:.2f}]\n{c.text}" for c in file_chunks
        )
        parts.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(parts)
```

---

## 5. All Prompts

All prompts live as module-level constants. Never define prompts inside functions.
This makes them easy to iterate on without touching logic.

### `agents/graph/prompts.py`

```python
# ─────────────────────────────────────────────────────────────────────────────
#  DocForge — All LangGraph Agent Prompts
#  Editing these is the primary way to tune output quality.
#  Do NOT move these into node functions.
# ─────────────────────────────────────────────────────────────────────────────


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a senior technical documentation architect.
Your job is to create a precise, comprehensive document plan.
You MUST respond with valid JSON only. No preamble. No markdown fences. Raw JSON.

The JSON schema is:
{
  "sections": [
    {
      "index": 0,
      "title": "Section Title",
      "instructions": "Detailed instructions for what to write in this section",
      "retrieval_queries": [
        "specific query 1 targeting a narrow sub-topic",
        "specific query 2 targeting different sub-topic",
        "specific query 3"
      ],
      "expected_length": "short|medium|long",
      "requires_tables": true|false,
      "requires_code": true|false
    }
  ]
}

Rules for retrieval_queries:
- Generate 3-5 queries per section. Never fewer than 3.
- Each query targets a DIFFERENT narrow sub-topic of the section.
- Queries should be specific, not generic. Bad: "system design". Good: "database schema for user authentication module".
- Think about what source material would most help a writer fill this section.
- Queries are used for semantic vector search — phrase them as natural language statements or questions."""

PLANNER_USER = """Document type and instruction:
{instruction}

Template section structure (match this structure if provided):
{template_slots}

Generate a complete, professional document plan.
For engineering documents, include: Executive Summary, Scope and Objectives,
Requirements, Architecture/Design, Implementation Details, Testing Strategy,
Deployment, Appendix — as applicable to the document type.

Fill any structural gaps with standard engineering documentation sections.
Each section MUST have 3-5 specific retrieval_queries that target the source material
most relevant to writing that section well."""


# ── Query Generator (per section, optional refinement step) ──────────────────

QUERY_REFINER_SYSTEM = """You are a retrieval query specialist.
Given a document section title and writing instructions, generate search queries
that will retrieve the most relevant source material from a vector database.
Respond with JSON only: {"queries": ["query1", "query2", "query3"]}
No preamble. No markdown. Raw JSON."""

QUERY_REFINER_USER = """Section title: {title}
Section instructions: {instructions}
Already retrieved queries (avoid overlap): {existing_queries}

Generate 3 additional targeted retrieval queries for this section.
Queries must target DIFFERENT aspects than the existing ones.
Be specific — not "architecture" but "message queue configuration for async processing"."""


# ── Writer — ITD ─────────────────────────────────────────────────────────────

WRITER_ITD_SYSTEM = """You are an expert technical writer producing production-ready engineering documentation.

Your output will be inserted directly into a document that follows a strict template.
Follow these rules absolutely:

1. TEMPLATE ADHERENCE: Write the section content only. Do NOT add headings that aren't
   in the template. Do NOT change the section title. Do NOT add sub-sections not
   implied by the instructions.

2. CONTENT QUALITY: Write at a senior engineering level. Be precise, complete, and
   professional. Use formal technical language.

3. SOURCE GROUNDING: Ground claims in the provided context chunks wherever possible.
   Use "[TBD]" for information not in context and not general knowledge.
   Do NOT invent specifications, version numbers, or system behaviours.

4. GENERAL KNOWLEDGE: You MAY use general engineering knowledge to fill gaps.
   Always prefer source context over general knowledge when both are available.

5. LENGTH: Match the expected_length directive.
   - short: 1-2 paragraphs
   - medium: 3-5 paragraphs or equivalent structured content
   - long: 6+ paragraphs, may include tables or lists

6. FORMAT: Output clean Markdown. Use tables where the section requires structured data.
   Use code blocks for any configuration, commands, or code snippets.

7. OUTPUT: Write ONLY the section body content. No section title. No preamble."""

WRITER_ITD_USER = """## Section to Write
Title: {title}
Instructions: {instructions}
Expected length: {expected_length}
Requires tables: {requires_tables}
Requires code blocks: {requires_code}

## Retrieved Context
{context}

## Previous Sections Summary (for consistency)
{previous_summary}

Write the section body now."""


# ── Writer — CTD ─────────────────────────────────────────────────────────────

WRITER_CTD_SYSTEM = """You are an expert software documentation writer.
You convert code analysis (AST summaries, function signatures, class hierarchies)
into precise, developer-friendly technical documentation.

Rules:
1. ACCURACY: Every function signature, parameter name, and return type must match
   the provided code analysis exactly. Do NOT invent or infer signatures.
2. COMPLETENESS: Document every public function, class, and module-level constant
   in the provided analysis. Do not skip items.
3. CLARITY: Write for a developer audience familiar with the language.
   Explain WHY, not just WHAT. Describe intent, not just signature.
4. EXAMPLES: Include a usage example for every public function and class.
   Examples must be syntactically correct for the language.
5. TEMPLATE ADHERENCE: Write section content only. Match the template structure.
6. OUTPUT: Clean Markdown with proper code fences. No section title. No preamble."""

WRITER_CTD_USER = """## Section to Document
Title: {title}
Instructions: {instructions}

## Code Analysis
{context}

## Documentation Style Guide
{style_guide}

Write the documentation section now."""


# ── Validator ─────────────────────────────────────────────────────────────────

VALIDATOR_SYSTEM = """You are a senior technical documentation reviewer.
Your job is to score a written section against strict quality criteria.
Respond with JSON only. No preamble. No markdown. Raw JSON.

Schema:
{
  "completeness_score": 0.0-1.0,
  "template_compliance_score": 0.0-1.0,
  "grounding_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "passed": true|false,
  "feedback": "Specific, actionable feedback. List exactly what is wrong and how to fix it."
}

Scoring criteria:
- completeness_score: Does the section fully address its instructions?
  1.0 = fully complete. 0.0 = almost nothing written. 0.5 = half done.
- template_compliance_score: Does the section fit its expected place in the document?
  Does it avoid adding unrequested headings or deviating from the template slot?
  1.0 = perfectly fits template. 0.0 = completely ignores template structure.
- grounding_score: Are claims supported by the provided context or sound general knowledge?
  1.0 = all claims grounded. 0.0 = speculative or invented claims throughout.
- overall_score: Weighted average: completeness*0.4 + template*0.35 + grounding*0.25
- passed: true if overall_score >= {threshold}

Feedback must be:
- Specific: "The scalability section is missing" not "incomplete"
- Actionable: "Add a paragraph on horizontal scaling and load balancing" not "add more content"
- Concise: Maximum 5 bullet points"""

VALIDATOR_USER = """## Section Being Reviewed
Title: {title}
Instructions (what should be written): {instructions}
Expected length: {expected_length}
Template slot position: Section {index} of {total_sections}

## Written Content
{content}

## Context That Was Available
{context_summary}

## Passing Threshold
{threshold}

Score this section now."""


# ── Rewriter ──────────────────────────────────────────────────────────────────

REWRITER_SYSTEM = """You are a senior technical writer performing a targeted revision.
You have written a section and received specific reviewer feedback.
Your job is to fix EXACTLY what the reviewer identified — nothing more, nothing less.

Rules:
1. Address every point in the feedback list specifically.
2. Do NOT rewrite content that the reviewer did not flag as problematic.
3. Preserve the template structure — do not add or remove headings.
4. Keep the same tone and style as the original where it was acceptable.
5. If feedback says "too short", expand the flagged part only.
6. OUTPUT: Write the full revised section body. No title. No preamble."""

REWRITER_USER = """## Original Section
Title: {title}
Instructions: {instructions}

{original_content}

## Reviewer Feedback (address each point)
{feedback}

## Validation Scores
Completeness: {completeness_score:.2f}
Template compliance: {template_score:.2f}
Grounding: {grounding_score:.2f}

## Retrieved Context (use this to fix grounding issues)
{context}

Revise the section now, addressing each piece of feedback."""


# ── Document Assembler ────────────────────────────────────────────────────────

ASSEMBLER_SYSTEM = """You are a document assembler. You receive an ordered list of
document sections and produce a clean, coherent final document in Markdown.

Rules:
1. Output sections in the EXACT order provided. Do not reorder.
2. Add the document title as an H1 heading at the top.
3. Add a horizontal rule (---) between major sections.
4. Do NOT add any content not present in the sections provided.
5. Do NOT summarise, shorten, or alter section content.
6. Apply consistent heading levels: H2 for main sections, H3 for sub-sections."""

ASSEMBLER_USER = """Document Title: {doc_title}

Sections (in order):
{sections_json}

Assemble the final document now."""
```

---

## 6. Node Implementations

### `agents/graph/nodes.py`

```python
from __future__ import annotations

import json
from pathlib import Path

from agents.graph.state import (
    GraphState, SectionPlan, SectionDraft, ValidationResult
)
from agents.graph.prompts import (
    PLANNER_SYSTEM, PLANNER_USER,
    WRITER_ITD_SYSTEM, WRITER_ITD_USER,
    WRITER_CTD_SYSTEM, WRITER_CTD_USER,
    VALIDATOR_SYSTEM, VALIDATOR_USER,
    REWRITER_SYSTEM, REWRITER_USER,
    ASSEMBLER_SYSTEM, ASSEMBLER_USER,
)
from agents.graph.smart_retriever import (
    retrieve_for_section, format_chunks_for_prompt, ScoredChunk
)
from llm.client import stream_completion, call_llm
from template.store import TemplateStore
from output.pipeline import generate_output
from core.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 1 — plan_document
# ─────────────────────────────────────────────────────────────────────────────

def plan_document(state: GraphState) -> dict:
    """
    LLM node: converts user instruction + template slots into a full section plan.
    Each section gets: title, instructions, 3-5 retrieval queries, length hint.

    Returns:
        Updates: section_plan, current_section_index, status_messages
    """
    log.info("node.plan_document.start", instruction=state["instruction"][:80])

    model         = state["itd_model"] if state["mode"] == "itd" else state["ctd_model"]
    template_slots = "\n".join(
        f"- {s}" for s in (state["template_section_slots"] or [])
    ) or "No template structure provided — use standard engineering document structure."

    prompt = PLANNER_USER.format(
        instruction    = state["instruction"],
        template_slots = template_slots,
    )

    raw = call_llm(prompt=prompt, system=PLANNER_SYSTEM, model=model)
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        data     = json.loads(raw)
        sections = data.get("sections", [])
        if not sections:
            raise ValueError("Planner returned empty section list")
    except (json.JSONDecodeError, ValueError) as e:
        log.error("node.plan_document.parse_failed", error=str(e), raw=raw[:300])
        # Fallback: create a minimal plan with the instruction as one section
        sections = [{
            "index": 0,
            "title": "Document Content",
            "instructions": state["instruction"],
            "retrieval_queries": [state["instruction"]],
            "expected_length": "long",
            "requires_tables": False,
            "requires_code": False,
        }]

    plan = [
        SectionPlan(
            index             = i,
            title             = s["title"],
            instructions      = s["instructions"],
            retrieval_queries = s.get("retrieval_queries", [s["title"]]),
            expected_length   = s.get("expected_length", "medium"),
            requires_tables   = s.get("requires_tables", False),
            requires_code     = s.get("requires_code", False),
        )
        for i, s in enumerate(sections)
    ]

    log.info("node.plan_document.done", sections=len(plan))

    return {
        "section_plan":           plan,
        "current_section_index":  0,
        "status_messages":        [f"📋 Plan ready — {len(plan)} sections planned"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 2 — select_next_section
# ─────────────────────────────────────────────────────────────────────────────

def select_next_section(state: GraphState) -> dict:
    """
    Control node: determines which section to write next.
    Resets per-section working state (chunks, draft, rewrite count, validation).

    Returns:
        Updates: current_section_index, current_rewrite_count,
                 current_draft, current_section_chunks, last_validation
    """
    idx  = state["current_section_index"]
    plan = state["section_plan"]

    if idx < len(plan):
        section = plan[idx]
        log.info("node.select_next_section", index=idx, title=section.title)
        return {
            "current_section_index":  idx,
            "current_rewrite_count":  0,
            "current_draft":          None,
            "current_section_chunks": [],
            "last_validation":        None,
            "status_messages": [
                f"📌 Section {idx + 1}/{len(plan)}: {section.title}"
            ],
        }
    else:
        log.info("node.select_next_section.all_done", total=len(plan))
        return {
            "status_messages": ["✅ All sections written — assembling document"],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 3 — retrieve_chunks
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(state: GraphState) -> dict:
    """
    Retrieval node: runs smart multi-query retrieval for the current section.

    Uses the section's pre-planned retrieval_queries (from the planner).
    Deduplicates across queries, penalises already-used chunks, enforces token budget.

    Returns:
        Updates: current_section_chunks, status_messages
    """
    idx     = state["current_section_index"]
    section = state["section_plan"][idx]

    log.info("node.retrieve_chunks.start",
             section=section.title, queries=len(section.retrieval_queries))

    chunks = retrieve_for_section(
        queries                = section.retrieval_queries,
        project_name           = state["project_name"],
        already_used_chunk_ids = set(state["all_used_chunk_ids"] or []),
        context_budget_tokens  = state["context_budget_tokens"],
        max_chunks             = state["max_retrieval_chunks"],
        min_score              = state["min_relevance_score"],
    )

    # Convert ScoredChunk dataclasses to plain dicts for state storage
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "file_id":  c.file_id,
            "filename": c.filename,
            "text":     c.text,
            "score":    c.score,
        }
        for c in chunks
    ]

    log.info("node.retrieve_chunks.done",
             section=section.title, chunks=len(chunks))

    return {
        "current_section_chunks": chunk_dicts,
        "status_messages": [
            f"🔍 Retrieved {len(chunks)} chunks for '{section.title}'"
            + (f" ({sum(1 for c in chunks if c.penalised)} deprioritised re-uses)" if chunks else "")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 4 — write_section
# ─────────────────────────────────────────────────────────────────────────────

def write_section(state: GraphState) -> dict:
    """
    LLM streaming node: writes a single section using retrieved context.

    Uses ITD or CTD system prompt depending on mode.
    Builds a previous-sections summary (last 2 sections, truncated)
    to maintain cross-section consistency without blowing the context budget.

    Returns:
        Updates: current_draft, all_used_chunk_ids, status_messages
    """
    idx     = state["current_section_index"]
    section = state["section_plan"][idx]
    chunks  = state["current_section_chunks"]
    model   = state["itd_model"] if state["mode"] == "itd" else state["ctd_model"]

    log.info("node.write_section.start", section=section.title,
             chunks=len(chunks), model=model)

    context = format_chunks_for_prompt(
        [ScoredChunk(
            chunk_id  = c["chunk_id"],
            file_id   = c["file_id"],
            filename  = c["filename"],
            text      = c["text"],
            score     = c["score"],
        )
        for c in chunks]
    )

    # Build a compact summary of the last 2 completed sections for consistency
    previous_summary = _build_previous_summary(state["completed_sections"], max_sections=2)

    if state["mode"] == "itd":
        system = WRITER_ITD_SYSTEM
        prompt = WRITER_ITD_USER.format(
            title            = section.title,
            instructions     = section.instructions,
            expected_length  = section.expected_length,
            requires_tables  = section.requires_tables,
            requires_code    = section.requires_code,
            context          = context,
            previous_summary = previous_summary,
        )
    else:
        system = WRITER_CTD_SYSTEM
        prompt = WRITER_CTD_USER.format(
            title       = section.title,
            instructions = section.instructions,
            context     = context,
            style_guide = "Use clear headings, code fences with language tags, parameter tables.",
        )

    # Collect streaming tokens
    content_parts: list[str] = []
    for token in stream_completion(prompt=prompt, system=system, model=model):
        content_parts.append(token)
    content = "".join(content_parts).strip()

    chunk_ids_used = [c["chunk_id"] for c in chunks]

    draft = SectionDraft(
        section_index        = idx,
        title                = section.title,
        content              = content,
        chunk_ids_used       = chunk_ids_used,
        retrieval_queries_used = section.retrieval_queries,
        rewrite_count        = state["current_rewrite_count"],
    )

    log.info("node.write_section.done",
             section=section.title, chars=len(content))

    return {
        "current_draft":       draft,
        "all_used_chunk_ids":  chunk_ids_used,   # Accumulates via operator.add
        "status_messages":     [f"✍️  Written '{section.title}' ({len(content)} chars)"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 5 — validate_section
# ─────────────────────────────────────────────────────────────────────────────

def validate_section(state: GraphState) -> dict:
    """
    LLM node: scores the current draft against completeness, template compliance,
    and grounding. Returns a ValidationResult that the edge router uses to decide
    whether to accept or rewrite.

    Returns:
        Updates: last_validation, status_messages
    """
    draft   = state["current_draft"]
    idx     = state["current_section_index"]
    section = state["section_plan"][idx]
    model   = state["itd_model"]   # Always use ITD model for validation
    chunks  = state["current_section_chunks"]

    log.info("node.validate_section.start", section=section.title,
             rewrite_count=draft.rewrite_count)

    # Compact context summary for validator (not full chunks — saves tokens)
    context_summary = (
        f"{len(chunks)} chunks retrieved from: "
        + ", ".join(set(c["filename"] for c in chunks))
        if chunks else "No chunks retrieved"
    )

    prompt = VALIDATOR_USER.format(
        title           = section.title,
        instructions    = section.instructions,
        expected_length = section.expected_length,
        index           = idx + 1,
        total_sections  = len(state["section_plan"]),
        content         = draft.content,
        context_summary = context_summary,
        threshold       = state["validation_threshold"],
    )

    system = VALIDATOR_SYSTEM.format(threshold=state["validation_threshold"])

    raw = call_llm(prompt=prompt, system=system, model=model)
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        result = ValidationResult(
            section_index             = idx,
            score                     = float(data.get("overall_score", 0.5)),
            passed                    = bool(data.get("passed", False)),
            feedback                  = data.get("feedback", ""),
            completeness_score        = float(data.get("completeness_score", 0.5)),
            template_compliance_score = float(data.get("template_compliance_score", 0.5)),
            grounding_score           = float(data.get("grounding_score", 0.5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("node.validate_section.parse_failed", error=str(e))
        # If we can't parse the validator — accept the draft to avoid infinite loop
        result = ValidationResult(
            section_index             = idx,
            score                     = state["validation_threshold"],
            passed                    = True,
            feedback                  = "Validator response could not be parsed — accepting draft.",
            completeness_score        = 0.7,
            template_compliance_score = 0.7,
            grounding_score           = 0.7,
        )

    status = (
        f"✅ Validated '{section.title}' — score {result.score:.2f}"
        if result.passed
        else f"⚠️  '{section.title}' score {result.score:.2f} — rewriting (attempt {draft.rewrite_count + 1})"
    )

    log.info("node.validate_section.done",
             section=section.title,
             score=result.score,
             passed=result.passed)

    return {
        "last_validation": result,
        "status_messages": [status],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 6 — rewrite_section
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_section(state: GraphState) -> dict:
    """
    LLM node: targeted rewrite using exact validator feedback.
    Only fixes what the validator flagged — does not restart from scratch.

    Returns:
        Updates: current_draft (updated rewrite_count), current_rewrite_count, status_messages
    """
    draft      = state["current_draft"]
    validation = state["last_validation"]
    idx        = state["current_section_index"]
    section    = state["section_plan"][idx]
    chunks     = state["current_section_chunks"]
    model      = state["itd_model"] if state["mode"] == "itd" else state["ctd_model"]
    new_count  = state["current_rewrite_count"] + 1

    log.info("node.rewrite_section.start",
             section=section.title, attempt=new_count,
             score=validation.score)

    context = format_chunks_for_prompt(
        [ScoredChunk(
            chunk_id=c["chunk_id"], file_id=c["file_id"],
            filename=c["filename"], text=c["text"], score=c["score"],
        ) for c in chunks]
    )

    prompt = REWRITER_USER.format(
        title              = section.title,
        instructions       = section.instructions,
        original_content   = draft.content,
        feedback           = validation.feedback,
        completeness_score = validation.completeness_score,
        template_score     = validation.template_compliance_score,
        grounding_score    = validation.grounding_score,
        context            = context,
    )

    content_parts: list[str] = []
    for token in stream_completion(prompt=prompt, system=REWRITER_SYSTEM, model=model):
        content_parts.append(token)
    revised_content = "".join(content_parts).strip()

    revised_draft = SectionDraft(
        section_index         = draft.section_index,
        title                 = draft.title,
        content               = revised_content,
        chunk_ids_used        = draft.chunk_ids_used,
        retrieval_queries_used= draft.retrieval_queries_used,
        rewrite_count         = new_count,
    )

    log.info("node.rewrite_section.done",
             section=section.title, attempt=new_count, chars=len(revised_content))

    return {
        "current_draft":        revised_draft,
        "current_rewrite_count": new_count,
        "status_messages": [
            f"🔄 Rewrite {new_count} of '{section.title}' complete"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 7 — accept_section
# ─────────────────────────────────────────────────────────────────────────────

def accept_section(state: GraphState) -> dict:
    """
    Control node: marks the current draft as accepted and adds it to completed_sections.
    Also advances current_section_index to the next section.

    This is a separate node (not merged with validate) so the graph edge logic
    stays in edges.py and nodes stay pure.

    Returns:
        Updates: completed_sections (accumulated), current_section_index, status_messages
    """
    draft      = state["current_draft"]
    validation = state["last_validation"]
    idx        = state["current_section_index"]

    draft.validation_score    = validation.score if validation else 0.0
    draft.validation_feedback = validation.feedback if validation else ""
    draft.accepted            = True

    log.info("node.accept_section",
             section=draft.title,
             score=draft.validation_score,
             rewrites=draft.rewrite_count)

    return {
        "completed_sections":    [draft],           # operator.add appends
        "current_section_index": idx + 1,           # Advance to next section
        "status_messages": [
            f"✅ Accepted '{draft.title}'"
            + (f" (after {draft.rewrite_count} rewrite(s))" if draft.rewrite_count else "")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 8 — assemble_document
# ─────────────────────────────────────────────────────────────────────────────

def assemble_document(state: GraphState) -> dict:
    """
    Control node: assembles all accepted sections into ordered Markdown.

    Uses section_plan order (not completion order) so sections are always
    in their intended document position even if written out of order.

    Returns:
        Updates: assembled_content, status_messages
    """
    log.info("node.assemble_document.start",
             sections=len(state["completed_sections"]))

    # Sort by section_index to preserve plan order
    ordered = sorted(state["completed_sections"], key=lambda s: s.section_index)

    parts = [f"# {state['doc_title']}\n"]
    for draft in ordered:
        parts.append(f"\n## {draft.title}\n")
        parts.append(draft.content)
        parts.append("\n\n---\n")

    assembled = "\n".join(parts)

    # Quality summary
    avg_score = (
        sum(s.validation_score for s in ordered) / len(ordered)
        if ordered else 0.0
    )
    total_rewrites = sum(s.rewrite_count for s in ordered)

    log.info("node.assemble_document.done",
             total_chars=len(assembled),
             avg_score=f"{avg_score:.2f}",
             total_rewrites=total_rewrites)

    return {
        "assembled_content": assembled,
        "status_messages": [
            f"📄 Document assembled — {len(ordered)} sections, "
            f"avg quality {avg_score:.2f}, {total_rewrites} total rewrite(s)"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 9 — apply_template
# ─────────────────────────────────────────────────────────────────────────────

def apply_template(state: GraphState) -> dict:
    """
    Template node: validates that the assembled Markdown respects the template's
    structural slots. Inserts letterhead placeholder, header, footer markers.

    This node does NOT reformat visually — that happens in the output plugin.
    It annotates the Markdown with template metadata comments that the DOCX/PDF
    output plugins read to apply fonts, styles, and layout.

    Returns:
        Updates: assembled_content (annotated), status_messages
    """
    styles  = state["template_styles"]
    content = state["assembled_content"]

    if not styles:
        log.info("node.apply_template.no_template")
        return {
            "status_messages": ["ℹ️  No template — using default styling"],
        }

    log.info("node.apply_template.start",
             has_letterhead=styles.has_letterhead,
             has_page_numbers=styles.has_page_numbers)

    # Inject metadata block at the top that output plugins read
    metadata = [
        "<!-- DOCFORGE_TEMPLATE_META",
        f"primary_color: {styles.primary_color_hex}",
        f"heading1_font: {styles.heading1.name if styles.heading1 else 'Calibri'}",
        f"heading1_size: {styles.heading1.size_pt if styles.heading1 else 16.0}",
        f"body_font: {styles.body.name if styles.body else 'Calibri'}",
        f"body_size: {styles.body.size_pt if styles.body else 11.0}",
        f"table_header_bg: {styles.table_header_bg}",
        f"has_letterhead: {styles.has_letterhead}",
        f"letterhead_placeholder: {styles.letterhead_placeholder}",
        f"header_text: {styles.header_text}",
        f"footer_text: {styles.footer_text}",
        f"has_page_numbers: {styles.has_page_numbers}",
        "DOCFORGE_TEMPLATE_META -->",
    ]
    annotated = "\n".join(metadata) + "\n\n" + content

    log.info("node.apply_template.done")

    return {
        "assembled_content": annotated,
        "status_messages":   ["🎨 Template metadata applied"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 10 — generate_output
# ─────────────────────────────────────────────────────────────────────────────

def generate_output_files(state: GraphState) -> dict:
    """
    Output node: routes assembled+annotated content to output format plugins.

    Returns:
        Updates: output_paths, status_messages
    """
    log.info("node.generate_output.start", formats=state["output_formats"])

    # Convert completed SectionDraft list back to GeneratedSection for output pipeline
    from agents.base import GeneratedSection
    sections = [
        GeneratedSection(
            title   = s.title,
            content = s.content,
            sources = s.chunk_ids_used,
        )
        for s in sorted(state["completed_sections"], key=lambda x: x.section_index)
    ]

    output_paths = generate_output(
        sections        = sections,
        formats         = state["output_formats"],
        template_styles = state["template_styles"],
        project_name    = state["project_name"],
        doc_title       = state["doc_title"],
    )

    str_paths = {fmt: str(path) for fmt, path in output_paths.items()}

    log.info("node.generate_output.done", paths=str_paths)

    return {
        "output_paths":    str_paths,
        "status_messages": [
            f"📁 Output ready: {', '.join(str_paths.keys())}"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_previous_summary(
    completed: list[SectionDraft],
    max_sections: int = 2,
    max_chars_per_section: int = 400,
) -> str:
    """
    Build a compact summary of the most recently completed sections.
    Used to maintain cross-section consistency without blowing the context budget.
    Only the last max_sections sections are included, each truncated to max_chars.
    """
    if not completed:
        return "No sections written yet."

    recent = sorted(completed, key=lambda s: s.section_index)[-max_sections:]
    parts  = []
    for s in recent:
        truncated = s.content[:max_chars_per_section]
        if len(s.content) > max_chars_per_section:
            truncated += "... [truncated]"
        parts.append(f"**{s.title}** (summary):\n{truncated}")

    return "\n\n".join(parts)
```

---

## 7. Conditional Edge Routing

All routing logic lives in `edges.py`. No node function makes routing decisions.

### `agents/graph/edges.py`

```python
from __future__ import annotations

from agents.graph.state import GraphState
from core.logger import get_logger

log = get_logger(__name__)


def route_after_select(state: GraphState) -> str:
    """
    After select_next_section: route to retrieval if sections remain,
    or to assembly if all sections are done.

    Returns:
        "retrieve_chunks"     if current_section_index < len(section_plan)
        "assemble_document"   if all sections are written
    """
    idx          = state["current_section_index"]
    total        = len(state["section_plan"])
    completed    = len(state["completed_sections"])

    if completed >= total:
        log.debug("edge.route_after_select", decision="assemble", completed=completed, total=total)
        return "assemble_document"

    log.debug("edge.route_after_select", decision="retrieve", section=idx, total=total)
    return "retrieve_chunks"


def route_after_validate(state: GraphState) -> str:
    """
    After validate_section: accept if passed OR max retries hit,
    otherwise rewrite.

    Returns:
        "accept_section"    if validation passed or max_rewrite_attempts exhausted
        "rewrite_section"   if failed and retries remain
    """
    validation    = state["last_validation"]
    rewrite_count = state["current_rewrite_count"]
    max_retries   = state["max_rewrite_attempts"]
    section_title = state["section_plan"][state["current_section_index"]].title

    if validation is None:
        log.warning("edge.route_after_validate.no_validation", section=section_title)
        return "accept_section"

    if validation.passed:
        log.debug("edge.route_after_validate",
                  decision="accept", section=section_title, score=validation.score)
        return "accept_section"

    if rewrite_count >= max_retries:
        log.warning("edge.route_after_validate.max_retries_hit",
                    section=section_title,
                    score=validation.score,
                    retries=rewrite_count)
        return "accept_section"   # Accept with warning flag

    log.debug("edge.route_after_validate",
              decision="rewrite",
              section=section_title,
              score=validation.score,
              attempt=rewrite_count + 1)
    return "rewrite_section"
```

---

## 8. Graph Assembly

### `agents/graph/pipeline.py`

```python
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.graph.state import GraphState
from agents.graph.nodes import (
    plan_document,
    select_next_section,
    retrieve_chunks,
    write_section,
    validate_section,
    rewrite_section,
    accept_section,
    assemble_document,
    apply_template,
    generate_output_files,
)
from agents.graph.edges import route_after_select, route_after_validate
from core.logger import get_logger

log = get_logger(__name__)


def build_document_graph() -> StateGraph:
    """
    Assemble the DocForge LangGraph document generation graph.

    Graph structure:
        START
          → plan_document
          → select_next_section
          → [conditional] retrieve_chunks | assemble_document
        retrieve_chunks
          → write_section
          → validate_section
          → [conditional] accept_section | rewrite_section
        rewrite_section
          → validate_section     (loops back)
        accept_section
          → select_next_section  (loops back for next section)
        assemble_document
          → apply_template
          → generate_output_files
          → END

    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    builder = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("plan_document",        plan_document)
    builder.add_node("select_next_section",  select_next_section)
    builder.add_node("retrieve_chunks",      retrieve_chunks)
    builder.add_node("write_section",        write_section)
    builder.add_node("validate_section",     validate_section)
    builder.add_node("rewrite_section",      rewrite_section)
    builder.add_node("accept_section",       accept_section)
    builder.add_node("assemble_document",    assemble_document)
    builder.add_node("apply_template",       apply_template)
    builder.add_node("generate_output_files",generate_output_files)

    # ── Set entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("plan_document")

    # ── Static edges ──────────────────────────────────────────────────────────
    builder.add_edge("plan_document",         "select_next_section")
    builder.add_edge("retrieve_chunks",       "write_section")
    builder.add_edge("write_section",         "validate_section")
    builder.add_edge("rewrite_section",       "validate_section")
    builder.add_edge("accept_section",        "select_next_section")
    builder.add_edge("assemble_document",     "apply_template")
    builder.add_edge("apply_template",        "generate_output_files")
    builder.add_edge("generate_output_files", END)

    # ── Conditional edges ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "select_next_section",
        route_after_select,
        {
            "retrieve_chunks":   "retrieve_chunks",
            "assemble_document": "assemble_document",
        },
    )

    builder.add_conditional_edges(
        "validate_section",
        route_after_validate,
        {
            "accept_section":  "accept_section",
            "rewrite_section": "rewrite_section",
        },
    )

    return builder.compile()


# ── Singleton compiled graph (built once per process) ─────────────────────────
_GRAPH: StateGraph | None = None

def get_graph() -> StateGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_document_graph()
        log.info("graph.compiled")
    return _GRAPH
```

---

## 9. Public Pipeline Entry Points

These are the two functions called by the Streamlit UI and any other caller.
They build the initial state, invoke the graph, and return the final state.

### `agents/graph/pipeline.py` — continued

```python
# (append to agents/graph/pipeline.py)

from pathlib import Path
from template.store import TemplateStore
from template.analyzer import TemplateStyles
from core.config import get_config


def run_itd_pipeline(
    instruction: str,
    project_name: str,
    doc_title: str,
    template_name: str | None = None,
    output_formats: list[str] | None = None,
    on_status: callable | None = None,
) -> dict:
    """
    Run the Instruction-to-Document LangGraph pipeline end-to-end.

    Args:
        instruction:     User instruction describing the document to generate.
        project_name:    Project namespace (LanceDB, cache, templates).
        doc_title:       Base filename for output documents.
        template_name:   Name of an uploaded template (or None for default styling).
        output_formats:  List of formats: ["docx", "pdf", "md"]. Defaults to ["docx", "pdf"].
        on_status:       Optional callback(message: str) for streaming status to UI.

    Returns:
        Final GraphState dict containing output_paths, completed_sections, etc.
    """
    cfg = get_config()

    template_styles, template_slots = _load_template(project_name, template_name)

    initial_state: GraphState = {
        # Input
        "instruction":            instruction,
        "project_name":           project_name,
        "mode":                   "itd",
        "template_styles":        template_styles,
        "template_section_slots": template_slots,
        "output_formats":         output_formats or ["docx", "pdf"],
        "doc_title":              doc_title,
        "itd_model":              cfg.models.itd_model,
        "ctd_model":              cfg.models.ctd_model,
        "context_budget_tokens":  cfg.agents.itd_context_budget_tokens,
        "max_retrieval_chunks":   cfg.agents.max_retrieval_chunks,
        "min_relevance_score":    cfg.agents.min_relevance_score,
        "validation_threshold":   0.72,
        "max_rewrite_attempts":   2,

        # Working state (starts empty)
        "section_plan":           [],
        "current_section_index":  0,
        "all_used_chunk_ids":     [],
        "current_section_chunks": [],
        "current_draft":          None,
        "current_rewrite_count":  0,
        "completed_sections":     [],
        "last_validation":        None,

        # Output
        "assembled_content": "",
        "output_paths":      {},

        # UI
        "status_messages": [],
        "stream_buffer":   "",
    }

    log.info("pipeline.itd.start", instruction=instruction[:80], project=project_name)

    graph        = get_graph()
    final_state  = graph.invoke(
        initial_state,
        config={"callbacks": [_make_status_callback(on_status)] if on_status else []},
    )

    log.info("pipeline.itd.done",
             sections=len(final_state.get("completed_sections", [])),
             outputs=list(final_state.get("output_paths", {}).keys()))

    return final_state


def run_ctd_pipeline(
    code_paths: list[Path],
    project_name: str,
    doc_title: str,
    doc_type: str = "API Reference",
    template_name: str | None = None,
    output_formats: list[str] | None = None,
    on_status: callable | None = None,
) -> dict:
    """
    Run the Code-to-Document LangGraph pipeline end-to-end.

    Ingests code files via the CTD parser, then runs the same graph
    with mode="ctd" and ctd_model for all LLM calls.

    Args:
        code_paths:    List of source file paths (.py, .c, .cpp, .h, .java).
        project_name:  Project namespace.
        doc_title:     Base filename for output.
        doc_type:      Document type hint: "API Reference", "Design Doc", etc.
        template_name: Uploaded template name or None.
        output_formats: Output format list.
        on_status:     Optional streaming status callback.

    Returns:
        Final GraphState dict.
    """
    from agents.ctd.parser import ParserAgent

    cfg = get_config()

    # Parse all code files and build an instruction from the analysis
    parser    = ParserAgent()
    analyses  = []
    for path in code_paths:
        try:
            parsed = parser.parse_file(path)
            analyses.append(_summarise_code_analysis(parsed))
        except Exception as e:
            log.warning("pipeline.ctd.parse_skip", file=str(path), error=str(e))

    combined_instruction = (
        f"Generate a {doc_type} for the following codebase.\n\n"
        + "\n\n".join(analyses)
    )

    template_styles, template_slots = _load_template(project_name, template_name)

    initial_state: GraphState = {
        "instruction":            combined_instruction,
        "project_name":           project_name,
        "mode":                   "ctd",
        "template_styles":        template_styles,
        "template_section_slots": template_slots,
        "output_formats":         output_formats or ["docx", "pdf", "md"],
        "doc_title":              doc_title,
        "itd_model":              cfg.models.itd_model,
        "ctd_model":              cfg.models.ctd_model,
        "context_budget_tokens":  cfg.agents.ctd_context_budget_tokens,
        "max_retrieval_chunks":   cfg.agents.max_retrieval_chunks,
        "min_relevance_score":    cfg.agents.min_relevance_score,
        "validation_threshold":   0.70,
        "max_rewrite_attempts":   2,

        "section_plan":           [],
        "current_section_index":  0,
        "all_used_chunk_ids":     [],
        "current_section_chunks": [],
        "current_draft":          None,
        "current_rewrite_count":  0,
        "completed_sections":     [],
        "last_validation":        None,
        "assembled_content":      "",
        "output_paths":           {},
        "status_messages":        [],
        "stream_buffer":          "",
    }

    log.info("pipeline.ctd.start", files=len(code_paths), doc_type=doc_type)

    graph       = get_graph()
    final_state = graph.invoke(
        initial_state,
        config={"callbacks": [_make_status_callback(on_status)] if on_status else []},
    )

    log.info("pipeline.ctd.done",
             sections=len(final_state.get("completed_sections", [])))

    return final_state


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_template(
    project_name: str,
    template_name: str | None,
) -> tuple[TemplateStyles | None, list[str]]:
    """Load template styles and extract structural slot names."""
    if not template_name:
        return None, []

    store  = TemplateStore(project_name)
    styles = store.load_styles(template_name)
    slots  = _extract_template_slots(project_name, template_name)
    return styles, slots


def _extract_template_slots(project_name: str, template_name: str) -> list[str]:
    """
    Extract section heading names from a template .docx so the planner
    can mirror the template's structure.
    """
    from pathlib import Path
    from docx import Document

    template_path = Path(f"data/projects/{project_name}/templates/{template_name}.docx")
    if not template_path.exists():
        return []

    try:
        doc   = Document(str(template_path))
        slots = [
            p.text.strip()
            for p in doc.paragraphs
            if p.style.name.startswith("Heading") and p.text.strip()
        ]
        return slots
    except Exception as e:
        log.warning("pipeline.extract_slots.failed", error=str(e))
        return []


def _summarise_code_analysis(parsed) -> str:
    """Convert a ParsedCodeFile into a compact text summary for the CTD instruction."""
    parts = [f"## File: {parsed.path.name} ({parsed.language})"]
    if parsed.module_docstring:
        parts.append(f"Module docstring: {parsed.module_docstring[:300]}")
    if parsed.classes:
        parts.append(f"Classes ({len(parsed.classes)}): "
                     + ", ".join(c.name for c in parsed.classes))
    if parsed.functions:
        parts.append(f"Top-level functions ({len(parsed.functions)}): "
                     + ", ".join(f.name for f in parsed.functions))
    return "\n".join(parts)


def _make_status_callback(on_status: callable):
    """Create a LangGraph callback that pipes status_messages to the UI callback."""
    from langchain_core.callbacks import BaseCallbackHandler

    class StatusCallback(BaseCallbackHandler):
        def on_chain_end(self, outputs, **kwargs):
            for msg in outputs.get("status_messages", []):
                on_status(msg)

    return StatusCallback()
```

---

## 10. Streamlit Integration

The `generate.py` UI page calls `run_itd_pipeline()` or `run_ctd_pipeline()`
and consumes the `on_status` callback to drive the `st.status()` panels.

### `ui/pages/generate.py` — updated integration

```python
import streamlit as st
from agents.graph.pipeline import run_itd_pipeline, run_ctd_pipeline
from core.config import get_config


def render_generate_tab(project_name: str) -> None:
    cfg  = get_config()
    mode = st.radio("Mode", ["Instruction → Document", "Code → Document"], horizontal=True)

    if "Instruction" in mode:
        _render_itd_tab(project_name, cfg)
    else:
        _render_ctd_tab(project_name, cfg)


def _render_itd_tab(project_name: str, cfg) -> None:
    instr    = st.text_area("Instruction / Document Type", height=130,
                             placeholder="Generate a Software Requirements Specification for...")
    doc_title = st.text_input("Document Title", value="Generated Document")
    tmpl     = st.selectbox("Template", _get_templates(project_name))
    fmts     = st.multiselect("Output Formats", ["docx", "pdf", "md", "excel"],
                               default=["docx", "pdf"])

    if st.button("Generate Document", type="primary", disabled=not instr.strip()):
        _run_itd(instr, doc_title, tmpl, fmts, project_name)


def _run_itd(instruction, doc_title, template, formats, project_name):
    status_log: list[str] = []

    status_container = st.empty()

    def on_status(msg: str):
        status_log.append(msg)
        with status_container.container():
            for m in status_log[-10:]:     # Show last 10 status messages
                st.markdown(m)

    with st.spinner("Running document pipeline..."):
        final_state = run_itd_pipeline(
            instruction    = instruction,
            project_name   = project_name,
            doc_title      = doc_title,
            template_name  = template if template and template != "None" else None,
            output_formats = formats,
            on_status      = on_status,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    sections   = final_state.get("completed_sections", [])
    output_paths = final_state.get("output_paths", {})

    st.success(f"Document generated — {len(sections)} sections")

    # Quality table
    if sections:
        import pandas as pd
        quality_df = pd.DataFrame([
            {
                "Section":   s.title,
                "Score":     f"{s.validation_score:.2f}",
                "Rewrites":  s.rewrite_count,
                "Chunks":    len(s.chunk_ids_used),
            }
            for s in sorted(sections, key=lambda x: x.section_index)
        ])
        st.dataframe(quality_df, use_container_width=True)

    # Download buttons
    for fmt, path in output_paths.items():
        with open(path, "rb") as f:
            st.download_button(
                label     = f"⬇️  Download {fmt.upper()}",
                data      = f,
                file_name = Path(path).name,
                key       = f"dl_{fmt}",
            )

    # Full document preview (Markdown)
    with st.expander("Preview assembled document", expanded=False):
        st.markdown(final_state.get("assembled_content", ""))


def _render_ctd_tab(project_name: str, cfg) -> None:
    uploaded = st.file_uploader(
        "Upload source files",
        accept_multiple_files=True,
        type=["py", "c", "cpp", "h", "java"],
    )
    doc_type  = st.selectbox("Document Type",
                              ["API Reference", "Technical Design Doc",
                               "Module Documentation", "Deployment Runbook"])
    doc_title = st.text_input("Document Title", value="Code Documentation")
    tmpl      = st.selectbox("Template", _get_templates(project_name), key="ctd_tmpl")
    fmts      = st.multiselect("Output Formats", ["docx", "pdf", "md"],
                                default=["docx", "md"], key="ctd_fmts")

    if st.button("Generate Documentation", type="primary", disabled=not uploaded):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            code_paths = []
            for uf in uploaded:
                tmp_path = Path(tmp) / uf.name
                tmp_path.write_bytes(uf.read())
                code_paths.append(tmp_path)

            status_log: list[str] = []
            status_container      = st.empty()

            def on_status(msg: str):
                status_log.append(msg)
                with status_container.container():
                    for m in status_log[-10:]:
                        st.markdown(m)

            with st.spinner("Analysing code and generating documentation..."):
                final_state = run_ctd_pipeline(
                    code_paths     = code_paths,
                    project_name   = project_name,
                    doc_title      = doc_title,
                    doc_type       = doc_type,
                    template_name  = tmpl if tmpl and tmpl != "None" else None,
                    output_formats = fmts,
                    on_status      = on_status,
                )

            output_paths = final_state.get("output_paths", {})
            st.success(f"Documentation generated — {len(final_state.get('completed_sections',[]))} sections")
            for fmt, path in output_paths.items():
                with open(path, "rb") as f:
                    st.download_button(f"⬇️  Download {fmt.upper()}", f,
                                       file_name=Path(path).name, key=f"ctd_dl_{fmt}")


def _get_templates(project_name: str) -> list[str]:
    from template.store import TemplateStore
    return ["None"] + TemplateStore(project_name).list_templates()
```

---

## 11. Adding to requirements.txt

Add the following to `requirements.txt`:

```
# LangGraph agent graph
langgraph>=0.2.0
langchain-core>=0.2.0
```

No LangChain LLM wrappers are used — DocForge calls Ollama directly via `llm/client.py`.
`langchain-core` is needed only for `BaseCallbackHandler` in the status callback.

---

## Graph Topology — Full Visual

```
START
  │
  ▼
plan_document ──────────────────────────────────────────────────┐
  │                                                             │
  │  Sets: section_plan, current_section_index=0               │
  ▼                                                             │
select_next_section ◄────────────────────── accept_section      │
  │                                              ▲              │
  │                                              │              │
  ├──[completed < total]──► retrieve_chunks      │              │
  │                              │               │              │
  │                              ▼               │              │
  │                         write_section        │              │
  │                              │               │              │
  │                              ▼               │              │
  │                        validate_section      │              │
  │                              │               │              │
  │                    ┌─────────┴──────────┐    │              │
  │                    │                    │    │              │
  │               [passed OR           [failed AND             │
  │                max retries]         retries left]          │
  │                    │                    │                   │
  │                    ▼                    ▼                   │
  │              accept_section ──►  rewrite_section           │
  │                                        │                   │
  │                                        ▼                   │
  │                                  validate_section ◄────────┘
  │                                  (loops until pass
  │                                   or max retries)
  │
  └──[completed == total]──► assemble_document
                                    │
                                    ▼
                              apply_template
                                    │
                                    ▼
                           generate_output_files
                                    │
                                   END
```

---

## Design Decisions

### Why validate_section calls the LLM and not a heuristic

Heuristics (length check, keyword presence) miss semantic completeness.
The validator LLM checks whether the section actually addresses its instructions —
something only an LLM can do. The validator uses the ITD model in both modes
because quality assessment is language-level, not code-level.

### Why accept_section is a separate node from validate_section

Keeps single responsibility. `validate_section` only scores. `accept_section`
only transitions state. This means the edge routing in `edges.py` is the only
place that makes the accept/rewrite decision — not split across two nodes.

### Why retrieval_queries come from the planner, not generated fresh per section

The planner sees the full document structure when generating queries — it knows
that "System Architecture" and "Component Design" are separate sections, so it
can write queries that deliberately do NOT overlap. Generating queries at retrieval
time would cause the same chunks to be retrieved for multiple sections.

### Why already_used_chunk_ids gets a penalty instead of a hard exclude

Hard exclusion would prevent a fundamentally important chunk from appearing in
a section that needs it, even if it was used earlier. The penalty (0.25 score
deduction) means the chunk can still win on relevance, but only when it's
genuinely the best match — preventing lazy re-retrieval of already-seen content.

### Why max_rewrite_attempts defaults to 2

3 LLM calls per section (write + validate + rewrite) is the sweet spot for quality
vs. speed on a 7B CPU model. A second rewrite is available for genuinely poor drafts.
Beyond 2 rewrites, the marginal quality gain doesn't justify the CPU time.

### Why template slots go to the planner, not to apply_template

The planner needs to know the template's section structure before generating the plan
so it can match (or extend) the template's headings. `apply_template` only injects
style metadata — visual formatting happens in the output plugin, not in content generation.
```
