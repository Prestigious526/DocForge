# DocForge — Developer Guide
> How to write code for this system. Not architecture — the coding standard.
> Every pattern here maps directly to the architecture document.

---

## Core Principles

Every piece of DocForge code must be **traceable**, **predictable**, and **replaceable**.

- **Traceable** — every agent step, plugin call, and LLM request leaves a structured log trail.
- **Predictable** — function names state the action and the thing acted on. No surprises.
- **Replaceable** — every model, parser, and store is behind an interface. Swap without touching pipelines.

A fourth principle specific to DocForge: **budget-aware**. No function passes unbounded context to an LLM. Every agent has a token budget. Every retrieval has a chunk cap.

---

## Table of Contents

1. [Naming Conventions](#1-naming-conventions)
2. [Function Design](#2-function-design)
3. [Module Design](#3-module-design)
4. [Config Access](#4-config-access)
5. [Logging](#5-logging)
6. [Error Reporting](#6-error-reporting)
7. [Async Patterns](#7-async-patterns)
8. [Streaming Patterns](#8-streaming-patterns)
9. [Writing a Plugin](#9-writing-a-plugin)
10. [Writing an Agent](#10-writing-an-agent)
11. [Docker Development Patterns](#11-docker-development-patterns)
12. [Testing Conventions](#12-testing-conventions)
13. [Quick Reference](#13-quick-reference)

---

## 1. Naming Conventions

### Files & Modules

`snake_case`. Name files after what they **do**, not what they contain.
The DocForge module tree is the canonical reference — every file already follows this.

```
ingest/
  pipeline.py        ✅  orchestrates the ingestion flow
  file_cache.py      ✅  stores and retrieves cached files
  helpers.py         ❌  vague — helpers for what?
  utils.py           ❌  not a module, it's a junk drawer

ocr/
  preprocessor.py    ✅  preprocesses images before OCR
  extractor.py       ✅  extracts text via Tesseract
  ocr_stuff.py       ❌  describes nothing

agents/itd/
  planner.py         ✅  plans the document structure
  writer.py          ✅  writes document sections
  agent.py           ❌  which agent?

plugins/
  loader.py          ✅  loads and installs plugins
  registry.py        ✅  maps extensions and formats to plugins
  manager.py         ❌  manages what exactly?
```

### Functions

`verb_noun` always. The verb is an action. The noun is the thing being acted on.
Every function name in DocForge follows this — treat any deviation as a bug.

```python
# ✅ DocForge patterns
def ingest_file(source_path: Path, project_name: str) -> IngestedFile: ...
def analyze_template(template_path: Path) -> TemplateStyles: ...
def embed_texts(texts: list[str]) -> list[list[float]]: ...
def preprocess_image(image: Image) -> np.ndarray: ...
def extract_text_with_tesseract(image: np.ndarray) -> str: ...
def correct_ocr_output(raw_text: str) -> str: ...
def install_plugin_from_zip(zip_path: Path) -> str: ...
def generate_output(sections: list[GeneratedSection], ...) -> dict[str, Path]: ...
def load_all_installed_plugins() -> list: ...
def plan_document(ctx: AgentContext) -> list[dict]: ...

# ❌ Avoid
def process_doc(): ...
def do_ocr(): ...
def handle(): ...
def run(): ...
def get_stuff(): ...
```

### Variables

Descriptive over short. Align assignments for readability in multi-step pipelines.
Single-letter names only inside list comprehensions and tiny loops.

```python
# ✅
parsed_doc       = plugin.parse(source_path, config={})
preprocessed_img = preprocess_image(raw_image)
extracted_text   = extract_text_with_tesseract(preprocessed_img)
corrected_text   = correct_ocr_output(extracted_text)
chunks           = plugin.chunk(parsed_doc, chunk_size=512, overlap=64)

# ❌
d = plugin.parse(p, {})
img = preprocess(raw)
t = ocr(img)
```

### Classes

`PascalCase`. Name after the **role**, not the library or implementation.

```python
# ✅ Role-based
class PlannerAgent:       # what it does in the system
class WriterAgent:        # same
class EmbeddingStore:     # stores embeddings — the role
class FileCacheManager:   # manages the file cache — the role
class TemplateStore:      # stores templates — the role

# ✅ Impl in name only when the impl IS the public contract (plugins)
class PdfIngestor:        # the PDF ingestor — extension IS identity
class DocxOutputWriter:   # the DOCX writer — format IS identity

# ❌
class EmbeddingUtils:     # not a class — a module
class OcrManager:         # manager of what? call it OcrPipeline or just use functions
class Helper:             # never acceptable
```

### Private Helpers

Prefix with `_`. Private helpers live at the bottom of the file, never in `__init__.py`.

```python
def correct_ocr_output(raw_text: str) -> str:     # public — called by pipeline
    text = _remove_noise_lines(raw_text)           # private helpers
    text = _fix_common_substitutions(text)
    return _normalize_whitespace(text)

def _remove_noise_lines(text: str) -> str: ...    # private — only used in this file
def _fix_common_substitutions(text: str) -> str: ...
def _normalize_whitespace(text: str) -> str: ...
```

### Constants

`SCREAMING_SNAKE_CASE`. Live in `core/constants.py` or at the top of the file
if they are truly file-local. Never scatter magic numbers across functions.

```python
# core/constants.py — system-wide
MIN_TEXT_CHARS_PER_PAGE   = 80     # below this → needs OCR
DEFAULT_CHUNK_SIZE        = 512
DEFAULT_CHUNK_OVERLAP     = 64
MAX_RETRIEVAL_CHUNKS      = 8
MIN_RELEVANCE_SCORE       = 0.68

# In a plugin file — file-local constant, prefixed with _
_MAX_TABLE_ROWS = 500
```

---

## 2. Function Design

### The Standard Shape

Every non-trivial function in DocForge follows this exact structure.
No exceptions. The log calls at `start`, `done`, and `failed` are not optional.

```python
def ingest_file(
    source_path: Path,
    project_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int    = DEFAULT_CHUNK_OVERLAP,
) -> IngestedFile:
    """
    Full ingestion pipeline for a single file.

    Detects file type → routes to plugin → OCR if needed →
    chunks → embeds → stores in LanceDB + encrypted file cache.

    Args:
        source_path:  Absolute path to the file to ingest.
        project_name: Project namespace for storage isolation.
        chunk_size:   Target token count per chunk.
        overlap:      Token overlap between adjacent chunks.

    Returns:
        IngestedFile metadata record for display in the Files tab.

    Raises:
        IngestError:  If no plugin handles the file type.
        OcrError:     If OCR is required but Tesseract is unavailable.
        StoreError:   If the vector store write fails.
    """
    log.info("ingest_file.start", file=source_path.name, project=project_name)

    try:
        plugin = PluginRegistry.get_ingestor_for(source_path.suffix)
        if plugin is None:
            raise IngestError(f"No ingestor plugin for extension: {source_path.suffix}")

        parsed   = plugin.parse(source_path, config={})
        chunks   = plugin.chunk(parsed, chunk_size, overlap)
        store    = EmbeddingStore(project_name)
        store.upsert_chunks(file_id=file_id, filename=source_path.name, chunks=chunks)

        log.info("ingest_file.done", file=source_path.name, chunks=len(chunks))
        return IngestedFile(...)

    except (PluginParseError, StoreError) as e:
        log.error("ingest_file.failed", file=source_path.name, error=str(e))
        raise IngestError(f"Ingestion failed for {source_path.name}: {e}") from e
```

**Non-negotiable rules:**
- Type every argument and every return value. No untyped functions.
- Docstring always: `Args`, `Returns`, `Raises`. Match them to reality.
- `log.info("function_name.start", ...)` — first line after validation.
- `log.info("function_name.done", ...)` — last line before return.
- `log.error("function_name.failed", error=str(e))` — inside every except block.
- Catch specific exceptions. Never bare `except:` or `except Exception:` at module level.
- Raise domain errors (`IngestError`, `LLMError`, etc.), never re-raise library errors raw.
- One function does one thing. If you write "and" in the docstring verb — split it.

### Dataclasses Over Dicts for Return Values

Never return a raw `dict` from a public function. Use a `@dataclass`.
This makes call sites type-safe and self-documenting.

```python
# ✅
@dataclass
class IngestedFile:
    file_id: str
    filename: str
    chunk_count: int
    page_count: int
    was_ocrd: bool

result = ingest_file(path, "my_project")
print(result.chunk_count)   # clear, type-checked

# ❌
result = ingest_file(path, "my_project")
print(result["chunk_count"])  # fragile, not type-checked
```

### Guard Clauses First

Validate inputs at the top. Don't nest happy-path logic inside conditionals.

```python
# ✅
def save_template(self, source_path: Path) -> TemplateStyles:
    if not source_path.exists():
        raise TemplateAnalysisError(f"Template file not found: {source_path}")
    if source_path.suffix.lower() != ".docx":
        raise TemplateAnalysisError(f"Templates must be .docx, got: {source_path.suffix}")

    # Happy path — no nesting
    dest   = self.template_dir / source_path.name
    styles = analyze_template(dest)
    self._persist_styles(source_path.stem, styles)
    return styles

# ❌
def save_template(self, source_path: Path) -> TemplateStyles:
    if source_path.exists():
        if source_path.suffix.lower() == ".docx":
            dest   = self.template_dir / source_path.name
            styles = analyze_template(dest)
            ...
```

---

## 3. Module Design

### One Responsibility Per Module

Every file in DocForge has exactly one job. The module name states it.
If you find yourself adding an "also" to a module's description — extract a new module.

```
core/
  config.py        → loads and merges TOML + env var configuration
  logger.py        → configures structlog + Rich + file handler + module switches
  errors.py        → the entire DocForge exception hierarchy
  constants.py     → system-wide constants only

ingest/
  pipeline.py      → orchestrates full ingestion for one file
  file_cache.py    → encrypts and retrieves cached source files
  chunker.py       → text chunking utilities (no I/O)

ocr/
  detector.py      → decides if a parsed doc needs OCR
  preprocessor.py  → OpenCV image corrections (deskew, denoise, binarize)
  extractor.py     → Tesseract OCR caller
  corrector.py     → post-OCR text cleanup (no I/O, pure string transforms)

agents/itd/
  planner.py       → LLM call: instruction → section plan (JSON)
  retriever.py     → LanceDB search scoped to one section's context budget
  writer.py        → LLM streaming: section plan + chunks → content generator
  reviewer.py      → LLM call: review and refine a completed section

agents/ctd/
  parser.py        → tree-sitter AST extraction for Python, C/C++, Java
  analyzer.py      → LLM call: AST summary → code intent and relationships
  writer.py        → LLM streaming: code analysis → doc section generator
  reviewer.py      → LLM call: validates doc section against original AST

embeddings/
  client.py        → async Ollama /api/embeddings calls
  store.py         → LanceDB upsert, search, delete operations
```

### Standard File Layout

Every file in DocForge follows this exact top-down order. No exceptions.

```python
# 1. ── Future annotations (always first) ──────────────────────────────────────
from __future__ import annotations

# 2. ── Standard library ───────────────────────────────────────────────────────
import json
from dataclasses import dataclass
from pathlib import Path

# 3. ── Third-party ────────────────────────────────────────────────────────────
import httpx
import structlog

# 4. ── Internal (DocForge) — deepest dependencies first ──────────────────────
from core.config import get_config
from core.errors import IngestError, StoreError
from core.logger import get_logger

# 5. ── Module-level logger — always named after the module ───────────────────
log = get_logger(__name__)

# 6. ── Module-level constants (if file-local) ─────────────────────────────────
_RETRY_LIMIT = 3

# 7. ── Dataclasses / TypedDicts ───────────────────────────────────────────────
@dataclass
class IngestedFile:
    file_id: str
    ...

# 8. ── Public functions ────────────────────────────────────────────────────────
def ingest_file(...) -> IngestedFile: ...

# 9. ── Classes ────────────────────────────────────────────────────────────────
class EmbeddingStore:
    ...

# 10. ── Private helpers ────────────────────────────────────────────────────────
def _generate_file_id(path: Path) -> str: ...
```

### What goes in `__init__.py`

`__init__.py` files are almost always empty. Never put logic in them.
Use them only for re-exports when a sub-package has a clear public API.

```python
# plugins/__init__.py — the only acceptable use of __init__.py
from plugins.registry import PluginRegistry
from plugins.base import IngestorPlugin, OutputPlugin

__all__ = ["PluginRegistry", "IngestorPlugin", "OutputPlugin"]
```

---

## 4. Config Access

### Never Import Config Values Directly at Module Level

Config is loaded at runtime. Importing config values at import time causes errors
when the module is imported before `load_config()` has been called (e.g., in tests).

```python
# ✅ — read config inside the function that needs it
def embed_texts(texts: list[str]) -> list[list[float]]:
    cfg   = get_config()
    model = cfg.models.embedding_model
    url   = f"{cfg.ollama.base_url}/api/embeddings"
    ...

# ❌ — config read at import time, crashes in tests and CLI --help
from core.config import get_config
cfg   = get_config()                         # called before load_config()
MODEL = cfg.models.embedding_model           # AttributeError
```

### The One Exception: Module-Level Constants Derived from Config

If a module needs a constant derived from config, wrap it in a function
and call it lazily, or use `functools.cached_property` on a class.

```python
# ✅ Lazy access via property
class EmbeddingStore:
    @property
    def _model(self) -> str:
        return get_config().models.embedding_model
```

### Saving User Preferences

When the Settings tab or TUI writes a user preference, always go through `save_user_config`.
Never write to `config/default.toml` at runtime.

```python
from core.config import save_user_config

# User changed the ITD model in Settings tab
save_user_config({"models": {"itd_model": "gemma2:9b"}})

# User toggled a logging module switch
save_user_config({"logging": {"modules": {"ocr": False}}})
```

---

## 5. Logging

DocForge uses `structlog` for structured key=value logging, rendered by Rich in the terminal
with full colour coding. All logs also write to rotating files under `data/logs/`.

Module-level logging can be switched on or off at runtime via the TUI or Settings tab.
The `get_logger(__name__)` call returns a `_NoOpLogger` for disabled modules, so
disabled modules produce zero overhead — no string formatting, no I/O.

### Getting a Logger

```python
# One line, at the top of every module, after imports
log = get_logger(__name__)
```

### Log Syntax — Always `function.event` + Key=Value

The first positional argument is always `"function_name.event"`.
Everything after is a keyword argument. Never use f-strings or string concatenation.

```python
# ✅ Correct — structured, searchable, parseable
log.info("ingest_file.start",    file=source_path.name, project=project_name)
log.info("ingest_file.done",     file=source_path.name, chunks=len(chunks), pages=len(parsed.pages))
log.warning("ocr.low_quality",   file=source_path.name, avg_chars=avg, threshold=MIN_TEXT_CHARS_PER_PAGE)
log.error("ingest_file.failed",  file=source_path.name, error=str(e))
log.debug("store.search.result", returned=len(results), top_score=results[0].get("_distance"))

# ❌ Wrong — not structured, not searchable
log.info(f"Starting ingestion of {source_path.name}")
log.error("Failed: " + str(e))
log.info("Done")
```

### Log Levels

| Level | Colour | Use for |
|-------|--------|---------|
| `DEBUG` | dim | Token counts, intermediate values, raw chunk text. Off in production. |
| `INFO` | green | Normal operations: `.start`, `.done`, `.loaded`, `.ready` |
| `WARNING` | yellow | Recoverable: OCR fallback triggered, low relevance score, retry attempted |
| `ERROR` | red | Operation could not complete: plugin parse failed, LLM timeout, write error |
| `CRITICAL` | bold red | System cannot continue: Ollama unreachable, LanceDB unwritable |

### Log Events by Subsystem

Follow these event names consistently across the codebase:

```
ingest_file.start          ingest_file.done           ingest_file.failed
ocr.required               ocr.preprocess.done        ocr.extract.done
store.upsert.start         store.upsert.done          store.search.start
store.search.done          store.delete.done
embeddings.batch.start     embeddings.batch.done      embeddings.request.failed
template.analyze.start     template.analyze.done      template.saved
plugin.install.start       plugin.install.done        plugin.load.failed
planner.start              planner.done               planner.failed
writer.section.start       writer.section.done
reviewer.start             reviewer.done
output.write.start         output.write.done          output.write.failed
pipeline.start             pipeline.done              pipeline.llm_failure
```

### What Never Goes in a Log

```python
# ❌ PII — never log raw user-facing content
log.info("user.query", query=user_input)           # could contain names/emails

# ❌ Full prompts in production
log.debug("llm.prompt", prompt=full_prompt)        # ok in DEBUG, never in INFO

# ❌ Exception objects directly
log.error("failed", error=e)                       # log str(e), not e itself

# ❌ Secrets or key material
log.info("encryption.key.loaded", key=key_bytes)   # never
```

### Toggling Module Logging at Runtime

```python
from core.logger import set_module_logging

# Disable OCR logging (e.g., user unchecked it in Settings)
set_module_logging("ocr", enabled=False)

# Re-enable it
set_module_logging("ocr", enabled=True)
```

This is the only function that toggles log switches. The TUI checkboxes and Settings tab
both call this — never write directly to `_MODULE_LOGGERS`.

---

## 6. Error Reporting

### The DocForge Exception Hierarchy

All exceptions inherit from `DocForgeError`. Never raise a bare `Exception`.

```
DocForgeError
├── IngestError
│   └── OcrError
├── PluginInstallError
├── PluginLoadError
├── PluginParseError
├── PluginWriteError
├── EmbeddingError
├── StoreError
├── AgentError
│   └── CodeParseError
├── TemplateAnalysisError
├── OutputError
└── LLMError
    └── LLMTimeoutError
```

### The Error Pattern — Wrap at the Boundary

Every function wraps library exceptions into a DocForge domain error.
Library errors (`httpx.TimeoutException`, `lancedb.Error`) must never reach the pipeline.

```python
# ✅ Correct — wrap at the function boundary
async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    log.info("embeddings.batch.start", count=len(texts))
    try:
        async with httpx.AsyncClient(timeout=cfg.ollama.timeout_sec) as client:
            resp = await client.post(url, json={"model": model, "prompt": text})
            resp.raise_for_status()
            return resp.json()["embedding"]

    except httpx.TimeoutException as e:
        log.error("embeddings.request.failed", error=str(e), model=model)
        raise EmbeddingError(f"Embedding request timed out for model {model}") from e

    except httpx.HTTPStatusError as e:
        log.error("embeddings.request.failed", status=e.response.status_code, error=str(e))
        raise EmbeddingError(f"Ollama returned HTTP {e.response.status_code}") from e
```

Always use `raise NewError(...) from e`. This preserves the traceback chain for debugging
while presenting a clean domain error to callers.

### Pipeline-Level Strategy: Degrade, Retry, or Halt

The `ingest/pipeline.py` and `agents/itd/` orchestrators catch domain errors
and decide what to do. The three responses:

```python
def run_itd_pipeline(instruction: str, ctx: AgentContext) -> list[GeneratedSection]:
    log.info("pipeline.start", mode="itd", instruction=instruction[:60])

    try:
        plan = PlannerAgent(model).plan_document(ctx)
    except AgentError as e:
        # Planning is unrecoverable — halt
        log.critical("pipeline.planner_failure", error=str(e))
        raise

    sections = []
    for section in plan:
        try:
            chunks  = retriever.fetch_chunks_for_section(section, ctx)
            content = writer.write_section(ctx, section, chunks)
            sections.append(content)
        except StoreError as e:
            # Retrieval failed — degrade: write section without context
            log.warning("pipeline.retrieval_degraded", section=section["title"], error=str(e))
            content = writer.write_section_no_context(ctx, section)
            sections.append(content)
        except LLMError as e:
            # LLM failed on a section — halt the whole pipeline
            log.critical("pipeline.llm_failure", section=section["title"], error=str(e))
            raise

    log.info("pipeline.done", sections=len(sections))
    return sections
```

Rule of thumb:
- **Halt** if the LLM or planner fails (nothing useful can be produced).
- **Degrade** if retrieval fails (generate from general knowledge instead).
- **Retry** transient errors (Ollama timeout) up to `cfg.ollama.connect_retry_max` times.

---

## 7. Async Patterns

DocForge uses `asyncio` for Ollama API calls (embedding and streaming completion).
The rule is simple: async at the I/O boundary, sync everywhere else.

### The Async Boundary

```
Streamlit UI (sync) → pipeline functions (sync) → LLM/embedding client (async)
```

Streamlit does not natively run async code. Wrap async calls with `asyncio.run()`
at the pipeline level. Never call `asyncio.run()` inside an already-running loop.

```python
# embeddings/client.py — async I/O function
async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    ...

# embeddings/client.py — sync wrapper for pipeline use
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Synchronous wrapper. Safe to call from Streamlit and pipeline code."""
    return asyncio.run(embed_texts_async(texts))
```

### Streaming LLM Responses

Streaming uses a synchronous generator. Each `yield` pushes one token to the caller.
The pattern is used in `WriterAgent.write_section()` and consumed by `st.write_stream()`.

```python
# llm/client.py
def stream_completion(prompt: str, system: str, model: str):
    """
    Synchronous generator yielding text tokens from Ollama streaming response.
    Compatible with st.write_stream() and standard for loops.
    """
    import ollama
    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        stream=True,
    ):
        yield chunk["message"]["content"]
```

### Never Mix asyncio.run() and Streamlit's Event Loop

If you need to call async code from within a Streamlit callback, use this pattern:

```python
import asyncio

def _run_async(coro):
    """Safe async runner compatible with Streamlit's threading model."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
```

---

## 8. Streaming Patterns

Streaming in DocForge flows from the LLM generator through the agent's `write_section()`
method directly into Streamlit's `st.write_stream()`. Preserve the generator chain —
don't collect tokens into a list and then pass the list.

### Agent Writer — Generator Pattern

```python
# agents/itd/writer.py
def write_section(self, ctx: AgentContext, section: dict, chunks: list[dict]):
    """
    Yields tokens for live streaming. After exhausting, access self.result.
    """
    self.log.info("writer.section.start", title=section["title"])

    content_parts = []
    for token in self._stream_llm(prompt=prompt, system=WRITER_SYSTEM):
        content_parts.append(token)
        yield token                  # ← yield immediately, don't buffer

    # Store result for the caller to access after the generator is exhausted
    self.result = GeneratedSection(
        title   = section["title"],
        content = "".join(content_parts),
        sources = [c["chunk_id"] for c in chunks],
    )
    self.log.info("writer.section.done", title=section["title"], chars=len(self.result.content))
```

### Streamlit Consumption

```python
# ui/pages/generate.py
stream_gen = writer.write_section(ctx, section, chunks)

with st.status(f"✍️ Writing: {section['title']}", expanded=True):
    st.write_stream(stream_gen)          # consumes the generator live

section_result = writer.result           # access after stream is exhausted
sections.append(section_result)
```

### Never Buffer a Stream to Display It

```python
# ❌ Defeats the purpose of streaming — user sees nothing until full response
tokens = list(writer.write_section(ctx, section, chunks))
st.write("".join(tokens))

# ✅ Streams token-by-token to the UI
st.write_stream(writer.write_section(ctx, section, chunks))
```

---

## 9. Writing a Plugin

Plugins are the only way DocForge ingests a file type or produces an output format.
Every plugin is a ZIP file with a `plugin.toml` manifest and one or two Python files.

### Ingestor Plugin — Full Example

```
my_csv_plugin/
├── plugin.toml
├── __init__.py
└── ingestor.py
```

**`plugin.toml`**
```toml
[plugin]
name        = "csv-ingestor"
version     = "1.0.0"
author      = "Your Name"
description = "CSV file ingestion with row-level chunking"

[ingestor]
extensions   = [".csv", ".tsv"]
entry_module = "ingestor"
entry_class  = "CsvIngestor"
```

**`ingestor.py`**
```python
from __future__ import annotations
from pathlib import Path
import csv
from plugins.base import IngestorPlugin, ParsedDocument, ParsedPage
from core.errors import PluginParseError

class CsvIngestor(IngestorPlugin):

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def parse(self, path: Path, config: dict) -> ParsedDocument:
        """Parse a CSV/TSV file. Each row becomes a line of text."""
        try:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader  = csv.DictReader(f, delimiter=delimiter)
                rows    = list(reader)
                headers = reader.fieldnames or []

            # Represent the whole CSV as a single page of text
            lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
            text  = "\n".join(lines)

            page = ParsedPage(page_number=1, text=text, tables=[], images=[])
            return ParsedDocument(source_path=path, pages=[page],
                                  metadata={"headers": headers, "rows": len(rows)})

        except Exception as e:
            raise PluginParseError(f"CSV parse failed for {path.name}: {e}") from e

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: int) -> list[str]:
        """Chunk by lines (rows). Each chunk is at most chunk_size characters."""
        all_lines = doc.pages[0].text.splitlines()
        chunks, current, current_len = [], [], 0

        for line in all_lines:
            line_len = len(line)
            if current_len + line_len > chunk_size and current:
                chunks.append("\n".join(current))
                # Keep last N lines for overlap
                overlap_chars = 0
                overlap_lines = []
                for l in reversed(current):
                    overlap_chars += len(l)
                    overlap_lines.insert(0, l)
                    if overlap_chars >= overlap:
                        break
                current     = overlap_lines
                current_len = overlap_chars
            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current))
        return chunks
```

### Plugin Rules

- Always inherit from `IngestorPlugin` or `OutputPlugin` — never skip the base class.
- `parse()` must raise `PluginParseError` for file-level failures, nothing else.
- `chunk()` is pure: no I/O, no LLM calls, no logging. Just text manipulation.
- `write()` (OutputPlugin) must raise `PluginWriteError` for write failures, nothing else.
- Every plugin gets its own `log = get_logger(__name__)` — do not pass loggers in.
- Test your plugin's `chunk()` method with the full range of chunk sizes (64, 256, 512, 1024).

---

## 10. Writing an Agent

Agents in DocForge are thin classes that own one LLM interaction.
They inherit from `BaseAgent`, call `self._call_llm()` or `self._stream_llm()`,
and never touch the vector store, file cache, or output writers directly.

### Agent Rules

- One class, one LLM interaction type (plan, retrieve, write, review, parse, analyze).
- Never call another agent from inside an agent. The pipeline orchestrates sequencing.
- Never retrieve chunks inside a writer or reviewer. The retriever feeds chunks in.
- Context budget is enforced by the caller — agents receive pre-filtered chunks.
- LLM system prompts are module-level constants in SCREAMING_SNAKE_CASE.
- When the LLM must return structured data, prompt for JSON only and parse strictly.

### JSON-Output Agent Pattern

Used by `PlannerAgent` and `AnalyzerAgent` — any agent that needs structured output.

```python
# agents/itd/planner.py

PLANNER_SYSTEM = """You are a senior technical writer.
Given a user instruction, produce a structured document plan.
Respond ONLY with valid JSON: {"sections": [{"title": "...", "instructions": "..."}]}
No preamble. No markdown fences. JSON only."""

class PlannerAgent(BaseAgent):

    def plan_document(self, ctx: AgentContext) -> list[dict]:
        self.log.info("planner.start", instruction=ctx.instruction[:80])

        raw = self._call_llm(prompt=self._build_prompt(ctx), system=PLANNER_SYSTEM)

        # Always strip markdown fences before parsing — LLMs sometimes add them
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()

        try:
            sections = json.loads(clean).get("sections", [])
        except json.JSONDecodeError as e:
            self.log.error("planner.parse_failed", error=str(e), raw=raw[:200])
            raise AgentError(f"Planner returned invalid JSON: {e}") from e

        if not sections:
            raise AgentError("Planner returned an empty section list")

        self.log.info("planner.done", sections=len(sections))
        return sections

    def _build_prompt(self, ctx: AgentContext) -> str:
        return f"""Instruction: {ctx.instruction}

Generate a complete section plan for a professional engineering document.
Apply standard structure: Executive Summary, Scope, Requirements,
Design, Implementation, Testing, Appendix — as applicable.
Fill any gaps with professional standard sections for this document type."""
```

### Streaming-Output Agent Pattern

Used by `WriterAgent` in both ITD and CTD — any agent that streams to the UI.

```python
# agents/itd/writer.py

class WriterAgent(BaseAgent):

    def write_section(self, ctx: AgentContext, section: dict, chunks: list[dict]):
        """Generator. Yields tokens. Access self.result after exhausting."""
        self.log.info("writer.section.start", title=section["title"])
        prompt        = self._build_prompt(section, chunks)
        content_parts = []

        for token in self._stream_llm(prompt=prompt, system=WRITER_SYSTEM):
            content_parts.append(token)
            yield token

        self.result = GeneratedSection(
            title   = section["title"],
            content = "".join(content_parts),
            sources = [c["chunk_id"] for c in chunks],
        )
        self.log.info("writer.section.done",
                      title=section["title"], chars=len(self.result.content))

    def _build_prompt(self, section: dict, chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['filename']}]\n{c['text']}" for c in chunks
        )
        return f"""## Section: {section['title']}
Instructions: {section['instructions']}

## Context:
{context}

Write only the section content. No title. No preamble."""
```

---

## 11. Docker Development Patterns

### Detecting the Docker Environment

Some code paths behave differently in Docker (TUI is disabled, Ollama URL changes).
Always use the canonical detection function — never check env vars inline.

```python
from core.env import is_running_in_docker

if is_running_in_docker():
    log.info("startup.docker_detected")
    # Skip TUI, use env var for Ollama URL
```

### Config in Docker

Ollama's URL is `http://ollama:11434` inside Docker (the service name on the bridge network),
but `http://localhost:11434` on the host. This is handled entirely via the env var
`DOCFORGE_OLLAMA__BASE_URL` set in `docker-compose.yml`. No code change needed.

```yaml
# docker-compose.yml — already handles it
environment:
  - DOCFORGE_OLLAMA__BASE_URL=http://ollama:11434
```

Your code always calls `get_config().ollama.base_url` — never hardcode either URL.

### File Paths in Docker

The bind mounts in `docker-compose.yml` map host paths to container paths:

| Host | Container |
|------|-----------|
| `./data` | `/app/data` |
| `./config` | `/app/config` |
| `./plugins` | `/app/plugins` |

All DocForge code uses relative paths (`Path("data/projects/...")`) from the working
directory `/app`. This works identically on host (conda env) and in Docker.
Never hardcode `/app/` in Python code — it breaks the conda dev workflow.

### Rebuilding vs Restarting in Dev

```bash
# Source code changed (Python file) — restart only (hot reload handles it in dev mode)
docker compose restart docforge

# requirements.txt changed — must rebuild the image
docker compose up --build docforge

# plugin.toml or new plugin ZIP added — restart only
docker compose restart docforge

# docker-compose.yml changed — down and up
docker compose down && docker compose up -d
```

### Accessing Logs During Development

```bash
# Live logs from the app (most useful)
docker compose logs -f docforge

# Live logs from Ollama (model loading, inference)
docker compose logs -f ollama

# One-time model pull status
docker compose logs model-puller

# Structured logs written to file (visible on host because ./data is bind-mounted)
Get-Content data\logs\docforge.log -Wait     # Windows PowerShell
# tail -f data/logs/docforge.log             # Linux/macOS
```

---

## 12. Testing Conventions

### Structure

```
tests/
  unit/
    test_ocr_corrector.py        # pure functions — no I/O
    test_chunker.py
    test_template_analyzer.py
  integration/
    test_ingest_pipeline.py      # real files, mocked Ollama
    test_embedding_store.py      # real LanceDB in temp dir
    test_plugin_loader.py
  fixtures/
    sample.pdf
    sample.docx
    sample_template.docx
    bad_scan.pdf
    conftest.py
```

### Unit Test Pattern

Unit tests cover pure functions. No I/O, no LLM calls, no Ollama.

```python
# tests/unit/test_ocr_corrector.py
from ocr.corrector import correct_ocr_output

def test_removes_short_noise_lines():
    raw    = "A\nThis is real content.\nB\n"
    result = correct_ocr_output(raw)
    assert "This is real content." in result
    assert "A\n" not in result

def test_rejoins_hyphenated_words():
    raw    = "docu-\nment"
    result = correct_ocr_output(raw)
    assert "document" in result

def test_normalizes_extra_whitespace():
    raw    = "word    word\n\n\n\nparagraph"
    result = correct_ocr_output(raw)
    assert "  " not in result
    assert "\n\n\n" not in result
```

### Integration Test Pattern

Integration tests use real files and a real LanceDB in a temp directory.
Mock only Ollama (no network calls in CI).

```python
# tests/integration/test_ingest_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import patch
from ingest.pipeline import ingest_file

@pytest.fixture
def temp_project(tmp_path):
    # Override data dir to tmp_path so tests don't pollute real data
    return str(tmp_path)

@patch("embeddings.client.embed_texts", return_value=[[0.1] * 768])
def test_ingest_pdf_returns_ingested_file(mock_embed, temp_project, tmp_path):
    # Arrange
    sample_pdf = Path("tests/fixtures/sample.pdf")

    # Act
    result = ingest_file(sample_pdf, project_name=temp_project)

    # Assert
    assert result.filename == "sample.pdf"
    assert result.chunk_count > 0
    assert result.file_id != ""
    mock_embed.assert_called()
```

### Naming Tests

```
test_{what}_{condition}_{expected_outcome}

test_ingest_file_missing_plugin_raises_ingest_error
test_correct_ocr_output_hyphenated_words_rejoins_them
test_plan_document_empty_instruction_raises_agent_error
test_search_below_min_score_returns_empty_list
```

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths       = ["tests"]
asyncio_mode    = "auto"
filterwarnings  = ["ignore::DeprecationWarning"]

[tool.ruff]
target-version = "py311"
line-length    = 100
select         = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version         = "3.11"
strict                 = true
ignore_missing_imports = true
```

---

## 13. Quick Reference

```
NAMING
  Files/modules      snake_case, verb-noun, one job per file
  Functions          verb_noun()  →  ingest_file, analyze_template, embed_texts
  Classes            PascalCase, role-based  →  PlannerAgent, EmbeddingStore
  Constants          SCREAMING_SNAKE_CASE in core/constants.py
  Private helpers    _prefix, bottom of file

FUNCTION SHAPE
  → from __future__ import annotations at top of every file
  → typed args + return type, always
  → docstring: Args / Returns / Raises
  → log.info("fn.start", ...)  first line
  → log.info("fn.done", ...)   last line before return
  → log.error("fn.failed", error=str(e))  in every except block
  → raise DocForgeError subclass, never raw library errors
  → guard clauses first, happy path unindented

MODULE LAYOUT
  from __future__ import annotations
  stdlib → third-party → internal
  log = get_logger(__name__)
  constants
  dataclasses
  public functions
  classes
  private helpers _fn()

CONFIG
  → get_config() inside functions, never at module level
  → save_user_config() for runtime user changes
  → DOCFORGE_SECTION__KEY env vars override TOML in Docker

LOGGING
  → log = get_logger(__name__)  — one per module
  → "function.event" as first arg, key=value for everything else
  → never f-strings, never str concatenation
  → set_module_logging("module", enabled=bool) to toggle at runtime

ERRORS
  DocForgeError → IngestError / OcrError / PluginError / EmbeddingError
                  StoreError / AgentError / TemplateAnalysisError
                  OutputError / LLMError / LLMTimeoutError
  → wrap at function boundary with  raise MyError(...) from e
  → pipeline decides: halt (LLM) / degrade (retrieval) / retry (timeout)

STREAMING
  → writer.write_section() is a generator — yield tokens immediately
  → st.write_stream(generator) consumes it
  → access writer.result after the generator is exhausted
  → never buffer a stream to display it

PLUGINS
  → inherit IngestorPlugin or OutputPlugin
  → parse() raises PluginParseError only
  → chunk() is pure — no I/O, no LLM, no logging calls
  → always has its own log = get_logger(__name__)

AGENTS
  → inherit BaseAgent, call self._call_llm() or self._stream_llm()
  → one class, one LLM interaction
  → JSON-output agents: strip markdown fences, parse strictly, raise AgentError
  → never call another agent; never touch vector store or output directly

DOCKER
  → is_running_in_docker() to detect container context
  → all paths relative from /app — never hardcode /app/
  → Ollama URL comes from DOCFORGE_OLLAMA__BASE_URL — never hardcode
  → rebuild on requirements.txt change; restart-only on code change (dev mode)

TESTING
  → unit: pure functions, no I/O, no mocking of DocForge internals
  → integration: real files + real LanceDB in tmp_path, mock embed_texts
  → test names: test_{what}_{condition}_{expected_outcome}
```
