# DocForge — Agentic RAG Document Generator
## Architecture & Developer Reference v1.0
### Python 3.11 · Ollama · LanceDB · Streamlit · Windows-first, Cross-platform

---

> **What this document is:** A complete technical blueprint. Every module, every function signature,
> every tech choice with a reason, and every system boundary is defined here.
> This is the reference before a single line of code is written.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Design Goals](#2-design-goals)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Configuration System](#5-configuration-system)
6. [Plugin System](#6-plugin-system)
7. [Document Ingestion Pipeline](#7-document-ingestion-pipeline)
8. [OCR & Preprocessing Pipeline](#8-ocr--preprocessing-pipeline)
9. [Template Engine](#9-template-engine)
10. [Embedding & Vector Storage](#10-embedding--vector-storage)
11. [Agent System — Instruction-to-Document Mode](#11-agent-system--instruction-to-document-mode)
12. [Agent System — Code-to-Document Mode](#12-agent-system--code-to-document-mode)
13. [Output Generation](#13-output-generation)
14. [Encryption](#14-encryption)
15. [Multi-Tenancy (Project Namespacing)](#15-multi-tenancy-project-namespacing)
16. [Streamlit UI Architecture](#16-streamlit-ui-architecture)
17. [Launcher Architecture](#17-launcher-architecture)
18. [Logging Architecture](#18-logging-architecture)
19. [Developer Setup (Conda + Windows)](#19-developer-setup-conda--windows)
20. [Requirements Management](#20-requirements-management)
21. [Docker Architecture](#21-docker-architecture)
22. [Dockerfile](#22-dockerfile)
23. [Docker Compose — Full Stack](#23-docker-compose--full-stack)
24. [Docker Compose — Dev Overrides](#24-docker-compose--dev-overrides)
25. [Docker Compose — GPU Override](#25-docker-compose--gpu-override)
26. [Entrypoint & Model Pull](#26-entrypoint--model-pull)
27. [Environment Variables & Config Bridge](#27-environment-variables--config-bridge)
28. [Docker-Aware Launcher](#28-docker-aware-launcher)
29. [Dev Workflow with Docker](#29-dev-workflow-with-docker)

---

## 1. System Overview

DocForge is a **local, offline-first, multimodal agentic document generator**. It is not a chatbot.
It accepts source materials (documents, codebases, templates) and produces structured, production-ready
engineering documents that conform to an uploaded template in style, layout, font, header, footer,
and letterhead.

### Two Operational Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                          DocForge                               │
│                                                                 │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │  INSTRUCTION-TO-DOC     │   │   CODE-TO-DOC               │  │
│  │                         │   │                             │  │
│  │  Input: source docs,    │   │  Input: Python/C++/Java     │  │
│  │  knowledge base,        │   │  source files, headers,     │  │
│  │  instructions           │   │  READMEs, comments          │  │
│  │                         │   │                             │  │
│  │  Output: SRS, TDD,      │   │  Output: API docs, design   │  │
│  │  Design Docs, Req Docs  │   │  docs, TDD, architecture    │  │
│  └─────────────────────────┘   └─────────────────────────────┘  │
│                                                                 │
│  Both modes: Template-adhering output · Streaming · Agents     │
└─────────────────────────────────────────────────────────────────┘
```

### What Each Mode Produces

| Mode | Document Types |
|------|---------------|
| Instruction-to-Doc | SRS, Requirements Doc, TDD, Design Doc, Architecture Doc |
| Code-to-Doc | API Reference, Technical Design Doc, Module Doc, Deployment Runbook |

---

## 2. Design Goals

| Goal | Implementation Contract |
|------|------------------------|
| **Fast on CPU** | LanceDB (Arrow columnar), nomic-embed-text (small model), streaming output, cached embeddings |
| **No context overload** | Chunk-aware retrieval, per-agent context budgets, section-by-section generation |
| **Template fidelity** | Font, style, heading hierarchy, header/footer, tables, letterhead are all extracted and applied |
| **Replaceable components** | Every model, retriever, and parser is behind an interface — swap via config |
| **Full observability** | Every agent step, every retrieval call, every LLM request is logged with a traceable ID |
| **Plugin-first ingestion** | No file type is hardcoded — all parsing is a plugin |
| **Secure** | Files encrypted at rest with AES-128 Fernet; project data fully namespaced |

---

## 3. Tech Stack

### Runtime

| Library | Version | Why |
|---------|---------|-----|
| `Python` | 3.11 | Most compatible — 3.10 EOL soon, 3.12 breaks several ML libs, 3.13 too new |
| `ollama` (Python SDK) | latest | Async streaming API to local Ollama server; no GPU required |
| `lancedb` | ≥0.6 | Embedded vector DB, Apache Arrow format, zero server, fastest CPU vector search |
| `sentence-transformers` | ≥2.6 | Fallback CPU embeddings if Ollama nomic model not pulled |

### UI

| Library | Version | Why |
|---------|---------|-----|
| `streamlit` | ≥1.35 | User-preferred; `st.write_stream()` + `st.status()` handle streaming and agent display well |
| `streamlit-aggrid` | latest | Fast, sortable, filterable JSON tables in the Files tab |
| `streamlit-extras` | latest | `stx.tab_bar`, fragment refresh, sticky headers |
| `watchdog` | latest | File system change detection for hot-reload of plugins |

### Launcher / TUI

| Library | Version | Why |
|---------|---------|-----|
| `textual` | ≥0.55 | Full TUI framework — checkboxes, text inputs, arrow key nav, panels, all built-in |
| `typer` | ≥0.12 | CLI flags with type validation, --help auto-generation |
| `rich` | ≥13.7 | Color-coded terminal output, progress bars, service status trees |

### Document Ingestion

| Library | Version | Why |
|---------|---------|-----|
| `pymupdf` (fitz) | ≥1.24 | Fastest PDF text + layout extraction, no OCR required for text PDFs |
| `pdfplumber` | ≥0.11 | Table extraction from PDFs (better table detection than PyMuPDF) |
| `pdf2image` | ≥1.17 | Converts PDF pages to images for OCR pipeline |
| `python-docx` | ≥1.1 | DOCX read and write |
| `python-pptx` | ≥0.6 | PPTX reading |
| `openpyxl` | ≥3.1 | Excel read and write |
| `beautifulsoup4` | ≥4.12 | HTML parsing |
| `Pillow` | ≥10.3 | Image loading for image-only documents |

### OCR & Preprocessing

| Library | Version | Why |
|---------|---------|-----|
| `pytesseract` | ≥0.3 | Tesseract OCR binding — reliable, CPU-fast, proven |
| `opencv-python-headless` | ≥4.9 | Deskew, denoise, binarize before OCR; headless (no GUI dep) |
| `imutils` | ≥0.5 | Utility belt for OpenCV operations |

### Code Parsing (Code-to-Doc)

| Library | Version | Why |
|---------|---------|-----|
| `tree-sitter` | ≥0.22 | AST parsing for Python, C/C++, Java — extract classes, functions, docstrings |
| `tree-sitter-python` | latest | Python grammar |
| `tree-sitter-c` | latest | C/C++ grammar |
| `tree-sitter-java` | latest | Java grammar |

### Output Generation

| Library | Version | Why |
|---------|---------|-----|
| `python-docx` | ≥1.1 | DOCX generation (same lib as reading) |
| `weasyprint` | ≥62 | HTML-to-PDF; handles CSS styling better than reportlab for templates |
| `jinja2` | ≥3.1 | HTML template rendering for PDF conversion step |
| `openpyxl` | ≥3.1 | Excel output |
| `Pillow` | ≥10.3 | Image output from rendered pages |

### Storage & Config

| Library | Version | Why |
|---------|---------|-----|
| `lancedb` | ≥0.6 | Vector storage (see above) |
| `pyarrow` | ≥15 | Required by LanceDB; Arrow schema for chunks |
| `cryptography` | ≥42 | Fernet AES-128 encryption for file cache |
| `pydantic` | ≥2.6 | Config models, schema validation |
| `pydantic-settings` | ≥2.2 | Load config from TOML + env vars |
| `tomllib` | built-in | Python 3.11 built-in TOML parser |
| `tomli-w` | ≥1.0 | TOML writer (built-in only reads) |

### Logging

| Library | Version | Why |
|---------|---------|-----|
| `structlog` | ≥24.1 | Structured key=value logging, module-level filtering |
| `rich` | ≥13.7 | Color-coded terminal rendering of structured logs |

### Dev / Build

| Library | Version | Why |
|---------|---------|-----|
| `conda` | ≥23 | Environment management on Windows (handles non-pip deps like Tesseract, poppler) |
| `pytest` | ≥8 | Testing |
| `pytest-asyncio` | ≥0.23 | Async test support |
| `ruff` | latest | Linting + formatting (replaces black + flake8) |
| `mypy` | latest | Type checking |

### Models via Ollama

| Model | Role | Why |
|-------|------|-----|
| `gemma3:7b` | Instruction-to-Doc writer | Strong general reasoning, 7B fits in 8GB RAM |
| `qwen2.5-coder:7b` | Code-to-Doc writer | Outperforms Gemma at code comprehension at same size |
| `nomic-embed-text` | Embeddings | 137M params, extremely fast on CPU, 768-dim vectors |

---

## 4. Project Structure

```
docforge/
│
├── launcher.py                    # Entry point: Textual TUI + Typer CLI
│
├── app/
│   └── main.py                    # Streamlit app entry point
│
├── core/
│   ├── __init__.py
│   ├── config.py                  # Pydantic settings, TOML loader
│   ├── constants.py               # All system-wide constants
│   ├── errors.py                  # Domain exception hierarchy
│   └── logger.py                  # structlog + Rich setup, module switches
│
├── plugins/
│   ├── __init__.py
│   ├── registry.py                # Plugin discovery, registration, validation
│   ├── loader.py                  # ZIP install, importlib loading
│   ├── base.py                    # IngestorPlugin, OutputPlugin base classes
│   └── installed/                 # Extracted plugin ZIPs live here
│       └── .gitkeep
│
├── ingest/
│   ├── __init__.py
│   ├── pipeline.py                # Orchestrates full ingestion flow
│   ├── file_cache.py              # Encrypted file cache manager
│   ├── chunker.py                 # Context-aware chunking utilities
│   └── registry.py                # Maps extension → plugin
│
├── ocr/
│   ├── __init__.py
│   ├── detector.py                # Decides: text PDF vs scanned
│   ├── preprocessor.py            # OpenCV: deskew, denoise, binarize
│   ├── extractor.py               # Tesseract OCR caller
│   └── corrector.py               # Post-OCR correction (layout, spacing, artifacts)
│
├── template/
│   ├── __init__.py
│   ├── analyzer.py                # Infers template structure from example doc
│   ├── store.py                   # Saves/loads templates per project
│   └── renderer.py                # Applies template styles to generated content
│
├── embeddings/
│   ├── __init__.py
│   ├── client.py                  # Ollama embedding calls
│   └── store.py                   # LanceDB operations (upsert, search, delete)
│
├── agents/
│   ├── __init__.py
│   ├── base.py                    # BaseAgent class, context budget management
│   ├── itd/                       # Instruction-to-Document agents
│   │   ├── __init__.py
│   │   ├── planner.py             # Breaks instruction into section plan
│   │   ├── retriever.py           # Fetches relevant chunks per section
│   │   ├── writer.py              # Generates section content (streaming)
│   │   └── reviewer.py            # Reviews + refines each section
│   └── ctd/                       # Code-to-Document agents
│       ├── __init__.py
│       ├── parser.py              # tree-sitter AST extraction
│       ├── analyzer.py            # Understands code intent and relationships
│       ├── writer.py              # Generates doc section from code analysis
│       └── reviewer.py            # Reviews for technical accuracy
│
├── llm/
│   ├── __init__.py
│   ├── client.py                  # Async Ollama streaming client
│   └── prompt_builder.py          # Assembles system + context + instruction prompts
│
├── output/
│   ├── __init__.py
│   ├── pipeline.py                # Orchestrates output generation
│   ├── docx_writer.py             # python-docx output with template styles
│   ├── pdf_writer.py              # weasyprint HTML-to-PDF
│   ├── md_writer.py               # Markdown output
│   ├── excel_writer.py            # openpyxl output
│   └── image_writer.py            # Pillow image output
│
├── security/
│   ├── __init__.py
│   └── encryption.py              # Fernet key management, encrypt/decrypt
│
├── projects/
│   ├── __init__.py
│   └── manager.py                 # Project namespace CRUD
│
├── ui/
│   ├── __init__.py
│   ├── pages/
│   │   ├── upload.py              # Upload tab
│   │   ├── files.py               # Files tab (AgGrid table)
│   │   ├── generate.py            # Generate tab (streaming agent display)
│   │   ├── logs.py                # Logs viewer tab
│   │   └── settings.py            # Settings tab (models, logging switches, plugins)
│   └── components/
│       ├── agent_panel.py         # st.status() agent activity component
│       ├── log_panel.py           # Log display component
│       └── file_table.py          # AgGrid file table component
│
├── tui/
│   ├── __init__.py
│   ├── app.py                     # Textual TUI application
│   ├── screens/
│   │   ├── main_screen.py         # Main launcher dashboard
│   │   └── log_screen.py          # Live log output screen
│   └── widgets/
│       ├── service_status.py      # Service health indicators
│       ├── log_switches.py        # Checkbox panel for logging modules
│       └── service_log_panel.py   # Scrolling log panel per service
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── data/
│   ├── projects/                  # Per-project data (namespaced)
│   │   └── {project_name}/
│   │       ├── cache/             # Encrypted file cache
│   │       ├── vectors/           # LanceDB tables
│   │       ├── templates/         # Uploaded templates
│   │       └── outputs/           # Generated documents
│   └── logs/                      # Rotating log files
│
├── config/
│   ├── default.toml               # Default configuration
│   └── user.toml                  # User overrides (gitignored)
│
├── environment.yml                # Conda environment (dev)
├── requirements.txt               # Production pip requirements
├── requirements-dev.txt           # Dev-only dependencies
└── pyproject.toml                 # Tool config (ruff, mypy, pytest)
```

---

## 5. Configuration System

### `config/default.toml`

```toml
[app]
name        = "DocForge"
version     = "1.0.0"
data_dir    = "data"
log_dir     = "data/logs"
log_level   = "INFO"

[projects]
default_project = "default"

[ollama]
base_url           = "http://localhost:11434"
timeout_sec        = 120
stream             = true
connect_retry_max  = 5
connect_retry_wait = 3

[models]
itd_model        = "gemma3:7b"
ctd_model        = "qwen2.5-coder:7b"
embedding_model  = "nomic-embed-text"

[agents]
itd_context_budget_tokens = 3000
ctd_context_budget_tokens = 3000
max_retrieval_chunks      = 8
min_relevance_score       = 0.68
reviewer_enabled          = true

[ocr]
engine              = "tesseract"
dpi                 = 300
preprocessing       = true
deskew              = true
denoise             = true
binarize            = true
post_correction     = true

[encryption]
enabled    = true
key_file   = "data/.keystore"

[launcher]
service_retry_max  = 5
service_retry_wait = 3
ollama_start_cmd   = "ollama serve"

[logging.modules]
ingest     = true
ocr        = true
embeddings = true
agents     = true
llm        = true
output     = true
template   = true
plugins    = true
```

### `core/config.py`

```python
from __future__ import annotations
import tomllib
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ── Sub-models ────────────────────────────────────────────────────────────────

class OllamaConfig(BaseModel):
    base_url: str           = "http://localhost:11434"
    timeout_sec: int        = 120
    stream: bool            = True
    connect_retry_max: int  = 5
    connect_retry_wait: int = 3

class ModelsConfig(BaseModel):
    itd_model: str       = "gemma3:7b"
    ctd_model: str       = "qwen2.5-coder:7b"
    embedding_model: str = "nomic-embed-text"

class AgentConfig(BaseModel):
    itd_context_budget_tokens: int = 3000
    ctd_context_budget_tokens: int = 3000
    max_retrieval_chunks: int      = 8
    min_relevance_score: float     = 0.68
    reviewer_enabled: bool         = True

class OcrConfig(BaseModel):
    engine: str          = "tesseract"
    dpi: int             = 300
    preprocessing: bool  = True
    deskew: bool         = True
    denoise: bool        = True
    binarize: bool       = True
    post_correction: bool = True

class EncryptionConfig(BaseModel):
    enabled: bool  = True
    key_file: str  = "data/.keystore"

class LauncherConfig(BaseModel):
    service_retry_max: int  = 5
    service_retry_wait: int = 3
    ollama_start_cmd: str   = "ollama serve"

class LoggingModulesConfig(BaseModel):
    ingest: bool     = True
    ocr: bool        = True
    embeddings: bool = True
    agents: bool     = True
    llm: bool        = True
    output: bool     = True
    template: bool   = True
    plugins: bool    = True

# ── Root config ───────────────────────────────────────────────────────────────

class DocForgeConfig(BaseModel):
    ollama: OllamaConfig         = OllamaConfig()
    models: ModelsConfig         = ModelsConfig()
    agents: AgentConfig          = AgentConfig()
    ocr: OcrConfig               = OcrConfig()
    encryption: EncryptionConfig = EncryptionConfig()
    launcher: LauncherConfig     = LauncherConfig()
    log_modules: LoggingModulesConfig = LoggingModulesConfig()

# ── Loader ────────────────────────────────────────────────────────────────────

_config: DocForgeConfig | None = None

def load_config(path: Path = Path("config/default.toml")) -> DocForgeConfig:
    """Load and merge default + user TOML configs into DocForgeConfig."""
    global _config
    default_data: dict = {}
    user_data: dict    = {}

    if path.exists():
        with open(path, "rb") as f:
            default_data = tomllib.load(f)

    user_path = Path("config/user.toml")
    if user_path.exists():
        with open(user_path, "rb") as f:
            user_data = tomllib.load(f)

    merged = _deep_merge(default_data, user_data)
    _config = DocForgeConfig.model_validate(merged)
    return _config

def get_config() -> DocForgeConfig:
    """Return the singleton config. Raises if not yet loaded."""
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config

def save_user_config(updates: dict) -> None:
    """Merge updates into user.toml and persist."""
    import tomli_w
    user_path = Path("config/user.toml")
    existing: dict = {}
    if user_path.exists():
        with open(user_path, "rb") as f:
            existing = tomllib.load(f)
    merged = _deep_merge(existing, updates)
    with open(user_path, "wb") as f:
        tomli_w.dump(merged, f)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

---

## 6. Plugin System

Plugins are the only way file types are ingested or output formats produced.
No file format is hardcoded into the core pipeline.

### Plugin ZIP Structure

```
my_plugin.zip
├── plugin.toml           # Manifest
├── __init__.py
├── ingestor.py           # Optional: implements IngestorPlugin
└── output_writer.py      # Optional: implements OutputPlugin
```

### `plugin.toml` Manifest

```toml
[plugin]
name        = "pdf-ingestor"
version     = "1.0.0"
author      = "DocForge Team"
description = "PDF text and table extraction"

[ingestor]
extensions    = [".pdf"]
entry_module  = "ingestor"
entry_class   = "PdfIngestor"

[output_writer]
formats       = ["pdf"]
entry_module  = "output_writer"
entry_class   = "PdfOutputWriter"
```

### `plugins/base.py` — Interface Contracts

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ParsedDocument:
    """Result of ingesting a document via a plugin."""
    source_path: Path
    pages: list[ParsedPage]
    metadata: dict

@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: list[list[list[str]]]   # rows → cells
    images: list[bytes]              # raw image bytes per embedded image

class IngestorPlugin(ABC):
    """
    Contract every ingestor plugin must fulfil.
    One class handles one or more file extensions.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of extensions this plugin handles. e.g. ['.pdf']"""

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this plugin can parse the given file."""

    @abstractmethod
    def parse(self, path: Path, config: dict) -> ParsedDocument:
        """
        Parse the file into a ParsedDocument.

        Args:
            path:   Absolute path to the source file.
            config: Plugin-specific config from user.toml [plugins.<name>].

        Returns:
            ParsedDocument with pages, tables, and images.

        Raises:
            PluginParseError: If the file cannot be parsed.
        """

    @abstractmethod
    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: int) -> list[str]:
        """
        Chunk the parsed document into embeddable text segments.

        Args:
            doc:        Parsed document output.
            chunk_size: Token target per chunk.
            overlap:    Token overlap between adjacent chunks.

        Returns:
            List of text chunks ordered by document position.
        """

class OutputPlugin(ABC):
    """Contract every output format plugin must fulfil."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of output format identifiers. e.g. ['pdf']"""

    @abstractmethod
    def write(
        self,
        sections: list[GeneratedSection],
        template_styles: TemplateStyles | None,
        output_path: Path,
    ) -> Path:
        """
        Write generated sections to an output file.

        Args:
            sections:        Ordered list of generated document sections.
            template_styles: Extracted styles from the template, or None.
            output_path:     Destination file path.

        Returns:
            Resolved path of the written file.

        Raises:
            PluginWriteError: If writing fails.
        """
```

### `plugins/loader.py`

```python
import zipfile
import shutil
import importlib.util
from pathlib import Path
from plugins.base import IngestorPlugin, OutputPlugin
from plugins.registry import PluginRegistry
from core.logger import get_logger
from core.errors import PluginInstallError, PluginLoadError

log = get_logger(__name__)

INSTALL_DIR = Path("plugins/installed")

def install_plugin_from_zip(zip_path: Path) -> str:
    """
    Extract and register a plugin from a ZIP file.

    Args:
        zip_path: Path to the uploaded plugin ZIP.

    Returns:
        Plugin name as declared in plugin.toml.

    Raises:
        PluginInstallError: If ZIP is malformed or manifest is missing.
    """
    log.info("plugin.install.start", zip=zip_path.name)

    if not zipfile.is_zipfile(zip_path):
        raise PluginInstallError(f"Not a valid ZIP: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "plugin.toml" not in names:
            raise PluginInstallError("Missing plugin.toml manifest in ZIP")

        manifest = _read_manifest_from_zip(zf)
        plugin_name = manifest["plugin"]["name"]
        dest = INSTALL_DIR / plugin_name

        if dest.exists():
            shutil.rmtree(dest)
        zf.extractall(dest)

    plugin = load_plugin(dest)
    PluginRegistry.register(plugin)

    log.info("plugin.install.done", plugin=plugin_name)
    return plugin_name

def load_plugin(plugin_dir: Path) -> IngestorPlugin | OutputPlugin:
    """
    Dynamically load a plugin from its installed directory.

    Args:
        plugin_dir: Path to the extracted plugin directory.

    Returns:
        Instantiated plugin object.

    Raises:
        PluginLoadError: If the module or class cannot be loaded.
    """
    import tomllib
    manifest_path = plugin_dir / "plugin.toml"
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)

    if "ingestor" in manifest:
        return _load_class(plugin_dir, manifest["ingestor"])
    elif "output_writer" in manifest:
        return _load_class(plugin_dir, manifest["output_writer"])
    else:
        raise PluginLoadError(f"plugin.toml has no [ingestor] or [output_writer]: {plugin_dir}")

def load_all_installed_plugins() -> list:
    """Scan plugins/installed/ and load every valid plugin."""
    loaded = []
    for plugin_dir in INSTALL_DIR.iterdir():
        if plugin_dir.is_dir() and (plugin_dir / "plugin.toml").exists():
            try:
                plugin = load_plugin(plugin_dir)
                PluginRegistry.register(plugin)
                loaded.append(plugin)
                log.info("plugin.loaded", plugin=plugin_dir.name)
            except Exception as e:
                log.error("plugin.load.failed", plugin=plugin_dir.name, error=str(e))
    return loaded

def _load_class(plugin_dir: Path, section: dict) -> object:
    module_path = plugin_dir / f"{section['entry_module']}.py"
    spec = importlib.util.spec_from_file_location(section["entry_module"], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, section["entry_class"])
    return cls()

def _read_manifest_from_zip(zf: zipfile.ZipFile) -> dict:
    import tomllib
    with zf.open("plugin.toml") as f:
        return tomllib.load(f)
```

---

## 7. Document Ingestion Pipeline

### `ingest/pipeline.py`

```python
from __future__ import annotations
import hashlib
import uuid
from pathlib import Path
from dataclasses import dataclass
from plugins.registry import PluginRegistry
from ingest.file_cache import FileCacheManager
from ingest.chunker import chunk_text
from embeddings.store import EmbeddingStore
from ocr.detector import needs_ocr
from ocr.preprocessor import preprocess_image
from ocr.extractor import extract_text_with_tesseract
from ocr.corrector import correct_ocr_output
from core.logger import get_logger
from core.errors import IngestError

log = get_logger(__name__)

@dataclass
class IngestedFile:
    file_id: str
    filename: str
    source_path: Path
    chunk_count: int
    page_count: int
    has_tables: bool
    was_ocrd: bool

def ingest_file(
    source_path: Path,
    project_name: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> IngestedFile:
    """
    Full ingestion pipeline for a single file.

    Detects file type → routes to plugin → OCR if needed →
    corrects extraction → chunks → embeds → stores in LanceDB.

    Args:
        source_path:  Absolute path to the file to ingest.
        project_name: Project namespace for storage isolation.
        chunk_size:   Target token count per chunk.
        overlap:      Token overlap between adjacent chunks.

    Returns:
        IngestedFile metadata record for display in the Files tab.

    Raises:
        IngestError: If no plugin handles the file type, or any stage fails.
    """
    log.info("ingest.start", file=source_path.name, project=project_name)

    file_id = _generate_file_id(source_path)
    plugin  = PluginRegistry.get_ingestor_for(source_path.suffix)

    if plugin is None:
        raise IngestError(f"No ingestor plugin for extension: {source_path.suffix}")

    # ── Parse ─────────────────────────────────────────────────────────────────
    parsed = plugin.parse(source_path, config={})
    was_ocrd = False

    # ── OCR override if text extraction is empty or low quality ───────────────
    if needs_ocr(parsed):
        log.info("ingest.ocr.required", file=source_path.name)
        parsed = _run_ocr_pipeline(source_path, parsed)
        was_ocrd = True

    # ── Chunk ─────────────────────────────────────────────────────────────────
    full_text = _merge_pages(parsed)
    chunks    = plugin.chunk(parsed, chunk_size, overlap)

    # ── Embed + store ─────────────────────────────────────────────────────────
    store = EmbeddingStore(project_name)
    store.upsert_chunks(
        file_id=file_id,
        filename=source_path.name,
        chunks=chunks,
    )

    # ── Cache original file (encrypted) ───────────────────────────────────────
    cache = FileCacheManager(project_name)
    cache.store(source_path, file_id)

    result = IngestedFile(
        file_id=file_id,
        filename=source_path.name,
        source_path=source_path,
        chunk_count=len(chunks),
        page_count=len(parsed.pages),
        has_tables=any(p.tables for p in parsed.pages),
        was_ocrd=was_ocrd,
    )

    log.info("ingest.done", file=source_path.name, chunks=len(chunks), ocrd=was_ocrd)
    return result

def delete_file(file_id: str, project_name: str) -> None:
    """Remove file from vector store and file cache."""
    EmbeddingStore(project_name).delete_by_file_id(file_id)
    FileCacheManager(project_name).delete(file_id)
    log.info("ingest.delete.done", file_id=file_id)

def _generate_file_id(path: Path) -> str:
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem[:20]}_{content_hash}"

def _merge_pages(parsed) -> str:
    return "\n\n".join(page.text for page in parsed.pages)

def _run_ocr_pipeline(source_path: Path, parsed) -> object:
    from pdf2image import convert_from_path
    from plugins.base import ParsedPage, ParsedDocument

    pages = []
    if source_path.suffix.lower() == ".pdf":
        images = convert_from_path(str(source_path), dpi=300)
    else:
        images = [source_path]

    for i, img in enumerate(images):
        preprocessed = preprocess_image(img)
        raw_text     = extract_text_with_tesseract(preprocessed)
        clean_text   = correct_ocr_output(raw_text)
        pages.append(ParsedPage(page_number=i + 1, text=clean_text, tables=[], images=[]))

    return ParsedDocument(source_path=source_path, pages=pages, metadata={"ocr": True})
```

### `ingest/file_cache.py`

```python
from pathlib import Path
from security.encryption import encrypt_file, decrypt_file
from core.logger import get_logger

log = get_logger(__name__)

class FileCacheManager:
    """
    Manages encrypted storage of original source files per project.
    Files are stored as {file_id}.enc in data/projects/{project}/cache/.
    """

    def __init__(self, project_name: str) -> None:
        self.cache_dir = Path(f"data/projects/{project_name}/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def store(self, source_path: Path, file_id: str) -> Path:
        """Encrypt and store source file. Returns path to cached file."""
        dest = self.cache_dir / f"{file_id}.enc"
        encrypt_file(source_path, dest)
        log.info("cache.stored", file_id=file_id)
        return dest

    def retrieve(self, file_id: str, output_path: Path) -> Path:
        """Decrypt and write cached file to output_path."""
        src = self.cache_dir / f"{file_id}.enc"
        decrypt_file(src, output_path)
        return output_path

    def exists(self, file_id: str) -> bool:
        return (self.cache_dir / f"{file_id}.enc").exists()

    def delete(self, file_id: str) -> None:
        enc_path = self.cache_dir / f"{file_id}.enc"
        if enc_path.exists():
            enc_path.unlink()
```

---

## 8. OCR & Preprocessing Pipeline

### `ocr/detector.py`

```python
from plugins.base import ParsedDocument

MIN_TEXT_CHARS_PER_PAGE = 80

def needs_ocr(doc: ParsedDocument) -> bool:
    """
    Determine if a parsed document needs OCR.
    Returns True if average extracted text per page is below threshold,
    indicating the document is scanned or image-based.
    """
    if not doc.pages:
        return True
    avg = sum(len(p.text) for p in doc.pages) / len(doc.pages)
    return avg < MIN_TEXT_CHARS_PER_PAGE
```

### `ocr/preprocessor.py`

```python
import cv2
import numpy as np
from PIL.Image import Image
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)

def preprocess_image(image: Image) -> np.ndarray:
    """
    Apply quality corrections to an image before OCR.
    Pipeline: convert → grayscale → denoise → binarize → deskew.

    Args:
        image: PIL Image (from pdf2image or direct upload).

    Returns:
        Preprocessed numpy array ready for Tesseract.
    """
    cfg = get_config().ocr
    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if cfg.denoise:
        img = _denoise(img)
    if cfg.binarize:
        img = _binarize(img)
    if cfg.deskew:
        img = _deskew(img)

    log.debug("ocr.preprocess.done", shape=img.shape)
    return img

def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=10)

def _binarize(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def _deskew(img: np.ndarray) -> np.ndarray:
    """Detect skew angle and rotate to correct it."""
    coords = np.column_stack(np.where(img > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```

### `ocr/extractor.py`

```python
import numpy as np
import pytesseract
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)

def extract_text_with_tesseract(image: np.ndarray) -> str:
    """
    Run Tesseract OCR on a preprocessed image.

    Args:
        image: Preprocessed greyscale numpy array.

    Returns:
        Raw extracted text string.

    Raises:
        OcrError: If Tesseract is unavailable or fails.
    """
    cfg = get_config().ocr
    try:
        config_str = f"--dpi {cfg.dpi} --oem 3 --psm 6"
        text = pytesseract.image_to_string(image, config=config_str)
        log.debug("ocr.extract.done", chars=len(text))
        return text
    except pytesseract.TesseractNotFoundError as e:
        from core.errors import OcrError
        raise OcrError("Tesseract not found. Install via conda: conda install tesseract") from e
```

### `ocr/corrector.py`

```python
import re

def correct_ocr_output(raw_text: str) -> str:
    """
    Post-process raw OCR text to fix common extraction artifacts.
    Steps: strip noise chars → fix broken words → normalize whitespace →
           fix common OCR substitutions → remove headers/footers artifacts.

    Args:
        raw_text: Raw string from Tesseract.

    Returns:
        Cleaned text string.
    """
    text = raw_text
    text = _remove_noise_lines(text)
    text = _fix_common_substitutions(text)
    text = _normalize_whitespace(text)
    text = _rejoin_hyphenated_words(text)
    return text.strip()

def _remove_noise_lines(text: str) -> str:
    lines = [l for l in text.splitlines() if len(l.strip()) > 2 or l.strip() == ""]
    return "\n".join(lines)

def _fix_common_substitutions(text: str) -> str:
    subs = [("0", "O"), ("1", "l"), ("|", "I"), ("rn", "m")]
    # Only apply in clearly wrong contexts — leave alphanumeric intact
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)
    return text

def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def _rejoin_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
```

---

## 9. Template Engine

### `template/analyzer.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from core.logger import get_logger
from core.errors import TemplateAnalysisError

log = get_logger(__name__)

@dataclass
class FontStyle:
    name: str
    size_pt: float
    bold: bool
    italic: bool
    color_rgb: tuple[int, int, int] | None

@dataclass
class TemplateStyles:
    """All style information extracted from a template document."""
    page_width_emu: int
    page_height_emu: int
    margin_top_emu: int
    margin_bottom_emu: int
    margin_left_emu: int
    margin_right_emu: int
    heading1: FontStyle | None = None
    heading2: FontStyle | None = None
    heading3: FontStyle | None = None
    body: FontStyle | None     = None
    header_text: str           = ""
    footer_text: str           = ""
    has_letterhead: bool       = False
    letterhead_placeholder: str = "[LETTERHEAD]"
    table_header_bg: str       = "FFFFFF"
    primary_color_hex: str     = "000000"
    has_page_numbers: bool     = False

def analyze_template(template_path: Path) -> TemplateStyles:
    """
    Infer style, layout, font, header, footer, and letterhead
    from an uploaded template document.

    Args:
        template_path: Path to .docx template file.

    Returns:
        TemplateStyles describing the full template layout.

    Raises:
        TemplateAnalysisError: If the file cannot be parsed as a valid template.
    """
    log.info("template.analyze.start", file=template_path.name)

    try:
        doc = Document(str(template_path))
    except Exception as e:
        raise TemplateAnalysisError(f"Cannot open template: {e}") from e

    styles = TemplateStyles(
        page_width_emu  = doc.sections[0].page_width,
        page_height_emu = doc.sections[0].page_height,
        margin_top_emu    = doc.sections[0].top_margin,
        margin_bottom_emu = doc.sections[0].bottom_margin,
        margin_left_emu   = doc.sections[0].left_margin,
        margin_right_emu  = doc.sections[0].right_margin,
    )

    styles.heading1 = _extract_named_style(doc, "Heading 1")
    styles.heading2 = _extract_named_style(doc, "Heading 2")
    styles.heading3 = _extract_named_style(doc, "Heading 3")
    styles.body     = _extract_named_style(doc, "Normal")

    styles.header_text, styles.has_letterhead = _extract_header(doc)
    styles.footer_text, styles.has_page_numbers = _extract_footer(doc)
    styles.primary_color_hex = _detect_primary_color(doc)
    styles.table_header_bg   = _detect_table_header_color(doc)

    log.info("template.analyze.done",
             h1=styles.heading1.name if styles.heading1 else None,
             has_letterhead=styles.has_letterhead)
    return styles

def _extract_named_style(doc: Document, style_name: str) -> FontStyle | None:
    try:
        s = doc.styles[style_name]
        font = s.font
        rgb = None
        if font.color and font.color.rgb:
            c = font.color.rgb
            rgb = (c.red, c.green, c.blue)
        return FontStyle(
            name     = font.name or "Calibri",
            size_pt  = font.size.pt if font.size else 11.0,
            bold     = bool(font.bold),
            italic   = bool(font.italic),
            color_rgb = rgb,
        )
    except Exception:
        return None

def _extract_header(doc: Document) -> tuple[str, bool]:
    for section in doc.sections:
        if section.header:
            text = " ".join(p.text for p in section.header.paragraphs).strip()
            has_letterhead = bool(text) or _header_has_image(section)
            return text or "[LETTERHEAD]", has_letterhead
    return "", False

def _extract_footer(doc: Document) -> tuple[str, bool]:
    for section in doc.sections:
        if section.footer:
            text = " ".join(p.text for p in section.footer.paragraphs).strip()
            has_pages = "PAGE" in text.upper() or _footer_has_page_field(section)
            return text, has_pages
    return "", False

def _header_has_image(section) -> bool:
    for p in section.header.paragraphs:
        for run in p.runs:
            if run._element.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline"):
                return True
    return False

def _footer_has_page_field(section) -> bool:
    xml = section.footer._element.xml
    return "PAGE" in xml or "fldChar" in xml

def _detect_primary_color(doc: Document) -> str:
    for style_name in ["Heading 1", "Heading 2"]:
        try:
            color = doc.styles[style_name].font.color.rgb
            if color:
                return str(color)
        except Exception:
            pass
    return "2E3B4E"

def _detect_table_header_color(doc: Document) -> str:
    for table in doc.tables:
        if table.rows:
            for cell in table.rows[0].cells:
                tc = cell._tc
                shd = tc.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd")
                if shd is not None:
                    fill = shd.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill")
                    if fill and fill != "auto":
                        return fill
    return "D5E8F0"
```

### `template/store.py`

```python
from pathlib import Path
import shutil
from template.analyzer import TemplateStyles, analyze_template
from core.logger import get_logger
import json

log = get_logger(__name__)

class TemplateStore:
    """Manages template files and their analyzed styles per project."""

    def __init__(self, project_name: str) -> None:
        self.template_dir = Path(f"data/projects/{project_name}/templates")
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def save_template(self, source_path: Path) -> TemplateStyles:
        """Copy template file and run analysis. Returns extracted styles."""
        dest = self.template_dir / source_path.name
        shutil.copy2(source_path, dest)
        styles = analyze_template(dest)
        self._persist_styles(source_path.stem, styles)
        log.info("template.saved", name=source_path.name)
        return styles

    def load_styles(self, template_name: str) -> TemplateStyles | None:
        """Load pre-analyzed TemplateStyles from disk."""
        styles_path = self.template_dir / f"{template_name}.styles.json"
        if not styles_path.exists():
            return None
        with open(styles_path) as f:
            data = json.load(f)
        return TemplateStyles(**data)

    def list_templates(self) -> list[str]:
        return [f.stem for f in self.template_dir.glob("*.docx")]

    def _persist_styles(self, name: str, styles: TemplateStyles) -> None:
        import dataclasses
        path = self.template_dir / f"{name}.styles.json"
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(styles), f, indent=2)
```

---

## 10. Embedding & Vector Storage

### `embeddings/client.py`

```python
from __future__ import annotations
import asyncio
import httpx
from core.config import get_config
from core.errors import EmbeddingError
from core.logger import get_logger

log = get_logger(__name__)

async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """
    Batch embed a list of text strings via Ollama nomic-embed-text.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (one per input text).

    Raises:
        EmbeddingError: If the Ollama embedding endpoint fails.
    """
    cfg = get_config()
    model = cfg.models.embedding_model
    url   = f"{cfg.ollama.base_url}/api/embeddings"

    log.info("embeddings.batch.start", model=model, count=len(texts))
    vectors = []

    async with httpx.AsyncClient(timeout=cfg.ollama.timeout_sec) as client:
        for text in texts:
            try:
                resp = await client.post(url, json={"model": model, "prompt": text})
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
            except httpx.HTTPError as e:
                log.error("embeddings.request.failed", error=str(e))
                raise EmbeddingError(f"Embedding failed: {e}") from e

    log.info("embeddings.batch.done", count=len(vectors))
    return vectors

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Synchronous wrapper for embed_texts_async."""
    return asyncio.run(embed_texts_async(texts))
```

### `embeddings/store.py`

```python
from __future__ import annotations
from pathlib import Path
import lancedb
import pyarrow as pa
from embeddings.client import embed_texts
from core.logger import get_logger
from core.errors import StoreError

log = get_logger(__name__)

SCHEMA = pa.schema([
    pa.field("chunk_id",  pa.string()),
    pa.field("file_id",   pa.string()),
    pa.field("filename",  pa.string()),
    pa.field("text",      pa.string()),
    pa.field("chunk_idx", pa.int32()),
    pa.field("vector",    pa.list_(pa.float32(), 768)),
])

class EmbeddingStore:
    """LanceDB-backed vector store scoped to a project namespace."""

    TABLE_NAME = "chunks"

    def __init__(self, project_name: str) -> None:
        db_path = Path(f"data/projects/{project_name}/vectors")
        db_path.mkdir(parents=True, exist_ok=True)
        self._db    = lancedb.connect(str(db_path))
        self._table = self._get_or_create_table()

    def upsert_chunks(
        self,
        file_id: str,
        filename: str,
        chunks: list[str],
    ) -> None:
        """
        Embed and upsert all chunks for a file.
        Deletes existing entries for file_id before inserting fresh ones.

        Args:
            file_id:  Unique identifier for the source file.
            filename: Display name of the source file.
            chunks:   Ordered list of text chunks.

        Raises:
            StoreError: If embedding or LanceDB write fails.
        """
        log.info("store.upsert.start", file_id=file_id, chunks=len(chunks))
        self.delete_by_file_id(file_id)

        vectors = embed_texts(chunks)
        rows = [
            {
                "chunk_id":  f"{file_id}__{i}",
                "file_id":   file_id,
                "filename":  filename,
                "text":      chunk,
                "chunk_idx": i,
                "vector":    vec,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self._table.add(rows)
        log.info("store.upsert.done", file_id=file_id, rows=len(rows))

    def search(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.68,
    ) -> list[dict]:
        """
        Semantic search over stored chunks.

        Returns:
            List of chunk dicts ordered by relevance, filtered by min_score.
        """
        log.info("store.search.start", query=query[:60], top_k=top_k)
        vec = embed_texts([query])[0]
        results = (
            self._table.search(vec)
            .limit(top_k * 2)
            .to_list()
        )
        filtered = [r for r in results if r.get("_distance", 1.0) <= (1 - min_score)]
        log.info("store.search.done", returned=len(filtered))
        return filtered[:top_k]

    def delete_by_file_id(self, file_id: str) -> None:
        """Remove all chunks belonging to a file."""
        self._table.delete(f"file_id = '{file_id}'")
        log.info("store.delete.done", file_id=file_id)

    def list_files(self) -> list[dict]:
        """Return summary of all ingested files: filename, file_id, chunk count."""
        df = self._table.to_pandas()[["file_id", "filename", "chunk_idx"]]
        grouped = df.groupby(["file_id", "filename"]).count().reset_index()
        grouped = grouped.rename(columns={"chunk_idx": "chunk_count"})
        return grouped.to_dict("records")

    def _get_or_create_table(self):
        if self.TABLE_NAME in self._db.table_names():
            return self._db.open_table(self.TABLE_NAME)
        return self._db.create_table(self.TABLE_NAME, schema=SCHEMA)
```

---

## 11. Agent System — Instruction-to-Document Mode

All agents share a **context budget**: each agent receives only the chunks it needs,
never the full document base at once.

```
User instruction
      │
      ▼
 [PlannerAgent]  → Produces a section plan [{title, instructions}]
      │
      ▼ (for each section)
 [RetrieverAgent] → Fetches relevant chunks for THIS section only
      │
      ▼
 [WriterAgent]  → Streams section content from chunks + instruction
      │
      ▼
 [ReviewerAgent] → Refines section for quality, length, and accuracy
      │
      ▼
 Assembled document → Output pipeline
```

### `agents/base.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from core.config import get_config
from core.logger import get_logger
from llm.client import stream_completion

log = get_logger(__name__)

@dataclass
class AgentContext:
    """Scoped context passed between agents. Never grows unboundedly."""
    project_name: str
    mode: str               # "itd" or "ctd"
    instruction: str
    template_name: str | None
    budget_tokens: int

@dataclass
class GeneratedSection:
    title: str
    content: str
    sources: list[str]      # chunk_ids used

class BaseAgent:
    """Common functionality for all agents."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.log   = get_logger(self.__class__.__name__)

    def _call_llm(self, prompt: str, system: str) -> str:
        """Blocking LLM call. Returns full response string."""
        result = ""
        for token in stream_completion(prompt=prompt, system=system, model=self.model):
            result += token
        return result

    def _stream_llm(self, prompt: str, system: str):
        """Generator yielding tokens for streaming UI display."""
        yield from stream_completion(prompt=prompt, system=system, model=self.model)
```

### `agents/itd/planner.py`

```python
import json
from agents.base import BaseAgent, AgentContext
from core.config import get_config
from core.errors import AgentError

PLANNER_SYSTEM = """You are a senior technical writer and document architect.
Given a user instruction, produce a structured document plan.
Respond ONLY with valid JSON: {"sections": [{"title": "...", "instructions": "..."}]}
No preamble. No markdown. JSON only."""

class PlannerAgent(BaseAgent):
    def plan_document(self, ctx: AgentContext) -> list[dict]:
        """
        Convert user instruction into an ordered list of document sections.

        Args:
            ctx: Agent context with instruction and mode.

        Returns:
            List of dicts: [{title: str, instructions: str}]

        Raises:
            AgentError: If LLM returns unparseable JSON.
        """
        self.log.info("planner.start", instruction=ctx.instruction[:80])

        prompt = f"""Document type and instruction:
{ctx.instruction}

Generate a complete section plan for this document.
Use standard engineering document structure (Executive Summary, Scope,
Requirements, Design, Implementation, Testing, Appendix as applicable).
Fill any gaps with professional standard content for this document type."""

        raw = self._call_llm(prompt=prompt, system=PLANNER_SYSTEM)

        try:
            plan = json.loads(raw)
            sections = plan.get("sections", [])
            self.log.info("planner.done", sections=len(sections))
            return sections
        except json.JSONDecodeError as e:
            raise AgentError(f"Planner returned invalid JSON: {e}\nRaw: {raw[:200]}") from e
```

### `agents/itd/writer.py`

```python
from agents.base import BaseAgent, AgentContext, GeneratedSection
from core.config import get_config

WRITER_SYSTEM = """You are an expert technical writer producing professional engineering documentation.
Write clearly, precisely, and at a senior engineering level.
Use the provided context chunks as your primary source.
Fill knowledge gaps with accurate general technical knowledge.
Do not invent specifications — mark uncertainties with [TBD]."""

class WriterAgent(BaseAgent):
    def write_section(
        self,
        ctx: AgentContext,
        section: dict,
        chunks: list[dict],
    ):
        """
        Stream-generate content for a single document section.

        Args:
            ctx:     Agent context (budget, instruction, mode).
            section: {title, instructions} from PlannerAgent output.
            chunks:  Retrieved context chunks for this section.

        Yields:
            Text tokens for real-time streaming display.

        Returns (after exhausting generator):
            GeneratedSection accessible via `.result` attribute.
        """
        self.log.info("writer.section.start", title=section["title"])

        context_text = "\n\n---\n\n".join(
            f"[Source: {c['filename']}]\n{c['text']}" for c in chunks
        )

        prompt = f"""## Document Section: {section['title']}
Section instructions: {section['instructions']}

## Source Context:
{context_text}

## Your task:
Write a complete, professional section titled "{section['title']}".
Use the source context above. Fill gaps with general knowledge.
Output only the section content, no title, no preamble."""

        content_parts = []
        for token in self._stream_llm(prompt=prompt, system=WRITER_SYSTEM):
            content_parts.append(token)
            yield token

        self.result = GeneratedSection(
            title   = section["title"],
            content = "".join(content_parts),
            sources = [c["chunk_id"] for c in chunks],
        )
        self.log.info("writer.section.done", title=section["title"],
                      chars=len(self.result.content))
```

---

## 12. Agent System — Code-to-Document Mode

```
Code files
    │
    ▼
[ParserAgent]   → tree-sitter AST → extracts functions, classes, docstrings, signatures
    │
    ▼
[AnalyzerAgent] → understands intent, relationships, design patterns
    │
    ▼
[WriterAgent]   → generates doc section per module/class/function
    │
    ▼
[ReviewerAgent] → validates technical accuracy against original AST
```

### `agents/ctd/parser.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tree_sitter import Language, Parser
from core.errors import CodeParseError
from core.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".py": "python", ".c": "c", ".cpp": "c", ".h": "c", ".java": "java"}

@dataclass
class ParsedFunction:
    name: str
    signature: str
    docstring: str
    body_summary: str
    line_start: int
    line_end: int

@dataclass
class ParsedClass:
    name: str
    docstring: str
    methods: list[ParsedFunction]
    line_start: int

@dataclass
class ParsedCodeFile:
    path: Path
    language: str
    module_docstring: str
    imports: list[str]
    classes: list[ParsedClass]
    functions: list[ParsedFunction]
    constants: list[str]

class ParserAgent:
    """Parses source code files into structured representations via tree-sitter."""

    def parse_file(self, path: Path) -> ParsedCodeFile:
        """
        Parse a source code file into a structured code model.

        Args:
            path: Absolute path to source file (.py, .c, .cpp, .h, .java).

        Returns:
            ParsedCodeFile with classes, functions, docstrings.

        Raises:
            CodeParseError: If language is unsupported or file cannot be parsed.
        """
        ext = path.suffix.lower()
        lang = SUPPORTED_EXTENSIONS.get(ext)
        if not lang:
            raise CodeParseError(f"Unsupported code extension: {ext}")

        log.info("ctd.parser.start", file=path.name, lang=lang)
        source = path.read_text(encoding="utf-8", errors="replace")

        parser = _get_parser(lang)
        tree   = parser.parse(source.encode())

        result = _extract_structure(tree, source, lang, path)
        log.info("ctd.parser.done", file=path.name,
                 classes=len(result.classes), functions=len(result.functions))
        return result

def _get_parser(lang: str) -> Parser:
    import tree_sitter_python
    import tree_sitter_c
    import tree_sitter_java
    lang_map = {
        "python": tree_sitter_python.language(),
        "c":      tree_sitter_c.language(),
        "java":   tree_sitter_java.language(),
    }
    return Parser(Language(lang_map[lang]))

def _extract_structure(tree, source: str, lang: str, path: Path) -> ParsedCodeFile:
    # Dispatcher — each language has different AST node names
    if lang == "python":
        return _extract_python(tree, source, path)
    elif lang == "c":
        return _extract_c(tree, source, path)
    elif lang == "java":
        return _extract_java(tree, source, path)

def _extract_python(tree, source: str, path: Path) -> ParsedCodeFile:
    """Walk Python AST for module docstring, classes, and functions."""
    lines = source.splitlines()
    functions, classes, imports, constants = [], [], [], []
    module_doc = ""

    def visit(node, depth=0):
        nonlocal module_doc
        if node.type == "module":
            first = node.children[0] if node.children else None
            if first and first.type == "expression_statement":
                s = first.children[0] if first.children else None
                if s and s.type == "string":
                    module_doc = source[s.start_byte:s.end_byte].strip('"\' \n')

        if node.type == "function_definition":
            functions.append(_extract_py_function(node, source, lines))
        if node.type == "class_definition":
            classes.append(_extract_py_class(node, source, lines))
        if node.type == "import_statement" or node.type == "import_from_statement":
            imports.append(source[node.start_byte:node.end_byte].strip())

        for child in node.children:
            visit(child, depth + 1)

    visit(tree.root_node)
    return ParsedCodeFile(path=path, language="python", module_docstring=module_doc,
                          imports=imports, classes=classes, functions=functions, constants=constants)

def _extract_py_function(node, source: str, lines: list) -> ParsedFunction:
    name = ""
    doc  = ""
    for child in node.children:
        if child.type == "identifier":
            name = source[child.start_byte:child.end_byte]
        if child.type == "block":
            first = child.children[0] if child.children else None
            if first and first.type == "expression_statement":
                s = first.children[0] if first.children else None
                if s and s.type == "string":
                    doc = source[s.start_byte:s.end_byte].strip('"\' \n')
    sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
    return ParsedFunction(name=name, signature=sig_line.strip(), docstring=doc,
                          body_summary="", line_start=node.start_point[0],
                          line_end=node.end_point[0])

def _extract_py_class(node, source: str, lines: list) -> ParsedClass:
    name = ""
    doc  = ""
    methods = []
    for child in node.children:
        if child.type == "identifier":
            name = source[child.start_byte:child.end_byte]
        if child.type == "block":
            for item in child.children:
                if item.type == "function_definition":
                    methods.append(_extract_py_function(item, source, lines))
    return ParsedClass(name=name, docstring=doc, methods=methods,
                       line_start=node.start_point[0])

def _extract_c(tree, source, path): ...    # Mirror of _extract_python for C grammar
def _extract_java(tree, source, path): ... # Mirror of _extract_python for Java grammar
```

---

## 13. Output Generation

### `output/pipeline.py`

```python
from __future__ import annotations
from pathlib import Path
from agents.base import GeneratedSection
from template.analyzer import TemplateStyles
from plugins.registry import PluginRegistry
from core.logger import get_logger
from core.errors import OutputError

log = get_logger(__name__)

def generate_output(
    sections: list[GeneratedSection],
    formats: list[str],
    template_styles: TemplateStyles | None,
    project_name: str,
    doc_title: str,
) -> dict[str, Path]:
    """
    Route generated sections to all requested output format plugins.

    Args:
        sections:        Ordered list of agent-generated sections.
        formats:         Output format identifiers. e.g. ["docx", "pdf", "md"]
        template_styles: Extracted template styles or None.
        project_name:    Project namespace for output directory.
        doc_title:       Base filename for output files (sanitized).

    Returns:
        Dict mapping format → output file path.

    Raises:
        OutputError: If no plugin handles a requested format.
    """
    out_dir = Path(f"data/projects/{project_name}/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for fmt in formats:
        plugin = PluginRegistry.get_output_plugin_for(fmt)
        if plugin is None:
            raise OutputError(f"No output plugin registered for format: {fmt}")

        safe_title   = "".join(c for c in doc_title if c.isalnum() or c in "_ -")
        output_path  = out_dir / f"{safe_title}.{fmt}"

        log.info("output.write.start", format=fmt, path=str(output_path))
        written = plugin.write(sections, template_styles, output_path)
        results[fmt] = written
        log.info("output.write.done", format=fmt, path=str(written))

    return results
```

---

## 14. Encryption

### `security/encryption.py`

```python
from __future__ import annotations
from pathlib import Path
from cryptography.fernet import Fernet
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)

def _load_or_create_key() -> bytes:
    cfg      = get_config()
    key_path = Path(cfg.encryption.key_file)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        return key_path.read_bytes()
    else:
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        key_path.chmod(0o600)    # Owner read-only
        log.info("encryption.key.created", path=str(key_path))
        return key

def _fernet() -> Fernet:
    cfg = get_config()
    if not cfg.encryption.enabled:
        return None
    return Fernet(_load_or_create_key())

def encrypt_file(source: Path, dest: Path) -> None:
    """
    Encrypt source file and write ciphertext to dest.
    If encryption is disabled in config, copies the file unchanged.
    """
    f = _fernet()
    data = source.read_bytes()
    dest.write_bytes(f.encrypt(data) if f else data)

def decrypt_file(source: Path, dest: Path) -> None:
    """
    Decrypt source file and write plaintext to dest.
    If encryption is disabled in config, copies the file unchanged.
    """
    f = _fernet()
    data = source.read_bytes()
    dest.write_bytes(f.decrypt(data) if f else data)

def encrypt_text(text: str) -> str:
    """Encrypt a string and return base64-encoded ciphertext."""
    f = _fernet()
    return f.encrypt(text.encode()).decode() if f else text

def decrypt_text(ciphertext: str) -> str:
    """Decrypt a base64 ciphertext string."""
    f = _fernet()
    return f.decrypt(ciphertext.encode()).decode() if f else ciphertext
```

---

## 15. Multi-Tenancy (Project Namespacing)

Projects are pure directory namespacing — no auth, no tokens.
Every store, cache, template, and output is isolated under `data/projects/{name}/`.

### `projects/manager.py`

```python
from pathlib import Path
from core.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path("data/projects")
SUBDIRS      = ["cache", "vectors", "templates", "outputs"]

def create_project(name: str) -> Path:
    """Create directory structure for a new project namespace."""
    _validate_name(name)
    project_path = PROJECT_ROOT / name
    for sub in SUBDIRS:
        (project_path / sub).mkdir(parents=True, exist_ok=True)
    log.info("project.created", name=name)
    return project_path

def list_projects() -> list[str]:
    """Return all existing project names."""
    if not PROJECT_ROOT.exists():
        return []
    return [p.name for p in PROJECT_ROOT.iterdir() if p.is_dir()]

def delete_project(name: str) -> None:
    """Delete all project data. Irreversible."""
    import shutil
    path = PROJECT_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    log.info("project.deleted", name=name)

def project_exists(name: str) -> bool:
    return (PROJECT_ROOT / name).is_dir()

def _validate_name(name: str) -> None:
    if not name.isidentifier() and not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError(f"Invalid project name: '{name}'. Use alphanumeric, hyphens, underscores.")
```

---

## 16. Streamlit UI Architecture

### Tab Structure

```
┌────────────────────────────────────────────────────────────────┐
│  DocForge  [Project: my_project ▾]                             │
├────────────┬───────────┬───────────┬─────────┬────────────────-┤
│  📁 Upload │ 📄 Files  │ ⚡Generate│ 📋 Logs │ ⚙️ Settings     │
└────────────┴───────────┴───────────┴─────────┴─────────────────┘
```

### Upload Tab (`ui/pages/upload.py`)
- `st.file_uploader()` for source documents (any extension)
- Separate `st.file_uploader()` for templates (`.docx` only)
- Progress bar per file during ingestion
- Success/error feedback per file

### Files Tab (`ui/pages/files.py`)
- `st_aggrid` table with columns: `filename`, `file_id`, `chunk_count`, `page_count`, `has_tables`, `was_ocrd`, `ingested_at`
- Row selection enables delete or preview
- Preview: decrypt from cache → render first page as image via `st.image()`

### Generate Tab (`ui/pages/generate.py`)

```python
import streamlit as st
from agents.itd.planner import PlannerAgent
from agents.itd.retriever import RetrieverAgent
from agents.itd.writer import WriterAgent
from agents.itd.reviewer import ReviewerAgent
from agents.base import AgentContext
from output.pipeline import generate_output
from core.config import get_config

def render_generate_tab(project_name: str) -> None:
    cfg   = get_config()
    mode  = st.radio("Mode", ["Instruction → Document", "Code → Document"], horizontal=True)
    instr = st.text_area("Instruction / Document Type", height=120)
    tmpl  = st.selectbox("Template", ["None"] + get_templates(project_name))
    fmts  = st.multiselect("Output Formats", ["docx", "pdf", "md", "excel"], default=["docx", "pdf"])

    if st.button("Generate", type="primary"):
        _run_generation(mode, instr, tmpl, fmts, project_name, cfg)

def _run_generation(mode, instruction, template, formats, project_name, cfg):
    ctx = AgentContext(
        project_name  = project_name,
        mode          = "itd" if "Instruction" in mode else "ctd",
        instruction   = instruction,
        template_name = template if template != "None" else None,
        budget_tokens = cfg.agents.itd_context_budget_tokens,
    )

    output_container = st.container()

    with st.status("🧠 Planning document structure...", expanded=True) as status:
        planner  = PlannerAgent(model=cfg.models.itd_model)
        plan     = planner.plan_document(ctx)
        st.write(f"✅ Plan ready — {len(plan)} sections")

    sections = []
    retriever = RetrieverAgent(project_name=project_name)
    writer    = WriterAgent(model=cfg.models.itd_model)

    for i, section in enumerate(plan):
        with st.status(f"✍️ Writing: {section['title']} ({i+1}/{len(plan)})", expanded=True):
            chunks = retriever.fetch_chunks_for_section(section, ctx)
            st.write(f"  Retrieved {len(chunks)} context chunks")

            # Stream content into the UI
            stream_gen = writer.write_section(ctx, section, chunks)
            with output_container:
                st.write_stream(stream_gen)

            sections.append(writer.result)

    with st.status("🔍 Reviewing document...", expanded=False):
        reviewer = ReviewerAgent(model=cfg.models.itd_model)
        sections = reviewer.review_all(sections, ctx)
        st.write("✅ Review complete")

    with st.status("📄 Writing output files...", expanded=False):
        from template.store import TemplateStore
        styles = TemplateStore(project_name).load_styles(ctx.template_name) if ctx.template_name else None
        outputs = generate_output(sections, formats, styles, project_name, instruction[:40])
        for fmt, path in outputs.items():
            st.success(f"{fmt.upper()} → {path}")
            with open(path, "rb") as f:
                st.download_button(f"Download {fmt.upper()}", f, file_name=path.name)
```

### Settings Tab (`ui/pages/settings.py`)
Sections:
- **Models**: dropdowns to select ITD model, CTD model, embedding model (lists from `ollama list`)
- **Logging switches**: per-module checkboxes that write to `user.toml` on save
- **OCR settings**: dpi, preprocessing toggles
- **Encryption**: toggle on/off
- **Plugins**: drag-and-drop ZIP uploader → calls `install_plugin_from_zip()`
- **Agents**: context budget sliders, reviewer toggle, retrieval settings

---

## 17. Launcher Architecture

The launcher is the first thing a developer runs. It has two layers:

```
launcher.py
    │
    ├── CLI layer (Typer)     → parse flags, route to action
    └── TUI layer (Textual)   → interactive dashboard (--interactive flag)
```

### CLI Flags (`launcher.py`)

```python
import typer
from rich.console import Console
from typing import Annotated

app     = typer.Typer(name="docforge", help="DocForge Launcher")
console = Console()

@app.command()
def main(
    install:     Annotated[bool, typer.Option("--install",     "-i", help="Install missing dependencies and pull Ollama models")] = False,
    check:       Annotated[bool, typer.Option("--check",       "-c", help="Check all services, print status, and exit")] = False,
    debug:       Annotated[bool, typer.Option("--debug",       "-d", help="Enable DEBUG logging globally")] = False,
    test:        Annotated[bool, typer.Option("--test",        "-t", help="Run system health tests and exit")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-I", help="Launch the full Textual TUI")] = False,
    verbose:     Annotated[bool, typer.Option("--verbose",     "-v", help="Verbose output for all service calls")] = False,
    no_ui:       Annotated[bool, typer.Option("--no-ui",             help="Start backend services only (headless)")] = False,
    reset:       Annotated[bool, typer.Option("--reset",             help="Reset user.toml to defaults")] = False,
    project:     Annotated[str,  typer.Option("--project",     "-p", help="Launch directly into a named project")] = "",
    config:      Annotated[str,  typer.Option("--config",            help="Path to alternate config TOML")] = "",
) -> None:

    if reset:      _do_reset();       raise typer.Exit()
    if install:    _do_install(verbose)
    if check:      _do_check();       raise typer.Exit()
    if test:       _do_test();        raise typer.Exit()
    if interactive:
        _launch_tui(debug=debug)
    elif not no_ui:
        _launch_streamlit(project=project, debug=debug)

if __name__ == "__main__":
    app()
```

### Flag Behavior Reference

| Flag | Short | Behavior |
|------|-------|----------|
| `--install` | `-i` | Checks conda env, installs pip deps, pulls Ollama models, exits |
| `--check` | `-c` | Pings all services, prints color-coded status table, exits |
| `--debug` | `-d` | Sets log level to DEBUG globally before any other action |
| `--test` | `-t` | Runs pytest health suite (ingest, embed, generate small doc), exits |
| `--interactive` | `-I` | Launches Textual TUI instead of Streamlit |
| `--verbose` | `-v` | Prints raw output from every subprocess (conda, Ollama) |
| `--no-ui` | | Starts Ollama only, no Streamlit (for scripting/automation) |
| `--reset` | | Deletes `config/user.toml`, restores defaults |
| `--project` | `-p` | Pre-selects project namespace when Streamlit launches |
| `--config` | | Load alternate TOML config file path |
| `--help` | `-h` | Print full flag reference and exit |

### Service Orchestration

```python
# tui/app.py — service check sequence

SERVICES = [
    {"name": "Ollama",       "check": check_ollama,       "start": start_ollama},
    {"name": "LanceDB dir",  "check": check_lancedb_dir,  "start": None},
    {"name": "Plugin dir",   "check": check_plugin_dir,   "start": create_plugin_dir},
    {"name": "Config valid", "check": check_config_valid, "start": None},
    {"name": "Log dir",      "check": check_log_dir,      "start": create_log_dir},
]

def check_all_services(retry_max: int, retry_wait: int) -> dict[str, bool]:
    """
    Check every required service. Attempt to start failed ones up to retry_max times.

    Returns:
        Dict of service_name → is_healthy bool.
    """
    results = {}
    for svc in SERVICES:
        ok = _check_with_retry(svc, retry_max, retry_wait)
        results[svc["name"]] = ok
    return results

def _check_with_retry(svc: dict, max_retries: int, wait_sec: int) -> bool:
    import time
    for attempt in range(1, max_retries + 1):
        if svc["check"]():
            return True
        if svc["start"] and attempt < max_retries:
            svc["start"]()
            time.sleep(wait_sec)
    return False
```

### Textual TUI (`tui/app.py`)

```
┌──────────────────────────────────────────────────────────┐
│  DocForge Launcher                              v1.0.0   │
├──────────────────────┬───────────────────────────────────┤
│  SERVICES            │  LOGS                             │
│                      │                                   │
│  ✅ Ollama           │  [12:01:03] INFO  ollama.online   │
│  ✅ LanceDB dir      │  [12:01:03] INFO  lancedb.ready   │
│  ✅ Plugin dir       │  [12:01:04] INFO  plugins.loaded  │
│  ✅ Config           │  [12:01:04] INFO  config.loaded   │
│  ❌ Log dir          │  [12:01:04] ERROR log.dir.missing │
│                      │                                   │
├──────────────────────┴───────────────────────────────────┤
│  LOGGING MODULES                                         │
│                                                          │
│  [✓] ingest    [✓] ocr      [✓] embeddings              │
│  [✓] agents    [✓] llm      [ ] output                  │
│  [✓] template  [✓] plugins                              │
│                                                          │
│  ↑↓ navigate   SPACE toggle   ENTER confirm              │
├──────────────────────────────────────────────────────────┤
│  [Launch UI]  [Run Check]  [Install Deps]  [Quit]        │
└──────────────────────────────────────────────────────────┘
```

Textual widgets used:
- `Checkbox` — logging module toggles (navigable with arrow keys + Space)
- `Button` — action buttons in footer
- `RichLog` — scrolling log panel (Rich-formatted, color-coded)
- `DataTable` — service status table
- `Label` — status indicators
- `Input` — project name text field

---

## 18. Logging Architecture

### `core/logger.py`

```python
from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
import structlog
from rich.logging import RichHandler
from core.config import get_config

_MODULE_LOGGERS: dict[str, bool] = {}

def setup_logging(debug: bool = False) -> None:
    """
    Initialise structlog + Rich terminal handler + rotating file handler.
    Call once at application startup (in launcher.py and app/main.py).
    """
    cfg      = get_config()
    log_dir  = Path(cfg.app.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    level    = logging.DEBUG if debug else getattr(logging, cfg.app.log_level)

    # ── File handler (rotating, 5 MB × 5 files) ───────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename    = log_dir / "docforge.log",
        maxBytes    = 5 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    ))

    # ── Rich terminal handler (color-coded) ───────────────────────────────────
    rich_handler = RichHandler(
        rich_tracebacks = True,
        markup          = True,
        show_path       = False,
    )

    logging.basicConfig(
        level    = level,
        handlers = [rich_handler, file_handler],
        force    = True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class   = structlog.stdlib.BoundLogger,
        context_class   = dict,
        logger_factory  = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True,
    )

    # Load module switches from config
    cfg_mods = cfg.log_modules
    for module_name in ["ingest", "ocr", "embeddings", "agents", "llm", "output", "template", "plugins"]:
        _MODULE_LOGGERS[module_name] = getattr(cfg_mods, module_name, True)

def get_logger(name: str):
    """
    Return a structlog logger bound to `name`.
    If the module's logging switch is OFF, returns a no-op logger.
    """
    module_key = _resolve_module_key(name)
    if not _MODULE_LOGGERS.get(module_key, True):
        return _NoOpLogger()
    return structlog.get_logger(name)

def set_module_logging(module: str, enabled: bool) -> None:
    """Toggle logging for a module at runtime. Persists to user.toml."""
    _MODULE_LOGGERS[module] = enabled
    from core.config import save_user_config
    save_user_config({"logging": {"modules": {module: enabled}}})

def _resolve_module_key(name: str) -> str:
    """Map a dotted module path to its top-level module key."""
    parts = name.split(".")
    return parts[0] if parts[0] in _MODULE_LOGGERS else name.split("/")[-1].split("_")[0]

class _NoOpLogger:
    """Absorbs all log calls when a module's logging is switched off."""
    def __getattr__(self, _): return lambda *a, **kw: None
```

### Log Levels & Colors (Rich Terminal)

| Level | Color | When |
|-------|-------|------|
| `DEBUG` | dim white | Internal state, token counts, chunk sizes |
| `INFO` | green | Normal operations: start, done, loaded |
| `WARNING` | yellow | Recoverable: low score, fallback used, OCR override |
| `ERROR` | red | Operation failed, file rejected, LLM error |
| `CRITICAL` | bold red | System halted, pipeline dead |

### Log File Location

```
data/logs/
├── docforge.log        # Current log (5 MB max)
├── docforge.log.1      # Rotated backup 1
├── docforge.log.2      # Rotated backup 2
...
└── docforge.log.5      # Oldest retained
```

---

## 19. Developer Setup (Conda + Windows)

### `environment.yml`

```yaml
name: docforge-dev
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - tesseract          # Managed by conda on Windows (no manual PATH fiddling)
  - poppler            # For pdf2image
  - nodejs>=18         # If any JS tooling is needed
  - pip:
    - -r requirements.txt
    - -r requirements-dev.txt
```

### Bootstrap Sequence (Windows)

```batch
REM 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate docforge-dev

REM 2. Install Ollama (manual, from https://ollama.com)
REM    Then start it:
ollama serve

REM 3. Pull models
ollama pull gemma3:7b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

REM 4. Install tree-sitter language grammars
python scripts/install_grammars.py

REM 5. Launch via the launcher
python launcher.py --install --check
python launcher.py
```

### `scripts/install_grammars.py`

```python
"""Install tree-sitter language grammars for Python, C, and Java."""
import subprocess, sys

GRAMMARS = [
    "tree-sitter-python",
    "tree-sitter-c",
    "tree-sitter-java",
]

for grammar in GRAMMARS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", grammar])
    print(f"✅ Installed {grammar}")
```

---

## 20. Requirements Management

### `requirements.txt` (Production)

```
# Runtime
streamlit>=1.35.0
streamlit-aggrid>=0.3.4
streamlit-extras>=0.4.0
watchdog>=4.0.0

# LLM + Embeddings
ollama>=0.2.0
httpx>=0.27.0
sentence-transformers>=2.7.0

# Vector DB
lancedb>=0.6.0
pyarrow>=15.0.0

# Document Ingestion
pymupdf>=1.24.0
pdfplumber>=0.11.0
pdf2image>=1.17.0
python-docx>=1.1.0
python-pptx>=0.6.23
openpyxl>=3.1.0
beautifulsoup4>=4.12.0
Pillow>=10.3.0

# OCR
pytesseract>=0.3.10
opencv-python-headless>=4.9.0
imutils>=0.5.4

# Code Parsing
tree-sitter>=0.22.0
tree-sitter-python>=0.21.0
tree-sitter-c>=0.21.0
tree-sitter-java>=0.21.0

# Output Generation
weasyprint>=62.0
jinja2>=3.1.0

# Security
cryptography>=42.0.0

# Config
pydantic>=2.6.0
pydantic-settings>=2.2.0
tomli-w>=1.0.0

# Logging
structlog>=24.1.0
rich>=13.7.0

# Launcher
textual>=0.55.0
typer>=0.12.0
```

### `requirements-dev.txt`

```
# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=5.0.0

# Code quality
ruff>=0.4.0
mypy>=1.9.0

# Type stubs
types-Pillow
types-beautifulsoup4
```

### Configuring Requirements for a New Environment

The `requirements.txt` is the single production dependency file. To adjust it:

1. Edit `requirements.txt` directly — pin, unpin, or swap libraries.
2. Run `python launcher.py --install` to apply changes.
3. The `--install` flag runs: `pip install -r requirements.txt` with verbose output.
4. GPU variants (e.g. `torch` with CUDA) should be added to a `requirements-gpu.txt` and activated with `--gpu` flag on the launcher.

---

## Error Hierarchy (`core/errors.py`)

```python
class DocForgeError(Exception):          pass

# Ingestion
class IngestError(DocForgeError):        pass
class OcrError(IngestError):             pass

# Plugins
class PluginInstallError(DocForgeError): pass
class PluginLoadError(DocForgeError):    pass
class PluginParseError(DocForgeError):   pass
class PluginWriteError(DocForgeError):   pass

# Embeddings & Storage
class EmbeddingError(DocForgeError):     pass
class StoreError(DocForgeError):         pass

# Agents
class AgentError(DocForgeError):         pass
class CodeParseError(AgentError):        pass

# Template
class TemplateAnalysisError(DocForgeError): pass

# Output
class OutputError(DocForgeError):        pass

# LLM
class LLMError(DocForgeError):           pass
class LLMTimeoutError(LLMError):         pass
```

---

## Quick Reference Card

```
MODES         Instruction-to-Doc · Code-to-Doc (separate agents, separate models)
MODELS        gemma3:7b (ITD) · qwen2.5-coder:7b (CTD) · nomic-embed-text (embed)
VECTOR DB     LanceDB (embedded, Arrow, zero server, CPU-fast)
OCR           PyMuPDF (text PDF) → Tesseract (scanned) → OpenCV (preprocess)
PARSING       tree-sitter: Python · C/C++ · Java
TEMPLATES     Inferred from uploaded .docx (font/style/header/footer/letterhead)
PLUGINS       ZIP drag-and-drop → Settings · pip for distribution
ENCRYPTION    Fernet AES-128 · key in data/.keystore · toggle via config
PROJECTS      Directory namespacing only · no auth · full isolation
OUTPUTS       DOCX · PDF (weasyprint) · Markdown · Excel · Images (plugin-ext.)
LAUNCHER      Typer CLI + Textual TUI · --install · --check · --debug · --test
LOGGING       structlog + Rich (terminal, color-coded) + RotatingFileHandler (file)
LOG SWITCHES  Per-module checkboxes in TUI and Settings tab · persisted to user.toml
PYTHON        3.11 · conda for dev · single requirements.txt for prod
OS            Windows-first · cross-platform architecture
DOCKER        3 services: ollama · docforge · model-puller (init)
DOCKER DEV    docker compose -f docker-compose.yml -f docker-compose.dev.yml up
DOCKER GPU    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
UI PORT       http://localhost:8501
OLLAMA PORT   http://localhost:11434
```

---

## 21. Docker Architecture

### Service Map

```
┌─────────────────────────────────────────────────────────────────┐
│  Docker Network: docforge_net                                   │
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │   ollama      │    │   docforge    │    │  model-puller   │  │
│  │               │◄───│               │    │  (init, exits)  │  │
│  │  :11434       │    │  :8501        │    │                 │  │
│  │               │    │               │    │  pulls models   │  │
│  │  ollama/      │    │  python:3.11  │    │  into shared    │  │
│  │  ollama:latest│    │  -slim        │    │  volume then    │  │
│  └───────┬───────┘    └───────────────┘    │  exits          │  │
│          │                                 └─────────────────┘  │
│  Volume: ollama_models (shared between ollama + model-puller)   │
│  Volume: ./data        (bind mount — persists on host)          │
│  Volume: ./config      (bind mount — editable on host)          │
│  Volume: ./plugins     (bind mount — hot plugin installs)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Host Browser    │
                    │  localhost:8501   │
                    └───────────────────┘
```

### Why three services

| Service | Role | Exits when done? |
|---------|------|-----------------|
| `ollama` | Serves LLM and embedding requests | No — long-running |
| `docforge` | Runs Streamlit app | No — long-running |
| `model-puller` | Pulls `gemma3:7b`, `qwen2.5-coder:7b`, `nomic-embed-text` on first boot | Yes — init container |

The `model-puller` is an init container pattern: it runs once, pulls all required models into the shared `ollama_models` volume, then exits with code 0. On subsequent `docker compose up`, Ollama finds the models already cached and the puller exits instantly.

### Persistent Data Strategy

| What | How | Where on host |
|------|-----|---------------|
| Ollama models | Named Docker volume | Managed by Docker |
| Project data (files, vectors, outputs) | Bind mount | `./data/` |
| Config (user overrides) | Bind mount | `./config/` |
| Plugins | Bind mount | `./plugins/` |
| Encryption key | Inside `./data/.keystore` | `./data/` |
| Logs | Inside `./data/logs/` | `./data/` |

Bind mounts (`./data`, `./config`, `./plugins`) are used for everything the developer needs to inspect, back up, or edit on the host. Named volumes are only used for Ollama's model weights which are large binary blobs.

---

## 22. Dockerfile

```dockerfile
# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

# Metadata
LABEL maintainer="DocForge"
LABEL description="Agentic RAG Document Generator"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# ── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # PDF → image conversion (pdf2image backend)
    poppler-utils \
    # OpenCV headless runtime deps
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # WeasyPrint font rendering
    fonts-liberation \
    fonts-dejavu-core \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    # Misc utilities
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 docforge \
 && useradd  --uid 1001 --gid docforge --shell /bin/bash --create-home docforge

WORKDIR /app

# ── Python Dependencies (cached layer) ────────────────────────────────────────
# Copy only requirements first so this layer is rebuilt only when deps change
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Tree-sitter Language Grammars ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    tree-sitter-python \
    tree-sitter-c \
    tree-sitter-java

# ── Application Source ────────────────────────────────────────────────────────
COPY --chown=docforge:docforge . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p \
    data/projects \
    data/logs \
    plugins/installed \
    config \
 && chown -R docforge:docforge data plugins config

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY --chown=docforge:docforge docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER docforge

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
```

---

## 23. Docker Compose — Full Stack

### `docker-compose.yml`

```yaml
# DocForge — Base Compose (CPU, production-like)
# Usage:   docker compose up --build
# Dev:     docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
# GPU:     docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

version: "3.9"

services:

  # ── Ollama LLM Server ───────────────────────────────────────────────────────
  ollama:
    image: ollama/ollama:latest
    container_name: docforge_ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - docforge_net
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      start_period: 15s
      retries: 6
    environment:
      - OLLAMA_HOST=0.0.0.0

  # ── Model Puller (init container — exits after pulling) ─────────────────────
  model-puller:
    image: ollama/ollama:latest
    container_name: docforge_model_puller
    restart: "no"
    depends_on:
      ollama:
        condition: service_healthy
    networks:
      - docforge_net
    volumes:
      - ollama_models:/root/.ollama
      - ./docker/pull-models.sh:/pull-models.sh:ro
    entrypoint: ["/bin/bash", "/pull-models.sh"]
    environment:
      - OLLAMA_HOST=http://ollama:11434
      # Comma-separated list of models to pull. Override via .env
      - DOCFORGE_MODELS=${DOCFORGE_MODELS:-gemma3:7b,qwen2.5-coder:7b,nomic-embed-text}

  # ── DocForge App ────────────────────────────────────────────────────────────
  docforge:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: docforge_app
    restart: unless-stopped
    ports:
      - "8501:8501"
    depends_on:
      ollama:
        condition: service_healthy
      model-puller:
        condition: service_completed_successfully
    networks:
      - docforge_net
    volumes:
      # Bind mounts — data visible and editable on host
      - ./data:/app/data
      - ./config:/app/config
      - ./plugins:/app/plugins
    env_file:
      - .env
    environment:
      # Override Ollama URL to point at the service name inside Docker network
      - DOCFORGE_OLLAMA__BASE_URL=http://ollama:11434
      - DOCFORGE_APP__LOG_LEVEL=${LOG_LEVEL:-INFO}
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8501/_stcore/health"]
      interval: 15s
      timeout: 5s
      start_period: 40s
      retries: 3

# ── Networks ──────────────────────────────────────────────────────────────────
networks:
  docforge_net:
    driver: bridge

# ── Volumes ───────────────────────────────────────────────────────────────────
volumes:
  ollama_models:
    name: docforge_ollama_models
```

---

## 24. Docker Compose — Dev Overrides

### `docker-compose.dev.yml`

Mount source code as a bind mount so Streamlit's hot-reload picks up changes
without rebuilding the image. Log level defaults to DEBUG.

```yaml
# Dev overrides — merge on top of docker-compose.yml
# Usage: docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

version: "3.9"

services:

  docforge:
    build:
      context: .
      dockerfile: Dockerfile
      # In dev, target the base stage (no production optimisations)
      target: base
    volumes:
      # Bind mount full source so edits reflect immediately
      - .:/app
      # Keep data, config, plugins as bind mounts (already in base compose)
      - ./data:/app/data
      - ./config:/app/config
      - ./plugins:/app/plugins
    environment:
      - DOCFORGE_APP__LOG_LEVEL=DEBUG
      - DOCFORGE_OLLAMA__BASE_URL=http://ollama:11434
      # Tell Streamlit to watch files for auto-reload
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=watchdog
      - STREAMLIT_SERVER_RUN_ON_SAVE=true
    command: >
      streamlit run app/main.py
        --server.port=8501
        --server.address=0.0.0.0
        --server.fileWatcherType=watchdog
        --server.runOnSave=true
        --logger.level=debug

  ollama:
    # Expose Ollama on host in dev so you can call it directly
    ports:
      - "11434:11434"
```

---

## 25. Docker Compose — GPU Override

### `docker-compose.gpu.yml`

Enable NVIDIA GPU passthrough to Ollama for faster inference.
Requires: NVIDIA Container Toolkit installed on the host.

```yaml
# GPU override — adds NVIDIA runtime to Ollama
# Usage: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# Prereq: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

version: "3.9"

services:

  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all      # "all" or a specific count e.g. 1
              capabilities: [gpu]
    environment:
      - OLLAMA_HOST=0.0.0.0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

---

## 26. Entrypoint & Model Pull

### `docker/entrypoint.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║        DocForge — Starting Up            ║"
echo "╚══════════════════════════════════════════╝"

# ── Config sanity check ───────────────────────────────────────────────────────
echo "[1/3] Validating configuration..."
python -c "
from core.config import load_config
cfg = load_config()
print(f'  ✅ Config loaded — ITD model: {cfg.models.itd_model}')
print(f'  ✅ Ollama URL:     {cfg.ollama.base_url}')
"

# ── Wait for Ollama (belt-and-suspenders beyond Docker healthcheck) ───────────
echo "[2/3] Confirming Ollama connectivity..."
MAX_ATTEMPTS=20
ATTEMPT=0
until curl -sf "${DOCFORGE_OLLAMA__BASE_URL:-http://ollama:11434}/api/tags" > /dev/null 2>&1; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "  ❌ Ollama did not become ready after ${MAX_ATTEMPTS} attempts. Exiting."
        exit 1
    fi
    echo "  ⏳ Waiting for Ollama... (${ATTEMPT}/${MAX_ATTEMPTS})"
    sleep 3
done
echo "  ✅ Ollama is reachable"

# ── Ensure plugin and data directories exist ──────────────────────────────────
mkdir -p /app/data/projects /app/data/logs /app/plugins/installed

# ── Load all installed plugins ────────────────────────────────────────────────
echo "[3/3] Loading plugins..."
python -c "
from plugins.loader import load_all_installed_plugins
plugins = load_all_installed_plugins()
print(f'  ✅ {len(plugins)} plugin(s) loaded')
"

echo ""
echo "  🚀 Starting Streamlit on port 8501..."
echo ""

exec streamlit run app/main.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.fileWatcherType=none \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level="${STREAMLIT_LOG_LEVEL:-info}"
```

### `docker/pull-models.sh`

```bash
#!/bin/bash
set -euo pipefail

# Wait for Ollama to be reachable
echo "[model-puller] Waiting for Ollama at ${OLLAMA_HOST}..."
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    echo "[model-puller] Ollama not ready, retrying in 5s..."
    sleep 5
done
echo "[model-puller] ✅ Ollama is up"

# DOCFORGE_MODELS is a comma-separated list e.g. "gemma3:7b,qwen2.5-coder:7b,nomic-embed-text"
IFS=',' read -ra MODELS <<< "${DOCFORGE_MODELS:-gemma3:7b,qwen2.5-coder:7b,nomic-embed-text}"

for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo -e "${MODEL}" | tr -d '[:space:]')"

    # Check if already pulled — Ollama returns it in /api/tags if present
    if curl -sf "${OLLAMA_HOST}/api/tags" | grep -q "\"${MODEL}\""; then
        echo "[model-puller] ✅ ${MODEL} already present, skipping"
        continue
    fi

    echo "[model-puller] ⬇  Pulling ${MODEL}..."
    OLLAMA_HOST="${OLLAMA_HOST}" ollama pull "${MODEL}"
    echo "[model-puller] ✅ ${MODEL} ready"
done

echo "[model-puller] All models ready. Exiting."
exit 0
```

---

## 27. Environment Variables & Config Bridge

### `.env.example`

Copy to `.env` and edit before running `docker compose up`.

```dotenv
# ── Model selection ───────────────────────────────────────────────────────────
# Comma-separated list pulled by model-puller on first boot
DOCFORGE_MODELS=gemma3:7b,qwen2.5-coder:7b,nomic-embed-text

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO                  # DEBUG | INFO | WARNING | ERROR

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_LOG_LEVEL=info

# ── Encryption ───────────────────────────────────────────────────────────────
# Set to false to disable file encryption (faster, less secure)
# DOCFORGE_ENCRYPTION__ENABLED=true

# ── Agent tuning ─────────────────────────────────────────────────────────────
# DOCFORGE_AGENTS__MAX_RETRIEVAL_CHUNKS=8
# DOCFORGE_AGENTS__ITD_CONTEXT_BUDGET_TOKENS=3000

# ── OCR ───────────────────────────────────────────────────────────────────────
# DOCFORGE_OCR__DPI=300
```

### How Environment Variables Override TOML Config

`pydantic-settings` maps environment variables to nested config fields using
double-underscore (`__`) as the separator. The prefix is `DOCFORGE_`.

The `core/config.py` `DocForgeConfig` class must inherit from `BaseSettings`
(not `BaseModel`) for this to work:

```python
# core/config.py — update for env var support
from pydantic_settings import BaseSettings, SettingsConfigDict

class DocForgeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix        = "DOCFORGE_",
        env_nested_delimiter = "__",
        env_file          = ".env",
        env_file_encoding = "utf-8",
        extra             = "ignore",
    )

    ollama:      OllamaConfig         = OllamaConfig()
    models:      ModelsConfig         = ModelsConfig()
    agents:      AgentConfig          = AgentConfig()
    ocr:         OcrConfig            = OcrConfig()
    encryption:  EncryptionConfig     = EncryptionConfig()
    launcher:    LauncherConfig       = LauncherConfig()
    log_modules: LoggingModulesConfig = LoggingModulesConfig()
```

**Precedence (highest to lowest):**

```
OS environment variable
    └── .env file
        └── config/user.toml
            └── config/default.toml
```

### Mapping Reference

| Environment Variable | Config Field | Default |
|---------------------|-------------|---------|
| `DOCFORGE_OLLAMA__BASE_URL` | `ollama.base_url` | `http://localhost:11434` |
| `DOCFORGE_OLLAMA__TIMEOUT_SEC` | `ollama.timeout_sec` | `120` |
| `DOCFORGE_MODELS__ITD_MODEL` | `models.itd_model` | `gemma3:7b` |
| `DOCFORGE_MODELS__CTD_MODEL` | `models.ctd_model` | `qwen2.5-coder:7b` |
| `DOCFORGE_MODELS__EMBEDDING_MODEL` | `models.embedding_model` | `nomic-embed-text` |
| `DOCFORGE_APP__LOG_LEVEL` | `app.log_level` | `INFO` |
| `DOCFORGE_ENCRYPTION__ENABLED` | `encryption.enabled` | `true` |
| `DOCFORGE_AGENTS__MAX_RETRIEVAL_CHUNKS` | `agents.max_retrieval_chunks` | `8` |
| `DOCFORGE_OCR__DPI` | `ocr.dpi` | `300` |

---

## 28. Docker-Aware Launcher

The Textual TUI launcher is designed to run on the **host**, not inside a container,
because it needs a proper TTY and manages Docker itself. When running inside a container,
the launcher detects this and skips the TUI, delegating to a headless check instead.

### Docker Detection (`core/env.py`)

```python
from pathlib import Path

def is_running_in_docker() -> bool:
    """
    Returns True if the current process is running inside a Docker container.
    Checks for /.dockerenv (set by Docker runtime) or Docker cgroup markers.
    """
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup") as f:
            return "docker" in f.read() or "containerd" in f.read()
    except FileNotFoundError:
        return False
```

### Updated `launcher.py` — Docker Flags

```python
@app.command()
def main(
    # ... existing flags ...
    docker_build:   Annotated[bool, typer.Option("--docker-build",   help="Build Docker image")] = False,
    docker_up:      Annotated[bool, typer.Option("--docker-up",      help="docker compose up -d")] = False,
    docker_down:    Annotated[bool, typer.Option("--docker-down",    help="docker compose down")] = False,
    docker_logs:    Annotated[bool, typer.Option("--docker-logs",    help="Tail logs from all containers")] = False,
    docker_status:  Annotated[bool, typer.Option("--docker-status",  help="Show container status")] = False,
    docker_dev:     Annotated[bool, typer.Option("--docker-dev",     help="Use dev compose overrides")] = False,
    docker_gpu:     Annotated[bool, typer.Option("--docker-gpu",     help="Use GPU compose overrides")] = False,
) -> None:

    from core.env import is_running_in_docker
    if is_running_in_docker():
        console.print("[yellow]Running inside Docker — TUI disabled. Use --check for status.[/yellow]")
        if check:
            _do_check()
        raise typer.Exit()

    compose_files = ["docker-compose.yml"]
    if docker_dev: compose_files.append("docker-compose.dev.yml")
    if docker_gpu: compose_files.append("docker-compose.gpu.yml")
    compose_cmd   = _build_compose_cmd(compose_files)

    if docker_build:  _run_compose(compose_cmd + ["build"]); raise typer.Exit()
    if docker_up:     _run_compose(compose_cmd + ["up", "-d", "--build"]); raise typer.Exit()
    if docker_down:   _run_compose(compose_cmd + ["down"]); raise typer.Exit()
    if docker_logs:   _run_compose(compose_cmd + ["logs", "-f"]); raise typer.Exit()
    if docker_status: _run_compose(compose_cmd + ["ps"]); raise typer.Exit()

    # ... rest of existing routing ...

def _build_compose_cmd(files: list[str]) -> list[str]:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", f]
    return cmd

def _run_compose(cmd: list[str]) -> None:
    """Run a docker compose command, streaming output through Rich."""
    import subprocess
    from rich.live import Live
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        console.print(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        console.print(f"[red]Command exited with code {proc.returncode}[/red]")
```

### Extended Docker Flag Reference

| Flag | Behavior |
|------|----------|
| `--docker-build` | Builds the DocForge image only |
| `--docker-up` | Builds + starts all services detached |
| `--docker-down` | Stops and removes all containers |
| `--docker-logs` | Tails logs from all containers live |
| `--docker-status` | Prints container status table |
| `--docker-dev` | Adds `docker-compose.dev.yml` overlay (hot reload, DEBUG) |
| `--docker-gpu` | Adds `docker-compose.gpu.yml` overlay (NVIDIA passthrough) |

---

## 29. Dev Workflow with Docker

### First-Time Setup

```bash
# 1. Clone and enter the repo
git clone <repo> && cd docforge

# 2. Copy and edit environment file
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS

# 3. Build and start everything (CPU, standard)
python launcher.py --docker-up

# OR use docker compose directly:
docker compose up --build -d

# 4. Watch logs from all services
python launcher.py --docker-logs

# 5. Open the UI
start http://localhost:8501      # Windows
# xdg-open http://localhost:8501 # Linux
```

### Day-to-Day Dev (Hot Reload)

```bash
# Start with dev overrides (source bind-mounted + hot reload)
python launcher.py --docker-up --docker-dev

# OR:
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Edit any .py file → Streamlit reloads automatically
# Edit config/user.toml → restart docforge only:
docker compose restart docforge
```

### Useful One-Liners

```bash
# Run a health check inside the container
docker compose exec docforge python launcher.py --check

# Open a shell inside the app container
docker compose exec docforge bash

# Pull a new model without rebuilding
docker compose exec ollama ollama pull llama3.2:3b

# Inspect logs for a specific service
docker compose logs -f docforge
docker compose logs -f ollama
docker compose logs -f model-puller

# Wipe everything including volumes (full reset)
docker compose down -v

# Wipe containers only (keep ollama_models volume — saves re-downloading models)
docker compose down
```

### Windows Docker Desktop Notes

| Requirement | Detail |
|-------------|--------|
| Docker Desktop ≥ 4.27 | Install from https://www.docker.com/products/docker-desktop |
| WSL 2 backend | Enable in Docker Desktop → Settings → General → Use WSL 2 based engine |
| Memory allocation | Ollama + 7B model needs at least **10 GB RAM** allocated to WSL 2 |
| WSL 2 memory limit | Create `%USERPROFILE%\.wslconfig` with `[wsl2] memory=12GB` |
| File mounts | Keep the repo on the WSL 2 filesystem (`\\wsl$\Ubuntu\...`) for best I/O performance, not on the Windows `C:\` drive |
| GPU (optional) | Requires NVIDIA GPU + CUDA drivers on host + NVIDIA Container Toolkit |

### `.wslconfig` Example (place in `C:\Users\<you>\.wslconfig`)

```ini
[wsl2]
memory=12GB
processors=4
swap=4GB
```

### `.dockerignore`

```dockerignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Conda / virtualenv
.conda/
.venv/
env/
venv/

# Data (bind mounted at runtime — don't bake into image)
data/

# Logs
*.log
logs/

# Config user overrides (injected via env vars or bind mount)
config/user.toml
.env

# Dev tools
.git/
.gitignore
.mypy_cache/
.ruff_cache/
.pytest_cache/
tests/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### Startup Sequence Diagram

```
docker compose up
      │
      ├── [1] ollama starts (waits for healthcheck :11434/api/tags to pass)
      │         │
      ├── [2] model-puller starts (depends_on: ollama healthy)
      │         │   pulls gemma3:7b, qwen2.5-coder:7b, nomic-embed-text
      │         │   into shared ollama_models volume
      │         └── exits with code 0
      │
      └── [3] docforge starts (depends_on: ollama healthy + model-puller completed)
                │   entrypoint.sh validates config
                │   waits for Ollama connectivity
                │   loads installed plugins
                └── streamlit run app/main.py → :8501
```
