# 🚀 DocForge

> Local-first, offline-capable, multimodal agentic document generation platform for engineering and technical documentation.

DocForge transforms raw source material — repositories, documents, templates, diagrams, specifications, and codebases — into structured, production-ready engineering documentation using local LLMs, retrieval pipelines, OCR, and multi-agent orchestration.

Designed for:

* Engineering teams
* Developers
* Technical writers
* Research workflows
* Enterprise offline environments
* Privacy-sensitive deployments

---

# ✨ Why DocForge?

Most AI documentation tools:

* Flood context windows
* Hallucinate architecture details
* Ignore formatting requirements
* Break on large repositories
* Depend on cloud APIs

DocForge was built specifically to solve those problems.

---

# 🔥 Core Features

<table>
<tr>
<td width="50%">

## 🤖 Agentic Generation

Instead of a single prompt → output flow, DocForge uses a multi-agent pipeline:

```text
Plan → Retrieve → Write → Validate → Rewrite
```

Each section is independently:

* Planned
* Retrieved
* Written
* Validated
* Refined

This dramatically improves:

* Coherence
* Technical grounding
* Completeness
* Context efficiency

</td>

<td width="50%">

## 🧠 Smart Retrieval

DocForge avoids naive RAG.

It uses:

* Multi-query retrieval
* Chunk deduplication
* Context budgeting
* Retrieval scoring
* Relevance thresholds
* Previously-used chunk penalties

Result:

* Less hallucination
* Better grounding
* Reduced context flooding
* More focused generations

</td>
</tr>
</table>

---

# 🧩 What Can It Generate?

## 📝 Instruction-to-Document (ITD)

Turn instructions + source material into:

| Document Type       | Example                            |
| ------------------- | ---------------------------------- |
| SRS                 | Software Requirement Specification |
| TDD                 | Technical Design Document          |
| Architecture Docs   | System architecture references     |
| Requirement Docs    | Product/engineering requirements   |
| Engineering Reports | Internal documentation             |
| Design Specs        | Detailed implementation specs      |

---

## 💻 Code-to-Document (CTD)

Turn repositories into:

```text
API Documentation
Module Documentation
Architecture References
Deployment Runbooks
Developer Guides
Technical Specifications
Dependency Maps
```

Supports:

* Python
* C
* C++
* Java

via Tree-sitter AST parsing.

---

# 🏗️ High-Level Architecture

```text
                ┌──────────────────────────────┐
                │          DocForge            │
                └──────────────────────────────┘

      ┌────────────────────┐   ┌────────────────────┐
      │ Instruction-to-Doc │   │    Code-to-Doc     │
      └────────────────────┘   └────────────────────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                  ┌───────────────────────┐
                  │   LangGraph Pipeline  │
                  │ Plan → Retrieve →     │
                  │ Write → Validate      │
                  └──────────┬────────────┘
                             ▼
                  ┌───────────────────────┐
                  │     Smart Retrieval   │
                  │ LanceDB + Embeddings  │
                  └──────────┬────────────┘
                             ▼
                  ┌───────────────────────┐
                  │   Output Generation   │
                  │ DOCX · PDF · MD       │
                  └───────────────────────┘
```

---

# ⚡ Multimodal Ingestion

DocForge ingests far more than plain text.

<div align="center">

| Source Type       | Supported |
| ----------------- | --------- |
| PDFs              | ✅         |
| Scanned Documents | ✅         |
| DOCX              | ✅         |
| PPTX              | ✅         |
| XLSX              | ✅         |
| HTML              | ✅         |
| Images            | ✅         |
| Tables            | ✅         |
| Source Code       | ✅         |

</div>

---

# 🎨 Template Fidelity

Unlike generic AI document generators, DocForge preserves formatting structure.

### Preserved Elements

* Typography
* Heading hierarchy
* Tables
* Header/footer layout
* Letterheads
* Section formatting
* Spacing rules
* Structural styling

This allows generated output to match enterprise/internal document standards.

---

# 🔒 Offline-First by Design

```text
No cloud dependency required.
No external APIs required.
No telemetry required.
```

Everything can run locally using:

* Ollama
* Local embeddings
* Local vector storage
* Local OCR
* Local generation pipelines

Perfect for:

* Air-gapped systems
* Internal enterprise tooling
* Sensitive engineering data
* Research labs

---

# 🧠 LangGraph Pipeline

```text
[START]
   ↓
plan_document
   ↓
select_next_section
   ↓
generate_retrieval_queries
   ↓
retrieve_chunks
   ↓
write_section
   ↓
validate_section
   ├── passed ──────────────┐
   └── failed → rewrite ───┘
   ↓
assemble_document
   ↓
apply_template
   ↓
generate_output
   ↓
[END]
```

---

# 🛠️ Tech Stack

<table>
<tr>
<td>

### Core Runtime

* Python 3.11
* Ollama
* LanceDB
* LangGraph

</td>

<td>

### UI / Developer Experience

* Streamlit
* Textual
* Rich
* Typer

</td>
</tr>

<tr>
<td>

### OCR + Parsing

* Tesseract
* OpenCV
* PyMuPDF
* Tree-sitter

</td>

<td>

### Output Generation

* WeasyPrint
* python-docx
* openpyxl
* Jinja2

</td>
</tr>
</table>

---

# 🧠 Default Models

| Purpose    | Model              |
| ---------- | ------------------ |
| ITD Writer | `gemma3:7b`        |
| CTD Writer | `qwen2.5-coder:7b` |
| Embeddings | `nomic-embed-text` |

---

# 📂 Repository Structure

The DocForge repository uses a nested folder layout to group application source code inside a single `docforge` Python package, keeping the root directory clean and manageable:

```text
DocForge/
├── docforge/                     # Main Python Package Source
│   ├── agents/                   # LangGraph pipeline & multi-agent orchestration
│   ├── app/                      # Streamlit application entry point
│   ├── core/                     # System configuration, constants, errors, logger
│   ├── embeddings/               # Ollama embeddings & LanceDB operations
│   ├── ingest/                   # Multimodal file parsing & ingestion
│   ├── llm/                      # Async Ollama client & prompt builders
│   ├── ocr/                      # OpenCV and Tesseract OCR modules
│   ├── output/                   # DOCX, PDF, Excel, and Markdown writers
│   ├── plugins/                  # Dynamic plugin loader & ZIP installations
│   ├── projects/                 # Project workspace management
│   ├── security/                 # Key management & file encryption
│   ├── template/                 # Document template style analysis
│   ├── tui/                      # Textual Terminal UI dashboard
│   └── ui/                       # Streamlit UI page files & components
├── config/                       # Application configuration TOML files
├── data/                         # Persistent local files (logs, cache, databases)
├── docs/                         # User guides and developer documentation
└── tests/                        # Unit and integration test suites
```

### 📂 Directory Substitutions

The following table provides the mapping from the original flat root layout to the new nested directory structure:

| Original Root Path | New Nested Path | Description |
| :--- | :--- | :--- |
| `agents/` | `docforge/agents/` | Multi-agent pipelines and graph definitions. |
| `app/` | `docforge/app/` | Web UI/Streamlit entry point files. |
| `core/` | `docforge/core/` | System configuration, errors, constants, logger. |
| `embeddings/` | `docforge/embeddings/` | Embedding clients and vector storage files. |
| `ingest/` | `docforge/ingest/` | Multi-format parsing and chunking engine. |
| `llm/` | `docforge/llm/` | LLM client API wrapper. |
| `ocr/` | `docforge/ocr/` | Local document pre-processing and OCR extraction. |
| `output/` | `docforge/output/` | Generation pipelines for DOCX, PDF, MD formats. |
| `plugins/` | `docforge/plugins/` | Runtime plugins and ZIP extension loader. |
| `projects/` | `docforge/projects/` | Project workspaces management files. |
| `security/` | `docforge/security/` | Key generation and data encryption. |
| `template/` | `docforge/template/` | Design and template layout analyzer. |
| `tui/` | `docforge/tui/` | Terminal TUI dashboard. |
| `ui/` | `docforge/ui/` | Streamlit pages and custom UI controls. |


---

# 🚀 Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-org/docforge.git
cd docforge
```

---

## 2️⃣ Create Environment

```bash
conda env create -f environment.yml
conda activate docforge
```

---

## 3️⃣ Install Models

```bash
ollama pull gemma3:7b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

---

## 4️⃣ Start Ollama

```bash
ollama serve
```

---

## 5️⃣ Launch DocForge

### Streamlit UI

```bash
streamlit run docforge/app/main.py
```

### Terminal UI

```bash
python launcher.py
```

---

# 🐳 Docker Support

## Standard Startup

```bash
docker compose up
```

## GPU Mode

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

# ⚙️ Configuration

DocForge uses TOML-based configuration.

### Default Config

```text
config/default.toml
```

### User Overrides

```text
config/user.toml
```

### Example

```toml
[models]
itd_model = "gemma3:7b"

[agents]
max_retrieval_chunks = 8
reviewer_enabled = true

[ocr]
dpi = 300
deskew = true
```

---

# 🔌 Plugin System

Plugins are ZIP-installable and dynamically loaded at runtime.

## Plugin Layout

```text
my_plugin.zip
├── plugin.toml
├── __init__.py
├── ingestor.py
└── output_writer.py
```

---

## Plugin Capabilities

| Plugin Type      | Purpose                |
| ---------------- | ---------------------- |
| Ingestor Plugins | Add new file parsers   |
| Output Plugins   | Add new export formats |

---

# 📊 Logging & Observability

DocForge exposes detailed runtime visibility.

### Included

* Structured logging
* Agent traces
* Retrieval traces
* Validation scoring
* Runtime diagnostics
* Module-level log filtering
* Rich terminal rendering

Logs are written to:

```text
data/logs/
```

---

# 🧪 Development

## Run Tests

```bash
pytest
```

## Lint

```bash
ruff check .
```

## Type Check

```bash
mypy .
```

---

# 🧭 Contributing

Contributions are welcome.

## 📌 Before Opening a Pull Request

Please:

1. Open an issue first for major changes
2. Discuss architectural changes before implementation
3. Keep PRs focused and scoped
4. Ensure all tests pass
5. Follow the developer guide conventions

---

## 🧹 Contribution Etiquette

### Do

* Write typed Python
* Use structured logging
* Follow module responsibility boundaries
* Add meaningful commit messages
* Prefer explicitness over abstraction
* Keep functions focused

### Don't

* Add giant utility files
* Introduce hidden global state
* Bypass interfaces
* Add untyped public APIs
* Hardcode model-specific logic into pipelines

---

# 💡 Feature Requests & Issues

Have an idea or found a bug?

## Open an Issue For:

* Feature requests
* Plugin ideas
* Architecture discussions
* Bug reports
* Performance issues
* Model support requests
* UI/UX improvements
* Documentation improvements

---

## 📝 Good Issues Include

```text
- Clear description
- Expected behavior
- Actual behavior
- Logs/screenshots if applicable
- Reproduction steps
- Environment details
```

---

# 🔮 Planned Features

<table>
<tr>
<td>

### Retrieval & Memory

* Knowledge graphs
* Multimodal embeddings
* Citation tracking
* Incremental indexing

</td>

<td>

### Platform Expansion

* Distributed ingestion
* Multi-user workspaces
* Web search augmentation
* Collaborative editing

</td>
</tr>
</table>

---

# 📖 Documentation Sources

This README is based on:

* Architecture reference 
* Developer guide 
* LangGraph pipeline specification 

---

# 📜 License

MIT License

---

# 🌟 Vision

DocForge aims to become a fully local, enterprise-grade technical documentation platform capable of generating structured engineering documentation from heterogeneous source material without relying on cloud infrastructure.
