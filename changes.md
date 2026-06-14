# Changes & Directory Restructuring Record

The following table documents the additions, deletions, moves, and restructuring changes applied to the DocForge repository.

| Original Path | New Path / Substitution | Action | Reason |
| :--- | :--- | :--- | :--- |
| `agents/` | `docforge/agents/` | **Moved & Nested** | Consolidate agents source code under the main `docforge` package directory. |
| `app/main.py` | `docforge/app/main.py` | **Moved & Nested** | Part of nesting the application source directories. |
| `config/` | `config/` (with `.gitkeep`) | **Restructured** | Cleared out any untracked or unnecessary cached folders and re-created the target empty structure with `.gitkeep`. |
| `core/` | `docforge/core/` | **Moved & Nested** | Nest common system settings, configurations, and core exceptions under the `docforge` package. |
| `data/` | `data/` (with `.gitkeep`) | **Restructured** | Cleared out temporary run state and re-created clean directories for runtime logs, cache, etc. |
| `embeddings/` | `docforge/embeddings/` | **Moved & Nested** | Keep vector store client and integration logic inside the `docforge` module. |
| `ingest/` | `docforge/ingest/` | **Moved & Nested** | Keep file ingest and parsing pipeline code structured under the package root. |
| `llm/` | `docforge/llm/` | **Moved & Nested** | Move LLM clients and prompt builders under the package root. |
| `ocr/` | `docforge/ocr/` | **Moved & Nested** | Move OCR detectors, preprocessors, and extractors under the package root. |
| `output/` | `docforge/output/` | **Moved & Nested** | Move document/file writers under the package root. |
| `plugins/` | `docforge/plugins/` | **Moved & Nested** | Group plugin registry, loader, and base classes under the package root. |
| `projects/` | `docforge/projects/` | **Moved & Nested** | Move namespace/project managers under the package root. |
| `security/` | `docforge/security/` | **Moved & Nested** | Move encryption and security modules under the package root. |
| `template/` | `docforge/template/` | **Moved & Nested** | Move template analyzer and rendering engines under the package root. |
| `tests/` | `tests/` (with `.gitkeep`) | **Restructured** | Cleared out untracked test run directories and established a clean structure at the root. |
| `tui/` | `docforge/tui/` | **Moved & Nested** | Consolidate the Textual TUI code into the main package. |
| `ui/` | `docforge/ui/` | **Moved & Nested** | Move Streamlit visual components and pages under the main package. |
| `.venv/` | `.venv/` | **Reinstalled** | Deleted and cleanly reinstalled using `uv sync` to ensure correct development runtime packages. |
| `changes.md` | `changes.md` | **Added** | Document restructuring substitutions and reasons for changes. |
