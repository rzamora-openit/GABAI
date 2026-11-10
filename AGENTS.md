# AGENTS.md

Scope: this folder (`UPNEC`) only.

## Python environment

Use the conda env `agentic_ai` for all Python work in this folder.

- Path: `C:\Users\rzamora\AppData\Local\anaconda3\envs\agentic_ai`
- Interpreter: `C:\Users\rzamora\AppData\Local\anaconda3\envs\agentic_ai\python.exe` (Python 3.12.16)
- Install packages with `python.exe -m pip install ...` from that interpreter, not the base env.
- Run tests with `python.exe -m pytest`.

### Installed packages (key ones)

- **Agents / LLM:** langchain 1.4.9, langchain-core, langchain-classic, langchain-community, langchain-huggingface, langchain-ollama, langchain-text-splitters, langgraph 1.2.20 (+ checkpoint-sqlite, prebuilt), langsmith, ollama 0.6.3, huggingface_hub, transformers 5.22.2, sentence-transformers, tokenizers, safetensors, pydantic 2.14.3, pydantic-settings
- **ML:** torch 2.15.0, tensorflow 2.21.0, keras 3.15.1, keras-hub, scikit-learn, scipy, numpy 2.5.4, pandas 3.1.1
- **Retrieval / storage:** sqlite-vec, rank-bm25, SQLAlchemy 2.1.8, aiosqlite, pypdf
- **Web / UI:** Flask 3.1.3, dash 4.5.0, waitress, httpx, aiohttp, plotly, matplotlib
- **Dev:** pytest 9.2.0, ipykernel, ipython, jupyter_client, spyder-kernels, debugpy, python-dotenv
- **Other:** kagglehub

Run `python.exe -m pip list` in that env for the full list.

## App entry point

[app.py](app.py) is the main file: the Streamlit app. Run it from this folder with `python.exe -m streamlit run app.py`. Other pages live in [pages/](pages/) and are shown in the sidebar.

Shared helpers live in [Utility/](Utility/) (`env.py` for `.env` access, `config.py` for settings). Import them from the project root, e.g. `from Utility.env import env`. Do not run files inside `Utility/` directly.

The LLM query layer lives in [LLM/](LLM/). Pages call `get_provider(name).stream(messages, tools)` and handle the `ThinkingDelta`, `ContentDelta` and `ToolCall` events; they never call Ollama or Azure directly. To add a model, add an entry to `PROVIDERS` in `LLM/__init__.py`. To add a backend, subclass `ChatProvider` in `LLM/base.py`.

## Model schemas

[SCHEMAS.md](SCHEMAS.md) is the blueprint for the system's model schemas. The classes in [Model/](Model/) must follow it.

- **At the start of each user session only:** cross-reference `SCHEMAS.md` against the classes in `Model/` (class names, fields, types, defaults, relationships).
- Report any inconsistency to the user before doing other work. Do not fix it unasked; `SCHEMAS.md` is the source of truth, so ask which side to change if unclear.
- Do not repeat the check later in the session unless the user asks.
