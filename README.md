# Autonomous Research Agent (BirdCLEF Track)

AI-powered autonomous experimentation loop for the APA 2025/2026 course project.  
This repository is structured so the agent can propose, generate, run, evaluate, and iterate on deep learning experiments for BirdCLEF.

## Project Architecture

```text
autonomous-research-agent/
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .env.example
├── data/              # folder in git; contents are gitignored (see below)
├── configs/
├── logs/
├── notebooks/
├── scripts/
├── src/
└── submission/
```

## What Goes Where

- `.gitignore`  
  Ignores local-only files: everything under `data/` **except** `data/.gitkeep`, plus `logs/`, `models/`, `.env`, caches.

- `data/`  
  **Committed as an empty folder** (`data/.gitkeep`). Each developer **copies or extracts** the BirdCLEF competition files here locally (or sets `BIRDCLEF_DATA_DIR` in `.env` to another path). Nothing large is pushed to git.

- `README.md`  
  Main documentation: setup, architecture, and run instructions.

- `CONTRIBUTING.md`  
  Team collaboration workflow (branches, PRs, conflict handling, quality checklist).

- `requirements.txt`  
  Python dependencies for LLM orchestration, training, and audio processing.

- `.env.example`  
  Template for optional local variables. Copy to `.env` and edit locally (never commit `.env`).

- `configs/`  
  Runtime settings and prompt templates:
  - `prompts.yaml`: system/user templates used to guide experiment proposals.
  - `agent_config.json`: iteration limits, budget, and loop controls.

- `logs/`  
  Experiment memory and observability (snapshots, metrics, errors, LLM summaries).

- `notebooks/`  
  EDA and manual baseline experiments (outside the autonomous loop).

- `scripts/`  
  `setup_project.py`: create `.venv` and install dependencies.

- `src/`  
  Core Python application:
  - `paths.py`: `repo_root()` and `birdclef_data_dir()` (optional `BIRDCLEF_DATA_DIR` in `.env`; default `data/`).
  - `agent.py`: autonomous control loop (plan → generate → execute → evaluate → iterate).
  - `llm_client.py`: provider-agnostic LLM interface (Ollama/OpenAI-compatible).
  - `code_executor.py`: isolated execution of generated training code.
  - `evaluator.py`: metric extraction and feedback for the next iteration.

- `submission/`  
  Kaggle-ready outputs (`submission.csv`, notebooks, etc.).

## Quick Start

### 1) Add the dataset locally (team)

Download the competition from [Kaggle](https://www.kaggle.com/) (browser or your own tooling) and place the extracted files in the repo’s **`data/`** directory (same layout as on Kaggle).  
The **`data/`** directory is **in the repository**; **only** `data/.gitkeep` is tracked—your files stay on your machine.


In code, use:

```python
from src.paths import birdclef_data_dir

root = birdclef_data_dir()
```

### 2) Optional: `.env` for LLM settings

```bash
cp .env.example .env
```

Edit `.env` if you use non-default LLM URLs (see `src/llm_client.py`).

### 3) Python environment

From the repo root (Python **3.9+**):

```bash
python3 scripts/setup_project.py
```

Windows (example): `py -3 scripts\setup_project.py`

Then activate:

```bash
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

**Manual equivalent:** `python -m venv .venv`, activate, `pip install -r requirements.txt`.

### 4) Local LLM (when you use the agent)

Example with Ollama:

```bash
ollama run qwen3-coder
```

### 5) Run the agent (starter path)

```bash
python -m src.agent
```

## Suggested Internal Conventions

- Keep each experiment reproducible with a run id.
- Save each prompt and generated code artifact under `logs/`.
- Log both success and failure runs to avoid repeating bad ideas.
- Start with cheap experiments (small models, short epochs), then scale promising candidates.

## Deliverables Alignment

This structure is designed to support:
- reproducible code setup (`requirements.txt`, `.env.example`),
- clear implementation and architecture (`src/`, `configs/`),
- experiment tracking for report/video (`logs/`),
- Kaggle output artifacts (`submission/`),
- documented team process (`CONTRIBUTING.md`).
