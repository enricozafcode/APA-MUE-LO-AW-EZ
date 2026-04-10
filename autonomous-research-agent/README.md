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
├── data/
├── configs/
├── logs/
├── notebooks/
├── src/
└── submission/
```

## What Goes Where

- `.gitignore`  
  Ignore local-only and heavy files (`data/`, `logs/`, `models/`, `.env`, caches).

- `README.md`  
  Main documentation: setup, architecture, and run instructions.

- `CONTRIBUTING.md`  
  Team collaboration workflow (branches, PRs, conflict handling, quality checklist).

- `requirements.txt`  
  Python dependencies for LLM orchestration, training, and audio processing.

- `.env.example`  
  Template for local environment variables. Copy to `.env` and edit locally.

- `data/`  
  Local Kaggle files and other local datasets.  
  For BirdCLEF, place downloaded competition data here (or add a subfolder like `data/birdclef/`).

- `configs/`  
  Runtime settings and prompt templates:
  - `prompts.yaml`: system/user templates used to guide experiment proposals.
  - `agent_config.json`: iteration limits, budget, and loop controls.

- `logs/`  
  Experiment memory and observability:
  - generated code snapshots,
  - run metadata,
  - metrics history,
  - error traces,
  - LLM reasoning summaries.

- `notebooks/`  
  EDA and manual baseline experiments (outside the autonomous loop).

- `src/`  
  Core Python application:
  - `agent.py`: autonomous control loop (plan -> generate -> execute -> evaluate -> iterate).
  - `llm_client.py`: provider-agnostic LLM interface (Ollama/OpenAI-compatible).
  - `code_executor.py`: isolated execution of generated training code.
  - `evaluator.py`: metric extraction and feedback payloads for the next iteration.

- `submission/`  
  Kaggle-ready outputs (`submission.csv`, notebooks for competition constraints, etc.).

## Quick Start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` based on your local LLM setup (for example, Ollama at `http://localhost:11434`).

### 3) Start local LLM

Example with Ollama:

```bash
ollama run qwen3-coder
```

### 4) Place BirdCLEF data

Download competition data and place it under `data/` (recommended: `data/birdclef/`).

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
