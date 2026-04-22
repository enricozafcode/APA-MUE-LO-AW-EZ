from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow running as `python src/agent.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_code(llm_response: str) -> str:
    """Pull the Python code block out of an LLM response.

    LLMs often wrap code in ```python ... ``` fences — we strip that here.
    If there is no fence, we assume the whole response is code.
    """
    match = re.search(r"```python\s*\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response.strip()


def build_history_summary(experiment_log: list[dict]) -> str:
    """Summarise past experiments so the LLM knows what has been tried."""
    if not experiment_log:
        return "No experiments run yet. This is the first iteration."

    lines = []
    for entry in experiment_log:
        metrics = entry.get("metrics", {})
        score = metrics.get("roc_auc", "n/a")
        status = "OK" if entry.get("success") else "FAILED"
        hypothesis = entry.get("hypothesis", "unknown")
        lines.append(
            f"  Iteration {entry['iteration'] + 1}: [{status}] "
            f"roc_auc={score}  --  {hypothesis}"
        )
        if entry.get("error_summary"):
            lines.append(f"    Error: {entry['error_summary']}")

    best = max(
        (e.get("metrics", {}).get("roc_auc", 0) for e in experiment_log),
        default=0,
    )
    lines.append(f"\n  Best ROC-AUC so far: {best:.4f}")
    return "\n".join(lines)


def save_experiment_log(log: list[dict], logs_dir: Path) -> None:
    path = logs_dir / "experiment_log.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_experiment_log(logs_dir: Path) -> list[dict]:
    path = logs_dir / "experiment_log.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "agent_config.json"
    prompts_path = project_root / "configs" / "prompts.yaml"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    config = load_config(config_path)
    prompts = load_prompts(prompts_path)

    max_iterations = config.get("max_iterations", 5)
    max_failures = config.get("max_failures_before_stop", 3)
    data_dir = str(project_root / "data")

    print("=== Autonomous BirdCLEF Agent ===")
    print(f"Model      : {config['llm']['model']} via {config['llm']['provider']}")
    print(f"Iterations : {max_iterations}")
    print(f"Data dir   : {data_dir}")
    print()

    llm = LLMClient(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
    )
    # Resolve python executable to absolute path if it's relative
    python_exec = config["execution"]["python_executable"]
    python_exec_path = project_root / python_exec
    if python_exec_path.exists():
        python_exec = str(python_exec_path)

    executor = CodeExecutor(
        python_executable=python_exec,
        timeout_seconds=config["execution"]["timeout_seconds"],
    )
    evaluator = Evaluator()

    # Resume from previous run if a log already exists
    experiment_log = load_experiment_log(logs_dir)
    best_roc_auc = max(
        (e.get("metrics", {}).get("roc_auc", 0) for e in experiment_log),
        default=0.0,
    )
    consecutive_failures = 0
    start_iteration = len(experiment_log)

    for iteration in range(start_iteration, start_iteration + max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}  |  best so far: {best_roc_auc:.4f}")
        print(f"{'='*60}")

        # ── 1. Build the prompt ──────────────────────────────────────
        history_summary = build_history_summary(experiment_log)
        user_prompt = prompts["experiment_prompt_template"].format(
            previous_experiments=history_summary,
            compute_budget=f"{config.get('time_budget_minutes', 180)} minutes total",
            data_dir=data_dir,
        )

        # ── 2. Ask the LLM ──────────────────────────────────────────
        print("Asking LLM for next experiment...")
        raw_response = llm.generate_code(
            system_prompt=prompts["system_prompt"],
            user_prompt=user_prompt,
            temperature=config["llm"].get("temperature", 0.2),
        )

        if not raw_response.strip():
            print("LLM returned empty response - skipping.")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("Too many consecutive failures. Stopping.")
                break
            continue

        # Save full LLM response for debugging
        (logs_dir / f"llm_response_{iteration:03d}.txt").write_text(
            raw_response, encoding="utf-8"
        )

        # ── 3. Extract hypothesis (first non-empty line) ─────────────
        hypothesis = next(
            (ln.strip() for ln in raw_response.splitlines() if ln.strip()),
            "unknown",
        )[:120]
        print(f"Hypothesis : {hypothesis}")

        # ── 4. Extract Python code ───────────────────────────────────
        code = extract_code(raw_response)
        if not code:
            print("Could not extract code from LLM response - skipping.")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("Too many consecutive failures. Stopping.")
                break
            continue

        # ── 5. Save and run the script ───────────────────────────────
        script_path = logs_dir / f"experiment_{iteration:03d}.py"
        script_path.write_text(code, encoding="utf-8")
        print(f"Script saved -> {script_path.name}")

        print("Running experiment...")
        result = executor.run_file(script_path)
        print(f"Status     : {'SUCCESS' if result.success else 'FAILED'}")

        if result.stdout:
            print("\n--- stdout ---")
            print(result.stdout[:2000])
        if result.stderr:
            print("\n--- stderr ---")
            print(result.stderr[:500])

        # ── 6. Parse metrics ─────────────────────────────────────────
        summary = evaluator.build_summary(result.stdout, result.stderr)
        roc_auc = summary.metrics.get("roc_auc", 0.0)
        print(f"\nMetrics    : {summary.metrics}")

        # ── 7. Record everything ─────────────────────────────────────
        error_lines = [l for l in (result.stderr or "").splitlines() if l.strip()]
        error_summary = error_lines[-1][:200] if error_lines and not result.success else None

        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "success": result.success,
            "metrics": summary.metrics,
            "error_summary": error_summary,
            "script": script_path.name,
        }
        experiment_log.append(record)
        save_experiment_log(experiment_log, logs_dir)

        # ── 8. Track best model ──────────────────────────────────────
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            (logs_dir / "best_experiment.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(f"New best ROC-AUC: {best_roc_auc:.4f}")

        # ── 9. Failure tracking ──────────────────────────────────────
        if result.success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"\nStopping: {max_failures} consecutive failures reached.")
                break

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Agent finished.")
    print(f"Total experiments : {len(experiment_log)}")
    print(f"Best ROC-AUC      : {best_roc_auc:.4f}")
    print(f"Full log          : {logs_dir / 'experiment_log.json'}")


if __name__ == "__main__":
    main()
