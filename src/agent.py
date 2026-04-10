from __future__ import annotations

import json
from pathlib import Path

from src.code_executor import CodeExecutor
from src.evaluator import Evaluator
from src.llm_client import LLMClient


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "agent_config.json"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    config = load_config(config_path)
    llm = LLMClient(provider=config["llm"]["provider"], model=config["llm"]["model"])
    evaluator = Evaluator()
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=config["execution"]["timeout_seconds"],
    )

    # Starter loop: one proposal + one no-op generated script.
    proposal = llm.generate_experiment("Propose the first BirdCLEF experiment.")
    (logs_dir / "proposal_iter_001.txt").write_text(proposal.content, encoding="utf-8")

    generated_script = logs_dir / "generated_iter_001.py"
    generated_script.write_text("print('starter experiment executed')\n", encoding="utf-8")

    result = executor.run_file(generated_script)
    summary = evaluator.build_summary(result.stdout, result.stderr)
    (logs_dir / "metrics_iter_001.json").write_text(
        json.dumps(summary.metrics, indent=2),
        encoding="utf-8",
    )

    print("Starter autonomous loop completed.")
    print(f"Execution success: {result.success}")
    print(f"Next analysis prompt: {summary.analysis_prompt}")


if __name__ == "__main__":
    main()
