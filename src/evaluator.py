from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvaluationSummary:
    metrics: Dict[str, Any]
    analysis_prompt: str


class Evaluator:
    """Converts run outputs into structured feedback for the LLM."""

    def build_summary(self, stdout: str, stderr: str) -> EvaluationSummary:
        # TODO: parse real training logs/JSON artifacts to extract metrics.
        metrics = {
            "status": "ok" if not stderr else "warning",
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
        }
        analysis_prompt = (
            "Analyze this run and propose the next experiment.\n"
            f"Metrics: {metrics}\n"
        )
        return EvaluationSummary(metrics=metrics, analysis_prompt=analysis_prompt)
