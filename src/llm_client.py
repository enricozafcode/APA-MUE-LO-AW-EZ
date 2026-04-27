"""Client for communicating with a locally-hosted LLM via OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

_PLAN_SCHEMA = """\
{
  "search_stage": "exploration" | "refinement",
  "change_summary": "<what changed vs previous run, or 'first run'>",
  "reasoning": "<why this choice>",
  "conv_layers": [
    {"filters": <int>, "kernel_size": <int>},
    ...
  ],
  "dropout": <float 0.0-0.4>,
  "learning_rate": <float>,
  "batch_size": <int>,
  "epochs": <int>
}"""

_PLAN_CONSTRAINTS = """\
Hard constraints (never violate):
- Input representation: mel-spectrogram (fixed, not negotiable)
- Model family: 2D CNN only (Conv2D + MaxPooling2D layers)
- conv_layers: 2 to 4 entries
- filters per layer: 16 to 256
- kernel_size: 3 or 5
- dropout: 0.0 to 0.4
- learning_rate: 1e-4 to 1e-2
- batch_size: 8, 16, or 32
- epochs: 3 (fixed for now)
- search_stage: must be exactly "exploration" or "refinement"
- Output ONLY valid JSON — no markdown, no explanation outside the JSON.

Not yet in search space (do not change these, just leave them at defaults):
- num_mels: 64 (fixed — future extension)
- spectrogram_resolution: standard/hop_length=512 (fixed — future extension)
- thresholding_strategy: none (fixed — future extension)"""

_STRATEGY_HINTS: dict = {
    "exploration": (
        "This is early-stage exploration. Try a meaningfully different architecture — "
        "vary filter depth, number of layers, or learning rate significantly. "
        "Set search_stage to 'exploration'."
    ),
    "last_failed_fix": (
        "The previous run FAILED. Make a minimal, safe change to fix the issue. "
        "If unsure, use 2 small conv layers, low dropout, LR=0.001. "
        "Set search_stage to 'exploration'."
    ),
    "multiple_failures_simplify": (
        "Multiple consecutive failures. Simplify drastically: use exactly 2 conv layers "
        "with small filters (16–32), dropout=0.0, LR=0.001, batch_size=16. "
        "Set search_stage to 'exploration'."
    ),
    "improved_continue": (
        "The last run improved on the best result so far. Continue in the same direction — "
        "make only a small refinement (e.g. slightly adjust LR or dropout). "
        "Set search_stage to 'refinement'."
    ),
    "succeeded_weak_change": (
        "The last run succeeded but performance is weak. Change only 1–2 parameters "
        "(e.g. increase filters, or lower LR). Do not redesign the whole architecture. "
        "Set search_stage to 'exploration'."
    ),
    "no_improvement_switch": (
        "No meaningful improvement across the last few runs. Switch to a clearly different "
        "configuration — change both the architecture depth and the learning rate. "
        "Set search_stage to 'exploration'."
    ),
}

_ANALYSIS_SCHEMA = """\
{
  "outcome": "improved" | "regressed" | "failed",
  "likely_cause": "<capacity | overfitting | underfitting | learning_rate | data | other>",
  "next_step": "<one concrete hyperparameter suggestion>"
}"""


class LLMClient:
    """Wraps the local LLM server (Ollama / LM Studio)."""

    def __init__(self, provider: str = "ollama", model: str = "gemma3:4b") -> None:
        self.model_name = model
        base_url = (
            "http://localhost:11434/v1" if provider.lower() == "ollama"
            else "http://localhost:1234/v1"
        )
        self.client = OpenAI(base_url=base_url, api_key="local-dummy-key", timeout=1800.0)

    def _call(self, system: str, user: str, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strips markdown fences and extracts the first JSON object."""
        lines = text.splitlines()
        # Remove leading ``` fence
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove trailing ``` fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def generate_plan(
        self,
        data_summary: Dict[str, Any],
        history: List[Dict[str, Any]],
        strategy: str = "exploration",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Asks the LLM to propose CNN hyperparameters for the next experiment.

        Returns a validated plan dict. Falls back to a safe default on parse failure.
        """
        history_block = ""
        if history:
            recent = history[-3:]
            rows = []
            for h in recent:
                m = h.get("metrics", {})
                p = h.get("plan", {})
                exp_id = h.get("experiment_id", h.get("exp_id", "?"))
                metric = h.get("main_metric", m.get("val_auc", "n/a"))
                status = h.get("status", "success" if m.get("success") else "failed")
                rows.append(
                    f"  - {exp_id}: layers={p.get('conv_layers')}, "
                    f"lr={p.get('learning_rate')}, dropout={p.get('dropout')}, "
                    f"val_auc={metric}, status={status}, "
                    f"stage={p.get('search_stage', 'exploration')}"
                )
            history_block = "Previous experiments (most recent last):\n" + "\n".join(rows) + "\n\n"

        strategy_instruction = _STRATEGY_HINTS.get(strategy, _STRATEGY_HINTS["exploration"])

        system = (
            "You are an ML researcher designing a CNN for BirdCLEF audio classification. "
            "Your job is to choose hyperparameters for the next experiment. "
            + _PLAN_CONSTRAINTS
        )
        user = (
            f"Dataset: {data_summary['total_samples']} samples, "
            f"{data_summary['num_species']} species.\n\n"
            + history_block
            + f"Search strategy for this run: {strategy_instruction}\n\n"
            + "Propose the next experiment as JSON matching this schema exactly:\n"
            + _PLAN_SCHEMA
        )

        raw = self._call(system, user, temperature)
        try:
            plan = json.loads(self._extract_json(raw))
            # Migrate old 'rationale' key to 'reasoning'
            if "rationale" in plan and "reasoning" not in plan:
                plan["reasoning"] = plan.pop("rationale")
            elif "rationale" in plan:
                del plan["rationale"]
            # Ensure all required keys exist
            plan.setdefault("search_stage", "exploration")
            plan.setdefault("change_summary", "")
            plan.setdefault("reasoning", "")
            plan.setdefault("conv_layers", [{"filters": 32, "kernel_size": 3}])
            plan.setdefault("dropout", 0.0)
            plan.setdefault("learning_rate", 0.001)
            plan.setdefault("batch_size", 16)
            plan.setdefault("epochs", 3)
            return plan
        except (json.JSONDecodeError, ValueError):
            print("WARNING: LLM returned invalid JSON for plan — using safe default.")
            return {
                "search_stage": "exploration",
                "change_summary": "safe default (LLM parse failed)",
                "reasoning": "LLM returned invalid JSON",
                "conv_layers": [
                    {"filters": 32, "kernel_size": 3},
                    {"filters": 64, "kernel_size": 3},
                ],
                "dropout": 0.0,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 3,
            }

    def analyze_result(
        self,
        plan: Dict[str, Any],
        metrics: Dict[str, Any],
        history: List[Dict[str, Any]],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Asks the LLM to interpret the experiment result and suggest next steps.

        Returns a structured dict {outcome, likely_cause, next_step}.
        Falls back to a text-only dict if JSON parsing fails.
        """
        system = (
            "You are an ML researcher reviewing a BirdCLEF CNN experiment. "
            "Respond with ONLY a JSON object matching this schema — no markdown, no extra text:\n"
            + _ANALYSIS_SCHEMA
        )
        user = (
            f"Plan used: {json.dumps(plan, indent=2)}\n\n"
            f"Result: {json.dumps(metrics, indent=2)}\n\n"
            f"Number of previous runs: {len(history)}\n"
        )
        raw = self._call(system, user, temperature)
        try:
            return json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            return {"outcome": "unknown", "likely_cause": raw.strip(), "next_step": ""}
