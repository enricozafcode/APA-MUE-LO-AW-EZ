"""
Researcher — the outer loop reasoning agent.

The Researcher reads the full experiment history and decides what to try next.
It outputs a compact experiment spec (JSON) that the Coder turns into code.

The Researcher NEVER writes code. The Coder NEVER reads history.
This keeps each model focused on what it does best.
"""

from __future__ import annotations

import json
import re

from memory import ExperimentMemory


# Default search space the Researcher can pick from
SEARCH_SPACE = {
    "depth":                    [1, 2, 3, 4, 6, 8, 12],
    "filters_base":             [16, 32, 64, 128],
    "learning_rate":            [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "weight_decay":             [0.0, 1e-4, 1e-3, 1e-2],
    "dropout":                  [0.0, 0.2, 0.3, 0.5],
    "batch_size":               [16, 32, 64],
    "optimizer":                ["adam", "sgd_momentum"],
    "n_mels":                   [64, 128],
    "n_frames":                 [128, 256],
    "aug_prob":                 [0.0, 0.3, 0.5, 0.7],
    "aug_noise_std":            [0.0, 0.003, 0.007, 0.015],
    "aug_time_mask":            [0, 8, 16, 24],
    "aug_freq_mask":            [0, 4, 8, 12],
    "classifier_hidden_units":  [0, 128, 256, 512],
    "residuals":                [True, False],
    "batch_norm":               [True, False],
}

RESEARCHER_SYSTEM_PROMPT = """You are an expert ML researcher optimizing a BirdCLEF audio classification model.
Your job is to analyze past experiment results and decide what configuration to try next.

You reason carefully about:
- Which hyperparameters improved performance and by how much
- Which configurations failed and why
- What has NOT been tried yet that might work
- Whether to exploit (tweak the best) or explore (try something different)

You output ONLY a JSON object — no code, no explanations outside the JSON.

The JSON must contain:
- All hyperparameter values from the search space
- "reasoning": your analysis of past results (2-3 sentences)
- "hypothesis": why you think this config will improve on the current best (1-2 sentences)
- "strategy": one of "exploit", "explore", or "fix_failure"

Return ONLY valid JSON, nothing else."""


class Researcher:
    """
    Outer loop: reads memory, reasons, produces next experiment spec.
    Uses a reasoning model (e.g. deepseek-r1).
    """

    def __init__(self, llm, memory: ExperimentMemory, temperature: float = 0.6) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature

    def next_experiment(self) -> dict:
        """
        Reads full history, reasons about what to try, returns experiment spec.
        This is the only place memory context is used — never in code generation.
        """
        history = self.memory.researcher_context()
        best = self.memory.best_runs(1)
        best_auc = best[0]["macro_roc_auc"] if best else None
        total = self.memory.total()

        user_prompt = (
            f"{history}\n\n"
            f"Search space available:\n{json.dumps(SEARCH_SPACE, indent=2)}\n\n"
            f"Total experiments so far: {total}\n"
            f"Current best AUC: {f'{best_auc:.5f}' if best_auc else 'none yet'}\n\n"
            "Decide the next experiment. Return a JSON object with all hyperparameter "
            "values from the search space plus 'reasoning', 'hypothesis', and 'strategy'.\n"
            "Pick values that are meaningfully different from recent failures "
            "and build on what has worked."
        )

        best_auc_str = f"{best_auc:.5f}" if best_auc is not None else "none"
        print(f"\n  [Researcher] Analyzing {total} experiments, best AUC={best_auc_str}...")

        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )

        spec = self._parse_spec(response)

        print(f"  [Researcher] Strategy: {spec.get('strategy', '?')} | "
              f"depth={spec.get('depth')} lr={spec.get('learning_rate')} "
              f"filters={spec.get('filters_base')}")
        print(f"  [Researcher] Reasoning: {spec.get('reasoning', '')[:120]}")

        return spec

    def _parse_spec(self, response: str) -> dict:
        """Extract JSON from researcher response, fall back to defaults on failure."""
        # Try to extract JSON block
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                spec = json.loads(match.group())
                # Ensure required keys exist with defaults
                return _fill_defaults(spec)
            except json.JSONDecodeError:
                pass

        print("  [Researcher] Warning: could not parse JSON response, using safe defaults.")
        return _safe_defaults()


def _fill_defaults(spec: dict) -> dict:
    """Fill missing keys with safe default values."""
    defaults = _safe_defaults()
    for k, v in defaults.items():
        spec.setdefault(k, v)
    return spec


def _safe_defaults() -> dict:
    return {
        "depth": 3,
        "filters_base": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer": "adam",
        "n_mels": 64,
        "n_frames": 128,
        "aug_prob": 0.3,
        "aug_noise_std": 0.005,
        "aug_time_mask": 16,
        "aug_freq_mask": 8,
        "classifier_hidden_units": 256,
        "residuals": False,
        "batch_norm": True,
        "reasoning": "Fallback defaults — researcher output could not be parsed.",
        "hypothesis": "Baseline config.",
        "strategy": "explore",
    }
