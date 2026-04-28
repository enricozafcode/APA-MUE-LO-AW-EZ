from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationSummary:
    metrics: Dict[str, Any]
    analysis_prompt: str


class Evaluator:
    """Converts run outputs into structured feedback for the LLM."""

    # Patterns to extract cmAP from training stdout (order = priority)
    _CMAP_PATTERNS: List[str] = [
        r"cmAP[:\s=]+([0-9]+\.[0-9]+)",
        r"val_cmAP[:\s=]+([0-9]+\.[0-9]+)",
        r"validation cmAP[:\s=]+([0-9]+\.[0-9]+)",
        r"\"cmAP\"\s*:\s*([0-9]+\.[0-9]+)",
        r"mAP[:\s=]+([0-9]+\.[0-9]+)",
    ]

    def _parse_cmap(self, stdout: str) -> Optional[float]:
        """Extracts the last cmAP value found in stdout."""
        found = []
        for pattern in self._CMAP_PATTERNS:
            for match in re.finditer(pattern, stdout, re.IGNORECASE):
                found.append(float(match.group(1)))
        return found[-1] if found else None

    def _parse_json_metrics(self, stdout: str) -> Dict[str, Any]:
        """Tries to extract a JSON metrics block from stdout."""
        match = re.search(r"\{[^{}]*\"cmAP\"[^{}]*\}", stdout)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def build_summary(
        self,
        stdout: str,
        stderr: str,
        active_audio_strategies: List[str] | None = None,
        active_spec_strategies: List[str] | None = None,
    ) -> EvaluationSummary:
        status = "ok" if not stderr else "warning"
        cmap = self._parse_cmap(stdout)
        json_metrics = self._parse_json_metrics(stdout)

        metrics: Dict[str, Any] = {
            "status": status,
            "cmAP": cmap,
            **json_metrics,
        }

        active_audio = active_audio_strategies or []
        active_spec = active_spec_strategies or []

        cmap_str = f"{cmap:.4f}" if cmap is not None else "not found in output"

        analysis_prompt = (
            "You are an ML engineer optimizing a BirdCLEF audio classifier.\n\n"
            f"Last run status: {status}\n"
            f"cmAP score: {cmap_str}\n"
            f"Active audio augmentations: {active_audio if active_audio else 'none'}\n"
            f"Active spectrogram augmentations: {active_spec if active_spec else 'none'}\n\n"
            "Available augmentation strategies you can enable/disable:\n"
            "  Audio:       time_stretch, pitch_shift, noise_injection, time_shift\n"
            "  Spectrogram: time_mask, freq_mask, mixup, cutmix\n\n"
            "Based on the cmAP score, decide what to change in the next experiment:\n"
            "- If cmAP improved: keep current augmentations, consider adding one more\n"
            "- If cmAP dropped: disable the last added augmentation\n"
            "- If cmAP is not found: fix the training script to log cmAP clearly\n\n"
            "Propose the next experiment as a short Python training script that imports "
            "build_augmenters from augmentation and applies augmentation only to training data."
        )

        return EvaluationSummary(metrics=metrics, analysis_prompt=analysis_prompt)
