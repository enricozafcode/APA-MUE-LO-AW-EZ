"""
LLM-driven audio augmentation search for Perch embedding caches (meta-agent stage 1c).

The researcher proposes JSON augmentation configs compatible with AudioAugmenter + SNR mixing.
Configs are validated, used to rebuild embedding caches (same clip sample as stage 1a),
then the locked 1b head is trained on fixed head-train indices.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from augmentation import (
    AUDIO_EMBEDDING_BASELINES,
    describe_embedding_aug_compact,
    spec_to_embedding_aug,
    validate_embedding_aug,
)
from llm_client import LLMClient
from memory import ExperimentMemory

AUG_STRATEGIES = (
    "no_aug",
    "snr_only",
    "light_audio",
    "heavy_audio",
    "jungle_heavy_snr",
    "pitch_time_aggressive",
    "minimal_shift",
    "extreme_mix",
)

AUG_EXAMPLE_MEDIUM = {
    k: v
    for k, v in AUDIO_EMBEDDING_BASELINES["medium"].items()
    if k != "name"
}

AUG_RESEARCHER_SYSTEM_PROMPT = """You are an expert audio ML researcher designing augmentation for BirdCLEF jungle soundscape classification.

CONTEXT:
- Augmentation is applied to 5-second training clips BEFORE a frozen Perch ONNX encoder builds embeddings.
- Evaluation uses a FIXED classification head (already chosen); you only change how training audio is corrupted.
- Training clips are a stratified subset of the full dataset; head training uses a fixed 2000-row index subset of the cache.
- The competition has long 60s soundscapes with background noise, overlapping species, and rare birds.

AVAILABLE MECHANISMS (you must use only these — no new algorithm names):
1) SNR soundscape mixing (jungle background from train_soundscapes):
   - use_snr_mixing: bool
   - mix_prob: 0.0–1.0 (fraction of clips that get background mixed in)
   - snr_min_db, snr_max_db: SNR range in dB (lower = noisier background)

2) Per-strategy audio transforms (each under "audio" dict):
   - time_stretch: enabled, probability, rate_min, rate_max (e.g. 0.85–1.15)
   - pitch_shift: enabled, probability, steps_min, steps_max (semitones, e.g. -3 to 3)
   - noise_injection: enabled, probability, noise_level (e.g. 0.002–0.015)
   - time_shift: enabled, probability, shift_max_fraction (0.1–0.6)
   - gain_jitter: enabled, probability, min_db, max_db

EXPLORATION GOALS — cover a WIDE range across iterations:
- No augmentation vs SNR-only vs audio-only vs combined
- Light vs heavy jungle background (mix_prob, SNR range)
- Conservative vs aggressive time/pitch/gain
- Strategies suited to rare species (less destructive) vs robustness to noise (more mixing)
- Do NOT repeat configs similar to recent runs in the history

Respond with ONLY one JSON object (no markdown, no prose). Required keys:
- preset_name: short snake_case label (e.g. "heavy_jungle_snr")
- use_snr_mixing, mix_prob, snr_min_db, snr_max_db
- audio: object with the five strategy keys above (each with enabled + probability + params)
- strategy: "explore" or "exploit"
- reasoning: one sentence
- hypothesis: one sentence

Example:
{"preset_name": "jungle_heavy_snr", "use_snr_mixing": true, "mix_prob": 0.55, "snr_min_db": 0.0, "snr_max_db": 8.0, "audio": {"time_stretch": {"enabled": false}, "pitch_shift": {"enabled": false}, "noise_injection": {"enabled": true, "probability": 0.45, "noise_level": 0.006}, "time_shift": {"enabled": true, "probability": 0.5, "shift_max_fraction": 0.45}, "gain_jitter": {"enabled": true, "probability": 0.4, "min_db": -6.0, "max_db": 6.0}}, "strategy": "explore", "reasoning": "Strong soundscape SNR mix with mild audio jitter for jungle robustness.", "hypothesis": "Heavy background mixing improves soundscape AP without destroying species cues."}
""".strip()

AUG_REFINE_RESEARCHER_ADDENDUM = """
REFINE MODE — you are tuning ONE winning augmentation config from the explore phase.
- Start from the SEED CONFIG in the user message; make small, targeted changes (mix_prob, SNR bounds, one audio strategy).
- strategy MUST be "exploit".
- preset_name should be a new variant suffix e.g. "winner_tweak_mix_0.4".
- Goal: beat the seed macro AP on soundscape validation.
"""


def _extract_first_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _aug_safe_defaults() -> dict:
    base = dict(AUG_EXAMPLE_MEDIUM)
    base["preset_name"] = "fallback_medium"
    base["strategy"] = "explore"
    base["reasoning"] = "Parser fallback — medium baseline."
    base["hypothesis"] = "Safe default augmentation."
    return base


def _parse_aug_spec(response: str) -> dict:
    cleaned = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    spec = _extract_first_json_object(cleaned)
    if spec is not None:
        return spec
    print("  [AugResearcher] Warning: could not parse JSON — using medium baseline.")
    return _aug_safe_defaults()


class AugResearcher:
    def __init__(
        self,
        llm: LLMClient,
        memory: ExperimentMemory,
        temperature: float = 0.65,
        *,
        refine_mode: bool = False,
        seed_aug: dict | None = None,
        seed_score: float | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.seed_aug = seed_aug or {}
        self.seed_score = seed_score

    def next_experiment(self) -> dict:
        history = self.memory.researcher_context()
        best = self.memory.best_runs(1)
        total = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"

        examples = "\n".join(
            f"  - {name}: {describe_embedding_aug_compact(name)}"
            for name in ("light", "medium", "high")
        )
        strategy_menu = ", ".join(AUG_STRATEGIES)

        if self.refine_mode:
            seed_line = ""
            if self.seed_score is not None:
                seed_line = (
                    f"\nEXPLORE PHASE CHAMPION TO BEAT ({self.memory.ranking_metric}): "
                    f"{float(self.seed_score):.5f}\n"
                )
            if self.seed_aug:
                seed_line += f"Seed augmentation config:\n{json.dumps(self.seed_aug, indent=2)}\n"

            user_prompt = (
                f"{history}\n\n"
                f"AUGMENTATION REFINE — tune the winning explore config.\n"
                f"{seed_line}\n"
                f"Total refine experiments: {total}\n"
                f"Best refine run so far: {best_str}\n\n"
                f"Make small targeted changes to SNR mix and audio strategy probabilities.\n"
                f"strategy MUST be exploit. Respond with ONLY JSON.\n"
            )
            system_prompt = AUG_RESEARCHER_SYSTEM_PROMPT + "\n\n" + AUG_REFINE_RESEARCHER_ADDENDUM
        else:
            user_prompt = (
                f"{history}\n\n"
                f"BASELINE REFERENCE (do not copy blindly — explore beyond these):\n{examples}\n\n"
                f"Strategy families to explore across iterations: {strategy_menu}\n\n"
                f"Total experiments: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                f"Propose a NEW augmentation config different from recent runs.\n"
                f"Cover the full spectrum: no_aug, snr-only, audio-only, light, heavy, extreme.\n"
                f"Respond with ONLY JSON.\n"
            )
            system_prompt = AUG_RESEARCHER_SYSTEM_PROMPT

        print(f"\n  [AugResearcher] Iteration context: {total} prior runs, best {best_str}")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        raw = _parse_aug_spec(response)
        aug_dict, meta = validate_embedding_aug(raw)
        spec = {**meta, **aug_dict}
        if self.refine_mode:
            spec["strategy"] = "exploit"
        print(
            f"  [AugResearcher] preset={spec.get('preset_name')} "
            f"snr={spec.get('use_snr_mixing')} mix_p={spec.get('mix_prob')} "
            f"strategy={spec.get('strategy')}"
        )
        print(f"  [AugResearcher] {spec.get('reasoning', '')[:120]}")
        return spec


def slug_from_spec(spec: dict, iteration: int, phase: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", str(spec.get("preset_name", "custom")).lower())
    name = name.strip("_")[:40] or "custom"
    return f"{phase}_{iteration:03d}_{name}"


def aug_dict_from_logged_spec(spec: dict) -> dict:
    """Strip researcher metadata; return embedding aug dict for cache build."""
    return spec_to_embedding_aug(spec)
