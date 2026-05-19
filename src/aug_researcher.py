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
    _AUDIO_STRATEGIES,
    describe_embedding_aug_compact,
    spec_to_embedding_aug,
    validate_embedding_aug,
)
from llm_client import LLMClient, llm_response_failed
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

# Shorter prompt for small/fast models (gemma, llama3, etc.) — still valid JSON output.
AUG_RESEARCHER_COMPACT_PROMPT = """You design BirdCLEF training audio augmentation (5s clips → frozen Perch embeddings).

Output ONLY one JSON object. Start with { and end with }. No markdown, no explanation outside JSON.

Top-level keys (required — NOT inside audio):
- preset_name, use_snr_mixing, mix_prob, snr_min_db, snr_max_db
- strategy ("explore" or "exploit"), reasoning, hypothesis (one short sentence each)
- audio: ONLY the five strategy objects below

audio must contain exactly: time_stretch, pitch_shift, noise_injection, time_shift, gain_jitter
Each strategy: {"enabled": bool, "probability": 0-1, plus params when enabled}

Keep JSON compact so it is not cut off. Beat the best score in history when possible.
""".strip()

AUG_REFINE_RESEARCHER_ADDENDUM = """
REFINE MODE — you are tuning ONE winning augmentation config from the explore phase.
- Start from the SEED CONFIG in the user message; make small, targeted changes (mix_prob, SNR bounds, one audio strategy).
- strategy MUST be "exploit".
- preset_name should be a new variant suffix e.g. "winner_tweak_mix_0.4".
- Goal: beat the seed macro AP on soundscape validation.
"""

AUG_MEDIUM_HIGH_INTENSITY_ADDENDUM = """
INTENSITY FLOOR (every experiment in the batch):
- Target medium-to-high strength only — comparable to the "medium" or "high" embedding baselines.
- Do NOT propose light/minimal/no_aug configs (mix_prob < 0.25, most audio strategies off).
- use_snr_mixing: true with mix_prob in [0.28, 0.65] typical; snr_max_db at least 5.
- Enable at least two audio strategies with probability >= 0.35 when enabled.
- Custom preset_name (not just "light"); vary jungle SNR vs audio-heavy ideas across the three slots.
"""

AUG_BATCH_PLANNER_ADDENDUM = """
BATCH PLANNING — return exactly {batch_size} augmentation experiments in ONE JSON object:
{{
  "planner_note": "optional one short sentence",
  "experiments": [
    {{ "slot": "a1", ...all aug keys... }},
    {{ "slot": "a2", ... }},
    {{ "slot": "a3", ... }}
  ]
}}

Slot roles: three different medium/high custom configs (not copies of light baseline).
Output {{"planner_note":"...", "experiments":[...]}} OR a JSON array of {batch_size} objects.
Keep each experiment JSON compact (short reasoning/hypothesis).
"""

AUG_BATCH_SLOTS = ("a1", "a2", "a3")


def _extract_first_json_object(text: str) -> dict | None:
    """Parse first {...} object; repair with closing braces if the model truncated."""
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:]
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(fragment):
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
                    return json.loads(fragment[: i + 1])
                except json.JSONDecodeError:
                    return None
    if depth <= 0:
        return None
    repaired = fragment.rstrip().rstrip(",")
    repaired += "}" * depth
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _sanitize_parsed_aug(raw: dict) -> dict:
    """Move misplaced meta keys out of audio; keep only strategy blocks inside audio."""
    parsed = dict(raw)
    audio_in = parsed.get("audio")
    if not isinstance(audio_in, dict):
        return parsed
    audio = dict(audio_in)
    for key in list(audio.keys()):
        if key in _AUDIO_STRATEGIES:
            continue
        if key in ("reasoning", "hypothesis", "strategy", "preset_name"):
            parsed.setdefault(key, audio.pop(key))
        elif key == "description" and "reasoning" not in parsed:
            parsed["reasoning"] = audio.pop(key)
        else:
            audio.pop(key, None)
    parsed["audio"] = audio
    return parsed


def _aug_safe_defaults(*, reason: str = "Parser fallback — medium baseline.") -> dict:
    base = dict(AUG_EXAMPLE_MEDIUM)
    base["preset_name"] = "fallback_medium"
    base["strategy"] = "explore"
    base["reasoning"] = reason
    base["hypothesis"] = ""
    return base


def _aug_medium_merge_base() -> dict:
    """Fill missing SNR/audio fields when merging partial LLM JSON (no preset_name)."""
    out = dict(AUG_EXAMPLE_MEDIUM)
    out["strategy"] = "explore"
    out["reasoning"] = ""
    out["hypothesis"] = ""
    return out


_GENERIC_PRESET_NAMES = frozenset(
    {"", "llm_custom", "custom", "fallback_medium", "fallback", "medium", "high", "light"}
)


def _assign_slot_preset_name(raw: dict, *, slot: str, index: int = 1) -> dict:
    """Give each batch slot a distinct name when the model omits preset_name."""
    out = dict(raw)
    name = str(out.get("preset_name", "")).strip().lower()
    if name in _GENERIC_PRESET_NAMES or name.startswith("fallback_"):
        slug = re.sub(r"[^a-z0-9_]+", "_", str(slot or f"a{index}").lower()).strip("_")
        out["preset_name"] = f"llm_{slug or f'a{index}'}"
    return out


def _clean_aug_llm_response(response: str) -> str:
    _open, _close = "<" + "think" + ">", "</" + "think" + ">"
    cleaned = re.sub(
        re.escape(_open) + r"[\s\S]*?" + re.escape(_close),
        "",
        response,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _strip_prose_before_json(text: str) -> str:
    """Drop leading markdown / reasoning so json.loads sees the payload."""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    positions = [text.find(c) for c in ("{", "[") if text.find(c) >= 0]
    if not positions:
        return text.strip()
    return text[min(positions) :].strip()


def _merge_partial_aug_spec(parsed: dict) -> dict:
    """Fill missing audio / SNR fields from medium baseline."""
    parsed = _sanitize_parsed_aug(parsed)
    out = _aug_medium_merge_base()
    for key in (
        "preset_name",
        "use_snr_mixing",
        "mix_prob",
        "snr_min_db",
        "snr_max_db",
        "strategy",
        "reasoning",
        "hypothesis",
    ):
        if key in parsed and parsed[key] is not None:
            out[key] = parsed[key]
    audio_in = parsed.get("audio")
    if isinstance(audio_in, dict):
        audio_out = dict(out.get("audio") or {})
        for strat in _AUDIO_STRATEGIES:
            if strat not in audio_out:
                audio_out[strat] = {"enabled": False}
            if strat in audio_in and isinstance(audio_in[strat], dict):
                merged = dict(audio_out.get(strat) or {})
                merged.update(audio_in[strat])
                audio_out[strat] = merged
        out["audio"] = audio_out
    return out


def _parse_aug_spec(response: str) -> dict:
    if llm_response_failed(response):
        detail = str(response or "").strip()[:240]
        print("  [AugResearcher] LLM call failed — using medium baseline.")
        if detail:
            print(f"  [AugResearcher] Reason: {detail}")
        return _aug_safe_defaults(reason=detail or "LLM unavailable.")

    cleaned = _clean_aug_llm_response(response)

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    candidates: list[dict] = []
    try:
        root = json.loads(cleaned)
        if isinstance(root, dict):
            candidates.append(root)
    except json.JSONDecodeError:
        pass

    obj = _extract_first_json_object(cleaned)
    if obj is not None:
        candidates.append(obj)

    seen: set[int] = set()
    unique: list[dict] = []
    for raw in candidates:
        key = id(raw)
        if key in seen:
            continue
        seen.add(key)
        unique.append(raw)

    last_err: ValueError | TypeError | None = None
    for raw in unique:
        merged = _assign_slot_preset_name(_merge_partial_aug_spec(raw), slot="a1", index=1)
        try:
            validate_embedding_aug(merged)
            preset = str(merged.get("preset_name", "?"))
            if not preset.startswith("fallback_"):
                print(f"  [AugResearcher] Parsed preset={preset}")
            return _finalize_aug_spec(merged)
        except (ValueError, TypeError) as exc:
            last_err = exc
            continue

    print("  [AugResearcher] Warning: could not parse JSON — using medium baseline.")
    if last_err is not None:
        print(f"  [AugResearcher] Validation error: {last_err}")
    print(f"  [AugResearcher] Raw response (first 500 chars): {repr(cleaned[:500])}")
    return _finalize_aug_spec(_aug_safe_defaults())


def _is_valid_aug_spec_dict(spec: dict) -> bool:
    try:
        validate_embedding_aug(_merge_partial_aug_spec(spec))
        return True
    except (ValueError, TypeError):
        return False


def _enforce_medium_high_aug(spec: dict) -> dict:
    """Clamp parsed specs to medium/high intensity (no light exploration)."""
    merged = _merge_partial_aug_spec(spec)
    merged["use_snr_mixing"] = True
    merged["mix_prob"] = max(0.28, min(0.70, float(merged.get("mix_prob", 0.40))))
    merged["snr_min_db"] = float(max(-5.0, min(10.0, float(merged.get("snr_min_db", 0.0)))))
    merged["snr_max_db"] = float(max(5.0, min(22.0, float(merged.get("snr_max_db", 15.0)))))
    if merged["snr_min_db"] > merged["snr_max_db"]:
        merged["snr_min_db"], merged["snr_max_db"] = merged["snr_max_db"], merged["snr_min_db"]

    audio = dict(merged.get("audio") or {})
    enabled = 0
    for strat in _AUDIO_STRATEGIES:
        block = dict(audio.get(strat) or {})
        if block.get("enabled"):
            enabled += 1
            block["probability"] = max(0.35, min(0.85, float(block.get("probability", 0.45))))
            audio[strat] = block
        else:
            audio[strat] = block
    if enabled < 2:
        for strat in ("noise_injection", "time_shift", "gain_jitter"):
            if enabled >= 2:
                break
            block = dict(audio.get(strat) or {})
            block["enabled"] = True
            block["probability"] = max(0.40, float(block.get("probability", 0.45)))
            audio[strat] = block
            enabled += 1
    merged["audio"] = audio
    name = str(merged.get("preset_name", "custom")).lower()
    if any(x in name for x in ("light", "minimal", "no_aug", "none")):
        merged["preset_name"] = re.sub(
            r"light|minimal|no_aug", "medhigh", name, flags=re.IGNORECASE
        ).strip("_") or "medhigh_custom"
    return merged


def _finalize_aug_spec(raw: dict, *, slot: str = "") -> dict:
    spec = _enforce_medium_high_aug(raw)
    if slot:
        spec["slot"] = slot
    aug_dict, meta = validate_embedding_aug(spec)
    return {**meta, **aug_dict}


def _parse_aug_batch_response(response: str, batch_size: int, *, quiet: bool = False) -> list[dict]:
    if llm_response_failed(response):
        if not quiet:
            detail = str(response or "").strip()[:240]
            print("  [AugResearcher] LLM call failed — batch fallbacks (medium/high).")
            if detail:
                print(f"  [AugResearcher] Reason: {detail}")
        return _aug_batch_fallback(batch_size)

    cleaned = _strip_prose_before_json(_clean_aug_llm_response(response))

    specs: list[dict] = []
    root: dict | list | None = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            root = {"experiments": parsed}
        elif isinstance(parsed, dict):
            root = parsed
    except json.JSONDecodeError:
        obj = _extract_first_json_object(cleaned)
        if obj is not None:
            root = obj

    planner_note = ""
    if isinstance(root, dict):
        planner_note = str(root.get("planner_note") or root.get("batch_note") or "")[:200]
        raw_list = root.get("experiments")
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict) and _is_valid_aug_spec_dict(item):
                    specs.append(item)
    valid = [_enforce_medium_high_aug(s) for s in specs if isinstance(s, dict)]
    valid = [s for s in valid if _is_valid_aug_spec_dict(s)]
    if not valid:
        if not quiet:
            print("  [AugResearcher] Warning: could not parse batch JSON — slot fallbacks.")
            print(f"  [AugResearcher] Raw response (first 500 chars): {repr(cleaned[:500])}")
        return _aug_batch_fallback(batch_size)

    out: list[dict] = []
    for i, raw in enumerate(valid[:batch_size]):
        slot = str(raw.get("slot") or (AUG_BATCH_SLOTS[i] if i < len(AUG_BATCH_SLOTS) else f"a{i+1}"))
        out.append(_finalize_aug_spec(_assign_slot_preset_name(raw, slot=slot, index=i + 1), slot=slot))
    while len(out) < batch_size:
        out.extend(_aug_batch_fallback(batch_size - len(out)))
    out = out[:batch_size]
    if planner_note and out:
        out[0]["_planner_note"] = planner_note
    import run_log

    run_log.apply_planner_rationale_fallback(out, planner_note)
    return out


def _aug_batch_fallback(batch_size: int) -> list[dict]:
    """Three diverse medium/high templates when batch JSON fails."""
    bases = [
        dict(AUDIO_EMBEDDING_BASELINES["medium"]),
        dict(AUDIO_EMBEDDING_BASELINES["high"]),
    ]
    high = dict(AUDIO_EMBEDDING_BASELINES["high"])
    high["mix_prob"] = 0.55
    templates = [
        ("a1", bases[1], "medhigh_snr_audio", "Medium+SNR and audio blend."),
        ("a2", high, "heavy_jungle_mix", "High jungle mix_prob."),
        ("a3", bases[0], "medium_plus_shift", "Medium with stronger time_shift."),
    ]
    out: list[dict] = []
    for i in range(batch_size):
        slot, base, pname, note = templates[i % len(templates)]
        raw = dict(base)
        raw["preset_name"] = pname
        raw["strategy"] = "explore"
        raw["reasoning"] = "Batch fallback — " + note
        raw["hypothesis"] = note
        out.append(_finalize_aug_spec(raw, slot=slot))
    return out


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
        compact_prompt: bool = True,
        format_json: bool = True,
        num_predict: int | None = 4096,
        batch_size: int = 1,
        quiet: bool = False,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.seed_aug = seed_aug or {}
        self.seed_score = seed_score
        self.compact_prompt = compact_prompt
        self.format_json = format_json
        self.num_predict = num_predict
        self.batch_size = max(1, int(batch_size))
        self.quiet = bool(quiet)

    def next_experiment(self) -> dict:
        return self.next_experiments()[0]

    def next_experiments(self) -> list[dict]:
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
                    f"\nCHAMPION TO BEAT ({self.memory.ranking_metric}): "
                    f"{float(self.seed_score):.5f}\n"
                )
            if self.seed_aug:
                seed_line += f"Seed augmentation config:\n{json.dumps(self.seed_aug, indent=2)}\n"

            user_prompt = (
                f"{history}\n\n"
                f"AUGMENTATION TUNING — improve the champion config.\n"
                f"{seed_line}\n"
                f"Total experiments: {total}\n"
                f"Best so far: {best_str}\n\n"
                f"Make targeted changes to SNR mix and audio strategy probabilities.\n"
                f"strategy MUST be exploit. Respond with ONLY JSON.\n"
            )
            system_prompt = (
                (AUG_RESEARCHER_COMPACT_PROMPT if self.compact_prompt else AUG_RESEARCHER_SYSTEM_PROMPT)
                + "\n\n"
                + AUG_REFINE_RESEARCHER_ADDENDUM
            )
        batch_size = self.batch_size
        if batch_size > 1 and not self.refine_mode:
            user_prompt = (
                f"{history}\n\n"
                f"REFERENCE baselines (intensity guide — invent custom medium/high variants):\n"
                f"  medium: {describe_embedding_aug_compact('medium')}\n"
                f"  high:   {describe_embedding_aug_compact('high')}\n\n"
                f"Total experiments logged: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                f"Propose exactly {batch_size} custom augmentation configs for this round "
                f"(slots {', '.join(AUG_BATCH_SLOTS[:batch_size])}).\n"
                f"All must be medium-to-high strength. Decide quickly — short JSON only.\n"
            )
            system_prompt = (
                (AUG_RESEARCHER_COMPACT_PROMPT if self.compact_prompt else AUG_RESEARCHER_SYSTEM_PROMPT)
                + "\n\n"
                + AUG_MEDIUM_HIGH_INTENSITY_ADDENDUM
                + "\n\n"
                + AUG_BATCH_PLANNER_ADDENDUM.format(batch_size=batch_size)
            )
        else:
            user_prompt = (
                f"{history}\n\n"
                f"BASELINE REFERENCE (medium / high intensity guide):\n{examples}\n\n"
                f"Strategy families: {strategy_menu}\n\n"
                f"Total experiments: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                f"Propose ONE medium-to-high augmentation config. Short JSON only.\n"
                f"Respond with ONLY JSON — no markdown fences.\n"
            )
            system_prompt = (
                (AUG_RESEARCHER_COMPACT_PROMPT if self.compact_prompt else AUG_RESEARCHER_SYSTEM_PROMPT)
                + "\n\n"
                + AUG_MEDIUM_HIGH_INTENSITY_ADDENDUM
            )

        if not self.quiet:
            print(f"\n  [AugResearcher] Iteration context: {total} prior runs, best {best_str}")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            format_json=self.format_json,
            num_predict=self.num_predict,
        )
        if batch_size > 1 and not self.refine_mode:
            specs = _parse_aug_batch_response(response, batch_size, quiet=self.quiet)
        else:
            specs = [_parse_aug_spec(response)]

        for spec in specs:
            if self.refine_mode:
                spec["strategy"] = "exploit"

        import run_log

        run_log.print_researcher_proposals(
            specs,
            track="perch",
            round_label="aug planner",
        )

        if self.quiet and specs:
            names = ", ".join(str(s.get("preset_name", "?")) for s in specs[: self.batch_size])
            line = f"    → {names}"
            if llm_response_failed(response):
                line += "  (LLM error — using fallbacks)"
            elif any(str(s.get("preset_name", "")).startswith("fallback_") for s in specs):
                line += "  (parse failed — medium baseline)"
            elif len({str(s.get("preset_name")) for s in specs}) < len(specs):
                line += "  (duplicate names — check LLM JSON)"
            print(line)
        return specs


def slug_from_spec(spec: dict, iteration: int, phase: str, slot: str = "") -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", str(spec.get("preset_name", "custom")).lower())
    name = name.strip("_")[:36] or "custom"
    slot_s = re.sub(r"[^a-z0-9]+", "", str(slot or spec.get("slot", "")).lower())[:6]
    if slot_s:
        return f"{phase}_r{iteration:02d}_{slot_s}_{name}"
    return f"{phase}_{iteration:03d}_{name}"


def aug_dict_from_logged_spec(spec: dict) -> dict:
    """Strip researcher metadata; return embedding aug dict for cache build."""
    return spec_to_embedding_aug(spec)
