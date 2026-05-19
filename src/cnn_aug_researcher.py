"""
LLM augmentation search for CNN stage 1c (locked architecture from 1b).

Two augmentation layers (both tunable):
  1) Audio + jungle SNR mixing before log-mel (baked into focal_train_*.npz cache)
  2) Spectrogram noise / time-freq masking at train time (same clips, different masks)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from augmentation import (
    BASELINE_AUG_NAMES,
    get_audio_embedding_aug,
    get_cnn_spectrogram_aug,
    register_custom_embedding_aug,
    validate_embedding_aug,
)
from aug_researcher import (
    AUG_BATCH_SLOTS,
    _assign_slot_preset_name,
    _clean_aug_llm_response,
    _extract_first_json_object,
    _merge_partial_aug_spec,
    _sanitize_parsed_aug,
    _strip_prose_before_json,
)
from llm_client import LLMClient, llm_response_failed
from memory import ExperimentMemory

CNN_SPEC_KEYS = ("aug_prob", "aug_noise_std", "aug_time_mask", "aug_freq_mask")

# Hard ranges enforced before a spec ever reaches the training subprocess.
# Out-of-range values from the LLM (e.g. aug_prob=999) used to propagate
# unchanged into get_training_config(), silently degrading training.
_CNN_SPEC_RANGES: dict[str, tuple[float, float]] = {
    "aug_prob": (0.0, 1.0),
    "aug_noise_std": (0.0, 0.05),
    "aug_time_mask": (0, 64),
    "aug_freq_mask": (0, 32),
}

CNN_AUG_RESEARCHER_SYSTEM = """You tune BirdCLEF CNN augmentation. Architecture is LOCKED. \
Optimize macro_average_precision on train_soundscapes.

Each experiment sets two layers:
- AUDIO/SNR: audio_anchor in {light,medium,high}; use_snr_mixing, mix_prob[0..1], \
snr_min_db[-5..15], snr_max_db[5..25]
- SPECTROGRAM: aug_prob[0..1], aug_noise_std[0..0.03], aug_time_mask[0..48], aug_freq_mask[0..24]

Baselines: light(p=0.5,n=0.004,tm=12,fm=6) medium(p=0.75,n=0.007,tm=16,fm=8) \
high(p=1.0,n=0.012,tm=24,fm=12).

Explore CONTRASTS: no_aug, audio-heavy, spec-heavy, both-heavy. \
preset_name: unique snake_case. Output ONLY JSON."""

CNN_AUG_BATCH_ADDENDUM = """
Return ONE JSON object with EXACTLY this shape (no prose, no markdown):
{{
  "planner_note": "<one short sentence>",
  "experiments": [
    {{
      "slot": "a1",
      "preset_name": "<unique snake_case>",
      "audio_anchor": "medium",
      "use_snr_mixing": true,
      "mix_prob": 0.4,
      "snr_min_db": 0.0,
      "snr_max_db": 15.0,
      "aug_prob": 0.75,
      "aug_noise_std": 0.007,
      "aug_time_mask": 16,
      "aug_freq_mask": 8,
      "strategy": "explore",
      "reasoning": "<1 sentence>",
      "hypothesis": "<1 sentence>"
    }}
  ]
}}

Rules:
- experiments must be a LIST of length {batch_size}.
- Slot names in order: {slot_list}.
- Each slot must differ from the others on at least 2 numeric fields.
- The `audio` field is OPTIONAL; omit it unless you really want to override defaults.
"""

CNN_AUG_REFINE_ADDENDUM = """
REFINE mode: return ONE experiment dict (no wrapper, no list) with strategy="exploit", \
small numeric tweaks to SEED CONFIG, and a new preset_name suffix.
"""


def _slug_preset_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    return (s[:48] or "cnn_custom")


def _clamp_cnn_spec_value(key: str, value: Any) -> float | int:
    lo, hi = _CNN_SPEC_RANGES[key]
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(lo)
    v = max(lo, min(hi, v))
    return int(round(v)) if key in ("aug_time_mask", "aug_freq_mask") else float(v)


def _merge_cnn_spec_fields(out: dict, spec: dict, anchor: str) -> None:
    base = get_cnn_spectrogram_aug(anchor if anchor in BASELINE_AUG_NAMES else "medium")
    for k in CNN_SPEC_KEYS:
        if k in spec and spec[k] is not None:
            out[k] = _clamp_cnn_spec_value(k, spec[k])
        else:
            out[k] = _clamp_cnn_spec_value(k, base[k])


def spec_to_cnn_training_aug(
    spec: dict,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Build get_training_config() overrides and register custom embedding aug for focal cache.
    Returns (aug_dict, cache_preset_name).
    """
    preset = _slug_preset_name(spec.get("preset_name", "cnn_custom"))
    anchor = str(spec.get("audio_anchor", "medium")).strip().lower()
    if anchor not in BASELINE_AUG_NAMES:
        anchor = "medium"

    raw = dict(spec)
    raw.setdefault("preset_name", preset)
    raw["audio"] = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    merged = _merge_partial_aug_spec(_assign_slot_preset_name(raw, slot=spec.get("slot", "a1")))

    base_embed = get_audio_embedding_aug(anchor)
    for key in ("use_snr_mixing", "mix_prob", "snr_min_db", "snr_max_db", "audio"):
        if key in spec and spec[key] is not None:
            base_embed[key] = spec[key]
        elif key in merged and merged[key] is not None:
            base_embed[key] = merged[key]

    embed_payload = {
        "preset_name": preset,
        "strategy": str(spec.get("strategy", merged.get("strategy", "explore"))),
        "reasoning": str(spec.get("reasoning", merged.get("reasoning", ""))),
        "hypothesis": str(spec.get("hypothesis", merged.get("hypothesis", ""))),
        **base_embed,
    }
    embed_aug, _meta = validate_embedding_aug(embed_payload)

    if cache_dir is not None:
        register_custom_embedding_aug(Path(cache_dir), preset, embed_aug)

    train_aug: dict[str, Any] = {"aug_preset": preset}
    _merge_cnn_spec_fields(train_aug, spec, anchor)
    return train_aug, preset


def _is_valid_cnn_aug_spec(raw: dict) -> bool:
    try:
        spec_to_cnn_training_aug(raw, cache_dir=None)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def _cnn_aug_fallback(
    *,
    slot: str = "a1",
    index: int = 1,
    round_i: int = 0,
    history_count: int = 0,
) -> dict:
    """
    Diverse parser fallback. Without a `round_i` (or `history_count`) signal,
    repeated fallback calls used to produce the same three preset_names
    (`fallback_high_a1`, `fallback_medium_a2`, `fallback_light_a3`) which
    collide on the focal cache key — so every "new" experiment was
    re-running the same training config. Vary the preset (and slightly
    perturb the spectrogram knobs) by the round/history seed so the
    researcher actually explores when the LLM keeps failing.
    """
    seed = max(0, int(round_i) * 7 + int(history_count) + int(index))
    anchor = ("high", "medium", "light")[seed % 3]
    base_spec = get_cnn_spectrogram_aug(anchor)
    # rotate masks / noise a bit so the focal cache key differs across rounds
    masks = (8, 16, 24, 32, 12, 20)
    noises = (0.003, 0.007, 0.012, 0.018, 0.005, 0.010)
    base_spec = dict(base_spec)
    base_spec["aug_time_mask"] = int(masks[seed % len(masks)])
    base_spec["aug_freq_mask"] = max(2, base_spec["aug_time_mask"] // 2)
    base_spec["aug_noise_std"] = float(noises[seed % len(noises)])
    base_spec["aug_prob"] = float(min(1.0, 0.4 + 0.15 * ((seed % 5) + 1)))
    pname = f"fallback_r{int(round_i):02d}_{anchor}_{slot}"
    return {
        "slot": slot,
        "preset_name": pname,
        "audio_anchor": anchor,
        "strategy": "explore",
        "reasoning": "Parser fallback — varied CNN baseline (seed-based).",
        "hypothesis": f"Use {anchor} audio + perturbed spec baseline (seed={seed}).",
        **get_audio_embedding_aug(anchor),
        **base_spec,
    }


def _looks_like_experiment(d: dict) -> bool:
    """Heuristic: does this dict look like a single aug experiment?"""
    if not isinstance(d, dict):
        return False
    hints = ("audio_anchor", "aug_prob", "mix_prob", "use_snr_mixing", "snr_min_db", "audio")
    return sum(1 for h in hints if h in d) >= 2


def _coerce_experiments_list(root: Any) -> list[dict]:
    """
    Pull experiment dicts out of *any* common LLM shape:
      - {"experiments": [...]}        ← canonical
      - [{...}, {...}, ...]            ← bare list
      - {"a1": {...}, "a2": {...}}     ← slots as keys
      - {"slot": "a1", ...}            ← single experiment at top level
      - {"experiments": {"a1": {...}}} ← experiments as dict-of-slots
    """
    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)]
    if not isinstance(root, dict):
        return []
    raw_list = root.get("experiments")
    if isinstance(raw_list, list):
        out = [x for x in raw_list if isinstance(x, dict)]
        if out:
            return out
    if isinstance(raw_list, dict):
        out = [v for v in raw_list.values() if isinstance(v, dict)]
        if out:
            return out
    # slots-as-keys at root
    slot_like = [v for k, v in root.items()
                 if isinstance(v, dict) and (re.fullmatch(r"a\d+", str(k).lower()) or _looks_like_experiment(v))]
    if slot_like:
        return slot_like
    # single experiment at root
    if _looks_like_experiment(root):
        return [root]
    return []


def _parse_cnn_aug_spec(response: str, *, round_i: int = 0, history_count: int = 0) -> dict:
    if llm_response_failed(response):
        print("  [CnnAugResearcher] LLM call failed — using fallback.")
        return _cnn_aug_fallback(round_i=round_i, history_count=history_count)

    cleaned = _strip_prose_before_json(_clean_aug_llm_response(response))
    candidates: list[dict] = []
    try:
        root = json.loads(cleaned)
        candidates.extend(_coerce_experiments_list(root))
    except json.JSONDecodeError:
        pass
    if not candidates:
        obj = _extract_first_json_object(cleaned)
        if obj is not None:
            candidates.extend(_coerce_experiments_list(obj))

    for i, raw in enumerate(candidates):
        merged = _sanitize_parsed_aug(_assign_slot_preset_name(raw, slot="a1", index=i + 1))
        if "audio_anchor" not in merged:
            merged["audio_anchor"] = "medium"
        for k in CNN_SPEC_KEYS:
            if k not in merged:
                merged[k] = get_cnn_spectrogram_aug(str(merged["audio_anchor"]))[k]
        if _is_valid_cnn_aug_spec(merged):
            print(f"  [CnnAugResearcher] Parsed preset={merged.get('preset_name')}")
            return merged

    print("  [CnnAugResearcher] Parse failed — using fallback.")
    return _cnn_aug_fallback(round_i=round_i, history_count=history_count)


def _parse_cnn_aug_batch(
    response: str,
    batch_size: int,
    *,
    round_i: int = 0,
    history_count: int = 0,
) -> list[dict]:
    if llm_response_failed(response):
        print("  [CnnAugResearcher] LLM call failed — batch fallbacks.")
        return [
            _cnn_aug_fallback(
                slot=AUG_BATCH_SLOTS[i % 3],
                index=i,
                round_i=round_i,
                history_count=history_count,
            )
            for i in range(batch_size)
        ]

    cleaned = _strip_prose_before_json(_clean_aug_llm_response(response))
    root: dict | list | None = None
    try:
        root = json.loads(cleaned)
    except json.JSONDecodeError:
        root = _extract_first_json_object(cleaned)

    planner_note = ""
    if isinstance(root, dict):
        planner_note = str(root.get("planner_note") or root.get("batch_note") or "").strip()

    # `_coerce_experiments_list` accepts every common shape (canonical list,
    # bare list, slots-as-keys, single experiment at root) — the previous
    # parser only accepted `{"experiments": [...]}` and silently fell back
    # on the other shapes, which is why fallbacks dominated after round 1.
    specs = _coerce_experiments_list(root) if root is not None else []

    out: list[dict] = []
    for i, raw in enumerate(specs[:batch_size]):
        slot = str(raw.get("slot") or (AUG_BATCH_SLOTS[i] if i < len(AUG_BATCH_SLOTS) else f"a{i+1}"))
        merged = _sanitize_parsed_aug(_assign_slot_preset_name(raw, slot=slot, index=i + 1))
        merged["slot"] = slot
        if "audio_anchor" not in merged:
            merged["audio_anchor"] = ("high", "medium", "light")[i % 3]
        for k in CNN_SPEC_KEYS:
            if k not in merged:
                merged[k] = get_cnn_spectrogram_aug(str(merged["audio_anchor"]))[k]
        if _is_valid_cnn_aug_spec(merged):
            out.append(merged)
        else:
            out.append(
                _cnn_aug_fallback(
                    slot=slot, index=i, round_i=round_i, history_count=history_count
                )
            )

    while len(out) < batch_size:
        i = len(out)
        out.append(
            _cnn_aug_fallback(
                slot=AUG_BATCH_SLOTS[i % 3],
                index=i,
                round_i=round_i,
                history_count=history_count,
            )
        )
    out = out[:batch_size]
    if planner_note and out:
        out[0]["_planner_note"] = planner_note
    import run_log

    run_log.apply_planner_rationale_fallback(out, planner_note)
    return out


class CnnAugResearcher:
    """Propose CNN 1c augmentation configs (audio/SNR + spectrogram knobs)."""

    def __init__(
        self,
        llm: LLMClient,
        memory: ExperimentMemory,
        temperature: float = 0.35,
        *,
        batch_size: int = 3,
        refine_mode: bool = False,
        seed_spec: dict | None = None,
        compact_prompt: bool = True,
        format_json: bool = True,
        num_predict: int | None = 4096,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.batch_size = max(1, int(batch_size))
        self.refine_mode = refine_mode
        self.seed_spec = seed_spec or {}
        self.compact_prompt = compact_prompt
        self.format_json = format_json
        self.num_predict = num_predict

    def next_experiments(self, *, round_i: int = 0) -> list[dict]:
        history = self.memory.researcher_context()
        best = self.memory.best_runs(1)
        total = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"
        batch_size = self.batch_size

        # Compact context (history is the heavy block; the baselines table
        # was inflating the prompt without giving the LLM new info — the
        # system prompt already lists the same numbers).
        if self.refine_mode:
            seed_json = json.dumps(self.seed_spec, indent=2) if self.seed_spec else "{}"
            user = (
                f"{history}\n\nSEED:\n{seed_json}\n"
                f"runs={total} best={best_str}\n"
                "Return ONE refined experiment JSON (no wrapper). strategy=exploit.\n"
            )
            system = CNN_AUG_RESEARCHER_SYSTEM + "\n" + CNN_AUG_REFINE_ADDENDUM
            batch_size = 1
        elif batch_size > 1:
            slots = ", ".join(AUG_BATCH_SLOTS[:batch_size])
            user = (
                f"{history}\nruns={total} best({self.memory.ranking_metric})={best_str}\n"
                f"Return JSON with experiments=[{slots}] — {batch_size} configs.\n"
                "Make them CONTRAST (e.g. no-aug vs spec-only vs both-heavy).\n"
            )
            system = CNN_AUG_RESEARCHER_SYSTEM + "\n" + CNN_AUG_BATCH_ADDENDUM.format(
                batch_size=batch_size,
                slot_list=", ".join(f'"{s}"' for s in AUG_BATCH_SLOTS[:batch_size]),
            )
        else:
            user = (
                f"{history}\nruns={total} best={best_str}\n"
                "Return ONE experiment as a JSON object (no wrapper, no list).\n"
            )
            system = CNN_AUG_RESEARCHER_SYSTEM

        print(f"\n  [CnnAugResearcher] Planning {batch_size} config(s) ({total} in memory)...")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            format_json=self.format_json,
            num_predict=self.num_predict,
        )

        if batch_size > 1 and not self.refine_mode:
            specs = _parse_cnn_aug_batch(
                response, batch_size, round_i=round_i, history_count=total
            )
        else:
            specs = [_parse_cnn_aug_spec(response, round_i=round_i, history_count=total)]

        import run_log

        run_log.print_researcher_proposals(
            specs,
            track="cnn",
            round_label=f"1c aug round {round_i}",
        )
        return specs


def cnn_aug_trial_id(spec: dict, round_i: int, slot_i: int) -> str:
    slug = _slug_preset_name(spec.get("preset_name", f"r{round_i}_a{slot_i}"))
    return f"aug_r{round_i:02d}_{slot_i}_{slug}"[:64]
