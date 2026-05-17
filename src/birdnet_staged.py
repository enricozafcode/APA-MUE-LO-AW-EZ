"""
BirdNET staged head search — parallel to perch_agent / perch_memory (no cross-imports).

Meta-agent calls ``birdnet_agent.py`` with ``birdnet_staged`` / ``birdnet_refine`` / ``birdnet_fixed_train``.
Embeddings: frozen BirdNET 1024-d (``birdnet_agent``). Heads: researcher + coder like Perch.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from birdnet_memory import BirdnetExperimentMemory
from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from memory import ExperimentMemory
from soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict

try:
    from .birdnet_agent import ensure_caches, init_birdnet, _build_species_map
except ImportError:
    from birdnet_agent import ensure_caches, init_birdnet, _build_species_map

ROOT = Path(__file__).resolve().parents[1]
BIRDNET_EMB_DIM = 1024
BACKBONE = "birdnet"



BIRDNET_SEARCH_SPACE = {
    "search_space_note": (
        "All lists below are suggested values / examples — not hard limits. "
        "You may pick values outside these lists when arch_description requires it."
    ),
    "arch_types_note": (
        "Examples only — explore many families; reuse or tweak labels freely; invent names when helpful. "
        "Stage 1a maps the space, not final optimization."
    ),
    "arch_types": [
        "residual_mlp",       # dense residual blocks with skip connections (baseline)
        "attention_mlp",      # self-attention / multi-head attention on projected features
        "gated_mlp",          # GLU-style gating: value * sigmoid(gate), two separate Dense layers
        "highway_network",    # highway gates: out = H*transform + (1-H)*input carry
        "bottleneck_mlp",     # wide → narrow → wide projection bottleneck
        "multi_scale_mlp",    # parallel branches at different widths, merged by concat or add
        "multi_tower_ensemble",  # 3–5 parallel specialist towers, fuse logits (Average or Concat→Dense)
        "transformer_block",  # one or two transformer encoder blocks (MHA + FFN + LayerNorm)
        "mixture_of_experts", # K parallel expert MLPs with soft gating router
        "dense_connections",  # DenseNet-style: each layer receives concat of all prior outputs
        "linear_probe",       # single Dense layer — minimal baseline
    ],
    "hidden_dim":    [256, 512, 1024, 2048],
    "proj_dim":      [128, 256, 512],
    "n_layers":      [1, 2, 3, 4],
    "dropout":       [0.1, 0.2, 0.3, 0.4, 0.5],
    "activation":    ["gelu", "relu", "swish"],
    "normalization": ["layer_norm", "batch_norm", "none"],
    "learning_rate": [1e-2, 5e-3, 1e-3, 8e-4, 5e-4, 1e-4],
    "batch_size":    [64, 128, 256, 512],
    "optimizer":     ["adam", "adamw", "sgd_momentum"],
    "epochs":        [15, 25, 40, 60],
    "patience":      [3, 5, 7, 10],
    "blend_weight":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

# Shared domain context — included in researcher + coder system prompts every LLM call.
BIRDNET_TASK_CONTEXT = """
TASK — BirdCLEF 2026 (jungle / rainforest soundscapes):
- You classify bird species from audio using frozen Google BirdNET embeddings (1024-d vectors).
- Training audio comes from tropical soundscape recordings; labels are multi-label over 234 species.
- Class imbalance is severe: many species are rare; most clip×species pairs are negative.
- The BirdNET encoder encoder is FIXED. You design only the TF/Keras classification HEAD and training hyperparameters.

COMPETITION CONTEXT (read before proposing architectures — background only, not a coding constraint):
- Recordings are jungle / rainforest soundscapes; the task is multi-label species detection in noisy, imbalanced data.
- On Kaggle, long test soundscapes are evaluated as a sequence of ~60-second windows (not one score per full file).
- Temporal intuition: if the same species is active across several consecutive 60s segments, it is more likely a true
  presence than a single isolated spike—design heads that output well-calibrated per-segment probabilities (not
  over-confident on weak evidence), so later temporal aggregation can filter false alarms.
- You do NOT implement windowing or sequence models in build_head; the harness trains on cached Perch embeddings
  (one 1024-d vector per training row). Use this context for hypotheses and architecture choices only.

PREDICTION: harness uses head output only (no backbone logit blend for BirdNET).
""".strip()

BIRDNET_RESEARCHER_SYSTEM_PROMPT = (
    """You are an expert ML researcher optimizing a BirdCLEF classification head on top of frozen BirdNET 1024-d embeddings.

"""
    + BIRDNET_TASK_CONTEXT
    + """

ITERATIVE CAMPAIGN (important mindset):
- You are one step in a long search (many iterations). There is NO pressure to nail the best model in one shot.
- Treat each run as an experiment: learn from scores in the history, form a direction, then propose one concrete try.
- Failed or mediocre runs are useful signal — adjust and try again. Exploration is expected.

HOW TO WORK EACH ITERATION (keep your thinking focused; output stays compact JSON):
1. Read the experiment history — what helped, what hurt, what is still uncertain?
2. State your direction in "reasoning" (what you learned + what to try next).
3. State a testable claim in "hypothesis" (one line).
4. Propose the run in arch_type, arch_description, and hyperparameters.

BREVITY (critical — long answers slow the pipeline and are not needed):
- reasoning: at most 2 short sentences (~50 words). Lead with what you learned from past results.
- hypothesis: 1 sentence (~25 words).
- arch_description: at most 4 sentences (~150 words). Enough for the coder to implement; no essays.
- Respond with ONLY the JSON object — no markdown, no preamble, no chain-of-thought outside JSON.

YOUR PRIMARY OBJECTIVE (architecture-discovery phase): Propose the next head to train.
You are **mapping the search space**, not optimizing a single winner yet. Spread tries across many
different architecture ideas — attention, gating, MoE, ensembles, residuals, etc.
You may also revisit or tweak an existing family (same arch_type, different layout or hypers) when that
helps you learn how that family behaves. Labels do not need to be new or invented.
A later refine phase will seriously tune the most promising designs; here breadth and learning matter more than +0.001 on the leaderboard.

EXPLORING THE SPACE (both are valid in 1a):
- Try **different structural families** you have not exercised much yet (highest priority over the campaign).
- **Tweak or extend** a family you already ran — different depth, fusion, dropout, blend_weight — when it fills a gap in your understanding.
- Past runs in the registry are guidance, NOT a checklist. Under-sampled regions of the space are worth a visit.

CUSTOM arch_type NAMES (you have full freedom):
- The arch_types list in the search space is EXAMPLES ONLY, not a closed menu.
- You MAY invent new arch_type strings (snake_case, descriptive, e.g. dual_path_gated_residual,
  calibrated_ensemble_v2) whenever the design does not fit a canned label.
- arch_description must fully specify the Keras graph; the coder implements from that text, not from the label alone.

Example architecture families (starting points — combine or extend freely):
- residual_mlp:       Dense residual blocks with skip connections (BN→Dense→LN→blocks→proj→sigmoid). Solid baseline; combine with exploring other families too.
- attention_mlp:      Multi-head self-attention on projected features. Project input to hidden_dim, apply MHA, then FFN. Good for capturing feature interactions.
- gated_mlp:          GLU-style gating: two parallel Dense(hidden_dim) — one linear (value), one sigmoid (gate) — multiplied element-wise, then residual add. Selects which embedding dims are informative.
- highway_network:    Transform gate H (sigmoid) and carry gate (1-H): out = H*Dense(x) + (1-H)*x. Learnable depth blending.
- bottleneck_mlp:     Project wide→narrow→wide to force information compression: Dense(2048)→Dense(128)→Dense(2048). Forces the head to learn a compact representation.
- multi_scale_mlp:    Parallel branches at different widths (e.g. 256, 512, 1024) processing the same input, outputs merged by concatenation then projected.
- multi_tower_ensemble: 3–5 parallel specialist towers on the same stem (each tower a different topology), per-tower
  logits fused by Average or Concatenate→Dense(sigmoid). Best when you want explicit ensemble diversity in one trainable model.
- transformer_block:  Reshape to (batch,1,emb_dim), apply MultiHeadAttention + FFN + LayerNorm (1-2 blocks). Classic transformer encoder.
- mixture_of_experts: K parallel expert Dense(hidden_dim) layers; router Dense(K, softmax) on x; weighted sum of experts (all tensors hidden_dim — never concat experts then multiply by router).
- dense_connections:  DenseNet-style: concat prior features, then Dense(hidden_dim) to compress — never Add() concat-sized tensors with hidden_dim tensors.
- linear_probe:       Single Dense(n_classes, sigmoid). Strong sanity-check baseline for raw embedding quality.
- ensemble head:      Different head topologies (residual, gated, attention, MoE, etc.) on the same Perch embedding (which ones to use should be decided by the researcher).

Learning from history:
- Note which families look strong, weak, or under-tested — use that to choose the **next region of the space** to probe.
- Do not repeat identical failed configs.
- strategy "explore" (typical in 1a): sample a different part of the space — new family, invented label, OR a meaningful variant/tweak of one you already tried.
- strategy "exploit" (also fine in 1a when useful): adjust an existing configuration to learn its behavior — not for squeezing the global best, but for fair comparison within a family.
- Avoid spending most runs only micro-tuning the current #1 unless the space is already well covered.
- Soundscape context: noisy jungle audio, rare species, calibrated per-window probabilities (see TASK above).

arch_description must be implementable (stem Dense(hidden_dim), blocks, fusion, final sigmoid) but stay brief.

OUTPUT: ONLY a single JSON object — start with { and end with }. No other text.

SHAPE-SAFE arch_description rules (embeddings are 1024-d; hidden_dim is your working width, typically 512–1024):
- Always state: "Project input Dense(hidden_dim) from emb_dim first" before any residual, MoE, or concat blocks.
- For residual/highway/gated blocks: every Add() input must already be hidden_dim (project skip connections if needed).
- For mixture_of_experts: "K experts each Dense(hidden_dim); router Dense(K, softmax); weighted sum stays hidden_dim" — do NOT describe concat of expert outputs.
- For dense_connections: "Concatenate then Dense(hidden_dim) to compress" after each growth step — do NOT Add() a wide concat tensor to a narrow tensor.
- For multi_tower_ensemble: list each tower (e.g. 5 towers: residual, gated, attention, bottleneck, linear_probe_on_stem);
  each tower outputs num_classes logits before fusion; state Average vs Concatenate→Dense fusion.

Example (note short reasoning / hypothesis / arch_description):
{"arch_type": "mixture_of_experts", "arch_description": "Dense(1024) stem. Four expert Dense(1024) branches, Concatenate, Dense(1024) compress, residual Add, LayerNorm. Dropout 0.3, Dense(n_classes, sigmoid).", "hidden_dim": 1024, "n_layers": 2, "dropout": 0.3, "activation": "gelu", "normalization": "layer_norm", "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 30, "patience": 5, "blend_weight": 0.2, "reasoning": "Registry is heavy on residual_mlp; MoE not meaningfully tested. Try routed experts for species-specific features.", "hypothesis": "MoE routing helps rare species in noisy soundscapes.", "strategy": "explore"}

Required keys: arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size, optimizer, epochs, patience, blend_weight, reasoning, hypothesis, strategy."""
)

BIRDNET_BATCH_PLANNER_ADDENDUM = """
BATCH PLANNING — return exactly {batch_size} experiments in ONE JSON object (one LLM call per round):
{{
  "planner_note": "optional one short sentence",
  "experiments": [
    {{ "slot": "tweak", ...all experiment keys... }},
    {{ "slot": "explore", ... }},
    {{ "slot": "free", ... }}
  ]
}}

Slot roles (when batch_size=3):
- tweak:  adjust or extend a promising family (layout or hypers) — learn/compare within a strong idea
- explore: sample a different part of the architecture space (different family or clear structural change)
- free:   your choice — fill a gap, sanity check, or creative wildcard

Each experiment MUST include arch_description (2–4 sentences, implementable Keras layout) — the coder
implements from that text; hyperparameters alone are NOT enough.
Required keys per experiment: slot, arch_type, arch_description, hidden_dim, n_layers, dropout, activation,
normalization, learning_rate, batch_size, optimizer, epochs, patience, blend_weight, reasoning, hypothesis, strategy.
Keep reasoning/hypothesis/arch_description SHORT.
Search-space lists are hints only (e.g. n_layers: 5 is allowed).
Output either {{"planner_note":"...", "experiments":[...]}} OR a JSON array of 3 experiment objects — not separate objects.
"""

BIRDNET_STAGE_1A_EXPLORE_ADDENDUM = """
STAGE 1a — EXPLORE THE ARCHITECTURE SPACE (you are in this phase now):
- Goal: learn what kinds of heads work — cover many different architectures over the campaign, not maximize one score yet.
- Favor variety: rotate through different families (attention, gated, MoE, multi-tower, highway, residual, etc.).
- Tweaking an existing arch_type (layout or hypers) is welcome when it helps you compare or understand that family.
- Nothing must be "new" — example labels and invented names are both fine; what matters is useful coverage of ideas.
- Do not obsess over beating the current best each iteration; small improvements to the leader are low priority vs unexplored space.
- Serious optimization of the top candidates happens in a LATER refine phase.
"""

BIRDNET_REFINE_RESEARCHER_ADDENDUM = """
REFINE MODE (stage 1b) — optimize the current best LOCKED head (same arch_type for every experiment).
- Do NOT switch architecture family. Decide quickly — short JSON, short reasoning.
- All experiments are independent tries to beat the champion score (hypers and/or layout within the family).
- There are NO explore/tweak/free roles — every slot is just another optimization attempt.
- strategy MUST be "exploit" for every experiment.
- Keep reasoning/hypothesis/arch_description SHORT (same limits as stage 1a).
"""

BIRDNET_REFINE_BATCH_PLANNER_ADDENDUM = """
STAGE 1b BATCH — return exactly {batch_size} refine experiments in ONE JSON (one LLM call per planner round):
{{
  "planner_note": "optional one short sentence",
  "experiments": [
    {{ "arch_type": "<LOCKED>", ...all experiment keys... }},
    {{ "arch_type": "<LOCKED>", ... }},
    ...
  ]
}}

arch_type MUST equal the locked champion family ({locked}) for every experiment.
Each experiment is another attempt to improve the same best model — vary hypers and/or layout as you see fit.
Optional "slot" field is only a label (e.g. r1, r2) — not a role.

Each experiment MUST include arch_description (2–4 sentences). Required keys: arch_type,
arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size,
optimizer, epochs, patience, blend_weight, reasoning, hypothesis, strategy (all exploit).
Output {{"planner_note":"...", "experiments":[...]}} OR a JSON array of {batch_size} objects.
"""
class BirdnetResearcher:
    def __init__(
        self,
        llm: LLMClient,
        memory: ExperimentMemory,
        temperature: float = 0.6,
        *,
        refine_mode: bool = False,
        locked_arch_type: str | None = None,
        seed_spec: dict | None = None,
        seed_score: float | None = None,
        batch_size: int = 1,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.locked_arch_type = locked_arch_type
        self.seed_spec = seed_spec or {}
        self.seed_score = seed_score
        self.batch_size = max(1, int(batch_size))

    def next_experiment(self) -> dict:
        return self.next_experiments()[0]

    def next_experiments(self) -> list[dict]:
        history  = self.memory.researcher_context()
        best     = self.memory.best_runs(1)
        total    = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"

        batch_size = self.batch_size
        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "residual_mlp")
            user_prompt = (
                f"{history}\n\n"
                f"Available search space (hints only):\n"
                f"{json.dumps(BIRDNET_SEARCH_SPACE, indent=2)}\n\n"
                f"Refine runs so far: {total} | best in this campaign: {best_str}\n\n"
            )
            if batch_size > 1:
                user_prompt += (
                    f"Stage 1b batch — propose exactly {batch_size} independent optimization tries "
                    f"(all arch_type={locked}).\n"
                    "Each experiment should try to beat the champion config/score — no explore roles.\n\n"
                    f'Respond with ONE JSON: {{"planner_note": "...", "experiments": [ ... ]}} '
                    f"with {batch_size} items. No prose outside JSON.\n"
                )
                refine_batch = BIRDNET_REFINE_BATCH_PLANNER_ADDENDUM.format(
                    batch_size=batch_size, locked=locked
                )
                system_prompt = (
                    BIRDNET_RESEARCHER_SYSTEM_PROMPT
                    + "\n\n"
                    + BIRDNET_REFINE_RESEARCHER_ADDENDUM
                    + "\n\n"
                    + refine_batch
                )
            else:
                user_prompt += (
                    "Propose ONE refine experiment (arch_type locked). Keep JSON short. strategy=exploit.\n"
                )
                system_prompt = BIRDNET_RESEARCHER_SYSTEM_PROMPT + "\n\n" + BIRDNET_REFINE_RESEARCHER_ADDENDUM
        elif batch_size > 1:
            user_prompt = (
                f"{history}\n\n"
                f"Available search space:\n{json.dumps(BIRDNET_SEARCH_SPACE, indent=2)}\n\n"
                f"Total experiments so far: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                f"Stage 1a batch plan — propose exactly {batch_size} experiments for this round "
                f"(slots: {', '.join(BIRDNET_BATCH_SLOTS[:batch_size])}).\n"
                "Explore the space across the three slots; not optimizing the leaderboard in one shot.\n\n"
                f'Respond with ONE JSON object: {{"planner_note": "...", "experiments": [ ... ]}} '
                f"with {batch_size} items. No prose outside JSON.\n"
            )
            batch_addendum = BIRDNET_BATCH_PLANNER_ADDENDUM.format(batch_size=batch_size)
            system_prompt = (
                BIRDNET_RESEARCHER_SYSTEM_PROMPT
                + "\n\n"
                + BIRDNET_STAGE_1A_EXPLORE_ADDENDUM
                + "\n\n"
                + batch_addendum
            )
        else:
            user_prompt = (
                f"{history}\n\n"
                f"Available search space:\n{json.dumps(BIRDNET_SEARCH_SPACE, indent=2)}\n\n"
                f"Total experiments so far: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                "Stage 1a — explore the architecture space: try different families across iterations; tweaks to existing configs are OK too.\n"
                "Workflow: (1) what is under-tested or unclear? (2) pick next experiment to learn (3) compact JSON. Not optimizing the leaderboard yet.\n\n"
                "Respond with ONLY a short JSON object — no prose outside JSON. Start with { and end with }. Keys: "
                "arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, "
                "learning_rate, batch_size, optimizer, epochs, patience, blend_weight, reasoning, hypothesis, strategy."
            )
            system_prompt = BIRDNET_RESEARCHER_SYSTEM_PROMPT + "\n\n" + BIRDNET_STAGE_1A_EXPLORE_ADDENDUM

        batch_label = f", batch={batch_size}" if batch_size > 1 else ""
        print(
            f"\n  [Researcher] Planning next "
            f"{'experiment' if batch_size == 1 else f'{batch_size} experiments'} "
            f"({total} in memory, best {best_str}{batch_label}, "
            f"timeout {self.llm.timeout_seconds:.0f}s)..."
        )
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        if _llm_response_failed(response):
            print(
                f"  [Researcher] LLM unavailable or timed out — "
                f"using safe defaults for this round."
            )
            if response:
                print(f"  [Researcher] Detail: {str(response)[:200]}")
            specs = self._refine_or_explore_batch_fallback(batch_size, timed_out=True)
        elif batch_size > 1:
            specs = self._parse_batch_specs(response, batch_size)
        else:
            specs = [self._parse_spec(response)]
        if self.refine_mode and self.locked_arch_type:
            slot_labels = _refine_batch_slot_labels(len(specs))
            for i, spec in enumerate(specs):
                spec["arch_type"] = self.locked_arch_type
                spec["strategy"] = "exploit"
                if i < len(slot_labels):
                    spec["slot"] = slot_labels[i]
        planner_note = ""
        if batch_size > 1 and specs and specs[0].get("_planner_note"):
            planner_note = specs[0].pop("_planner_note", "")
        if planner_note:
            print(f"  [Researcher] Plan: {planner_note[:160]}")
        for i, spec in enumerate(specs, 1):
            slot = spec.get("slot", f"s{i}")
            note = ""
            if spec.pop("_arch_description_synthesized", False):
                note = " | desc=synthesized (planner omitted arch_description)"
            desc_prev = str(spec.get("arch_description", ""))
            if len(desc_prev) > 70:
                desc_prev = desc_prev[:70] + "…"
            print(
                f"  [Researcher] [{slot}] {spec.get('strategy', '?')} | "
                f"arch={spec.get('arch_type')} | desc={desc_prev}{note}"
            )
        return specs

    @staticmethod
    def _clean_llm_response(response: str) -> str:
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

    def _parse_spec(self, response: str) -> dict:
        cleaned = self._clean_llm_response(response)

        # Try ```json ... ``` block
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if m:
            try:
                return _birdnet_fill_defaults(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

        root = self._parse_batch_root(cleaned)
        if isinstance(root, dict):
            exps = self._experiments_from_root(root)
            if exps:
                return exps[0]
            if root.get("arch_type"):
                normalized = _normalize_experiment_item(root)
                if normalized is not None:
                    return normalized

        print("  [Researcher] Warning: could not parse JSON, using safe defaults.")
        print(f"  [Researcher] Raw response (first 400 chars): {repr(cleaned[:400])}")
        return _birdnet_safe_defaults()

    def _parse_batch_root(self, cleaned: str) -> dict | None:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return {"experiments": parsed}
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        arr = _extract_first_json_array(cleaned)
        if arr is not None:
            return {"experiments": arr}
        root = _extract_first_json_object(cleaned)
        if root is not None:
            if "experiments" not in root and isinstance(root.get("arch_type"), str):
                return {"experiments": [root]}
            return root
        return None

    def _experiments_from_root(self, root: dict) -> list[dict]:
        raw_list = root.get("experiments")
        if not isinstance(raw_list, list):
            return []
        specs: list[dict] = []
        for item in raw_list:
            normalized = _normalize_experiment_item(item)
            if normalized is not None and _is_valid_experiment_spec(normalized):
                specs.append(normalized)
        return specs

    def _parse_batch_specs(self, response: str, batch_size: int) -> list[dict]:
        cleaned = self._clean_llm_response(response)
        root = self._parse_batch_root(cleaned)
        specs: list[dict] = []
        planner_note = ""

        if isinstance(root, dict):
            planner_note = str(root.get("planner_note") or root.get("batch_note") or "")[:200]
            specs = self._experiments_from_root(root)

        valid = [s for s in specs if _is_valid_experiment_spec(s)]
        if not valid:
            print("  [Researcher] Warning: could not parse batch JSON — using slot fallbacks.")
            print(f"  [Researcher] Raw response (first 500 chars): {repr(cleaned[:500])}")
            return self._refine_or_explore_batch_fallback(batch_size, timed_out=False)

        specs = valid
        slot_pool = (
            _refine_batch_slot_labels(batch_size)
            if self.refine_mode
            else BIRDNET_BATCH_SLOTS
        )
        for i, spec in enumerate(specs):
            if i < len(slot_pool):
                spec["slot"] = slot_pool[i]
        if planner_note and specs:
            specs[0]["_planner_note"] = planner_note
        if len(specs) < batch_size:
            print(
                f"  [Researcher] Warning: parsed {len(specs)}/{batch_size} experiments — "
                f"filling remainder with slot fallbacks."
            )
            specs.extend(
                self._refine_or_explore_batch_fallback(
                    batch_size - len(specs), timed_out=False
                )
            )
        return specs[:batch_size]

    def _refine_or_explore_batch_fallback(
        self, batch_size: int, *, timed_out: bool
    ) -> list[dict]:
        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "residual_mlp")
            return _birdnet_refine_batch_fallback(
                batch_size,
                timed_out=timed_out,
                locked_arch_type=str(locked),
                seed_spec=self.seed_spec,
            )
        return _birdnet_batch_fallback(batch_size, timed_out=timed_out)


def _resolve_experiments_per_round(config: dict, *, refine_mode: bool = False) -> int:
    """Experiments planned per researcher LLM call (1a explore + 1b refine)."""
    rc = config.get("researcher") or {}
    perch = config.get("perch") or {}
    if refine_mode:
        refine = config.get("perch_refine") or {}
        raw = refine.get("experiments_per_researcher_call")
        if raw is None:
            raw = rc.get("batch_size", 3)
        return max(1, int(raw))
    raw = rc.get("batch_size")
    if raw is None:
        raw = perch.get("experiments_per_researcher_call", 1)
    return max(1, int(raw))


def _ranking_value_from_memory_entry(entry: dict, metric: str) -> float:
    if metric == "macro_roc_auc":
        v = entry.get("macro_roc_auc")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_roc_auc")
    else:
        v = entry.get("macro_average_precision")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_average_precision")
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _best_run_from_memory_dir(
    mem_dir: Path,
    *,
    ranking_metric: str,
    locked_arch_type: str | None = None,
) -> dict | None:
    """Best successful run in a perch memory directory (jsonl + best_model_info)."""
    mem_dir = Path(mem_dir)
    if not mem_dir.is_dir():
        return None

    best_entry: dict | None = None
    best_val = -1.0

    jsonl = mem_dir / "experiment_memory.jsonl"
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not entry.get("success"):
                    continue
                spec = entry.get("spec") or {}
                at = spec.get("arch_type")
                if locked_arch_type and at and at != locked_arch_type:
                    continue
                val = _ranking_value_from_memory_entry(entry, ranking_metric)
                if val > best_val:
                    best_val = val
                    best_entry = dict(entry)
                    best_entry["_source_memory_dir"] = str(mem_dir)
                    best_entry["_ranking_value"] = val

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec_info = info.get("spec") or {}
            at = spec_info.get("arch_type")
            if locked_arch_type and at and at != locked_arch_type:
                pass
            else:
                val = float(
                    info.get("ranking_value", info.get("macro_average_precision", -1))
                )
                if val > best_val:
                    best_val = val
                    best_entry = {
                        "success": True,
                        "spec": spec_info,
                        "macro_average_precision": info.get("macro_average_precision"),
                        "macro_roc_auc": info.get("macro_roc_auc"),
                        "median_per_class_auc": info.get("median_per_class_auc"),
                        "metrics": info,
                        "_source_memory_dir": str(mem_dir),
                        "_ranking_value": val,
                    }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return best_entry


def _resolve_refine_champion(
    mem_dir: Path,
    refine_cfg: dict,
    *,
    ranking_metric: str,
) -> dict:
    """
    Scan refine + parent experiment memories and pick the best successful run
  (same locked arch_type). Beats relying only on the 1a handoff score.
    """
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )
    candidates: list[dict] = []

    handoff_spec = dict(refine_cfg.get("seed_spec") or {})
    handoff_score = float(refine_cfg.get("seed_score", -1.0))
    if handoff_spec:
        candidates.append(
            {
                "spec": handoff_spec,
                "ranking_value": handoff_score,
                "macro_average_precision": refine_cfg.get("seed_macro_ap"),
                "macro_roc_auc": refine_cfg.get("seed_macro_auc"),
                "median_per_class_auc": refine_cfg.get("seed_median_auc"),
                "source": "stage_1a_handoff",
                "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
            }
        )

    seen_dirs: set[str] = set()
    for raw in (refine_cfg.get("parent_memory_dir"), str(mem_dir)):
        if not raw:
            continue
        d = Path(raw)
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        best = _best_run_from_memory_dir(
            d, ranking_metric=ranking_metric, locked_arch_type=str(locked)
        )
        if best is None:
            continue
        spec = dict(best.get("spec") or {})
        val = float(best.get("_ranking_value", _ranking_value_from_memory_entry(best, ranking_metric)))
        candidates.append(
            {
                "spec": spec,
                "ranking_value": val,
                "macro_average_precision": best.get("macro_average_precision"),
                "macro_roc_auc": best.get("macro_roc_auc"),
                "median_per_class_auc": best.get("median_per_class_auc"),
                "source": f"memory:{d.name}",
                "memory_dir": best.get("_source_memory_dir", str(d)),
            }
        )

    if not candidates:
        return {
            "spec": handoff_spec or {"arch_type": locked},
            "ranking_value": handoff_score,
            "source": "default",
            "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
        }

    winner = max(candidates, key=lambda c: float(c.get("ranking_value", -1.0)))
    return winner


def _load_refine_champion_spec(mem_dir: Path) -> dict | None:
    """Canonical champion written at refine start (full spec + metadata)."""
    mem_dir = Path(mem_dir)
    for name in (REFINE_CHAMPION_SPEC_FILE, LEGACY_CHAMPION_SPEC_FILE):
        path = mem_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        spec = dict(payload.get("spec") or {})
        if payload.get("locked_arch_type"):
            spec["arch_type"] = payload["locked_arch_type"]
        return spec or None
    return None


def _persist_refine_champion_artifacts(
    mem_dir: Path, refine_cfg: dict, *, ranking_metric: str
) -> tuple[dict, float, dict]:
    """
    Resolve best model from experiment memories, write refine_champion_spec.json,
    and copy winning head artifacts into the refine memory dir.
    """
    import shutil

    mem_dir.mkdir(parents=True, exist_ok=True)
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    winner = _resolve_refine_champion(mem_dir, refine_cfg, ranking_metric=ranking_metric)
    seed_spec = dict(winner.get("spec") or {})
    seed_score = float(winner.get("ranking_value", refine_cfg.get("seed_score", -1.0)))
    source = str(winner.get("source", "?"))
    winner_mem = Path(winner.get("memory_dir") or mem_dir)

    if winner_mem.is_dir() and winner_mem != mem_dir:
        info_path = winner_mem / "best_model_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                for k, v in (info.get("spec") or {}).items():
                    if v is not None and k not in seed_spec:
                        seed_spec[k] = v
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists():
                shutil.copy2(src, mem_dir / name)
    elif winner_mem.is_dir():
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists() and not (mem_dir / name).exists():
                shutil.copy2(src, mem_dir / name)

    seed_spec["arch_type"] = locked
    if not seed_spec.get("arch_description"):
        seed_spec["arch_description"] = _synthesize_arch_description(seed_spec)

    payload = {
        "locked_arch_type": locked,
        "aug_baseline": refine_cfg.get("aug_baseline"),
        "champion_score": seed_score,
        "ranking_metric": ranking_metric,
        "source": source,
        "winner_memory_dir": str(winner_mem),
        "parent_memory_dir": refine_cfg.get("parent_memory_dir"),
        "macro_average_precision": winner.get("macro_average_precision"),
        "macro_roc_auc": winner.get("macro_roc_auc"),
        "median_per_class_auc": winner.get("median_per_class_auc"),
        "spec": seed_spec,
    }
    (mem_dir / REFINE_CHAMPION_SPEC_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"  [Refine] Champion from {source}: {ranking_metric}={seed_score:.5f} "
        f"({locked})"
    )
    return seed_spec, seed_score, winner


def _birdnet_refine_batch_fallback(
    batch_size: int,
    *,
    timed_out: bool,
    locked_arch_type: str,
    seed_spec: dict,
) -> list[dict]:
    """Refine fallbacks — independent optimization tries near the current champion."""
    reason = (
        "Researcher LLM timed out; refine fallback."
        if timed_out
        else "Refine batch JSON parse failed; fallback."
    )
    base = _birdnet_fill_defaults(dict(seed_spec))
    base["arch_type"] = locked_arch_type
    base["strategy"] = "exploit"
    if not base.get("arch_description"):
        base["arch_description"] = _synthesize_arch_description(base)

    def _vary(label: str, **overrides) -> dict:
        s = dict(base)
        s.update(overrides)
        s["slot"] = label
        s["strategy"] = "exploit"
        s["arch_type"] = locked_arch_type
        s.setdefault("hypothesis", f"Optimize champion ({label}).")
        s["reasoning"] = reason + f" {label}."
        return s

    lr = float(base.get("learning_rate", 1e-3))
    drop = float(base.get("dropout", 0.3))
    pw = float(base.get("blend_weight", 0.2))
    nl = int(base.get("n_layers", 2))
    templates: list[dict] = [
        _vary("r1", learning_rate=max(lr * 0.5, 1e-5)),
        _vary("r2", dropout=min(0.5, drop + 0.1), n_layers=min(4, nl + 1)),
        _vary("r3", blend_weight=max(0.0, pw - 0.1)),
    ]
    labels = _refine_batch_slot_labels(batch_size)
    out: list[dict] = []
    for i in range(batch_size):
        s = dict(templates[i % len(templates)])
        s["slot"] = labels[i]

def _resolve_researcher_timeout_seconds(config: dict) -> float:
    """Researcher LLM timeout; stage 1c uses meta_agent.stage_1c.* — perch reads researcher/llm_researcher."""
    meta = config.get("meta_agent") or {}
    for key in ("researcher_timeout_seconds",):
        if meta.get(key) is not None:
            return float(meta[key])
    rc = config.get("researcher") or {}
    llm_rc = config.get("llm_researcher") or {}
    for block in (rc, llm_rc):
        if block.get("timeout_seconds") is not None:
            return float(block["timeout_seconds"])
    return 300.0


_BIRDNET_DEFAULT_ARCH_DESCRIPTION = (
    "BatchNorm on input. Dense(1024) projection with LayerNorm. "
    "Then 2 residual blocks: each block applies Dense(1024), LayerNorm, GELU, "
    "Dropout(0.3), Dense(1024), then adds the block input (skip connection), "
    "followed by LayerNorm. Final Dense(512, gelu), Dropout(0.4), Dense(n_classes, sigmoid)."
)

# Hyperparameter defaults only — never copy parse-failure reasoning into LLM-parsed specs.
_BIRDNET_SPEC_FIELD_DEFAULTS: dict = {
    "hidden_dim":       1024,
    "proj_dim":         512,
    "n_layers":         2,
    "dropout":          0.3,
    "activation":       "gelu",
    "normalization":    "layer_norm",
    "learning_rate":    8e-4,
    "batch_size":       256,
    "optimizer":        "adam",
    "epochs":           25,
    "patience":         5,
    "blend_weight":     0.2,
    "strategy":         "explore",
    "reasoning":        "",
    "hypothesis":       "",
}


def _birdnet_safe_defaults() -> dict:
    return {
        "arch_type":        "residual_mlp",
        "arch_description": _BIRDNET_DEFAULT_ARCH_DESCRIPTION,
        **_BIRDNET_SPEC_FIELD_DEFAULTS,
        "reasoning":        "Fallback defaults — researcher output could not be parsed.",
        "hypothesis":       "Baseline residual head config.",
        "strategy":         "explore",
    }


def _birdnet_fill_defaults(spec: dict) -> dict:
    """Fill missing hyperparameter keys; preserve arch_type/description/reasoning from the LLM."""
    out = dict(spec)
    for k, v in _BIRDNET_SPEC_FIELD_DEFAULTS.items():
        if k not in out or out[k] is None:
            out[k] = v
    if not out.get("arch_type"):
        out["arch_type"] = "residual_mlp"
    return out


def _synthesize_arch_description(spec: dict) -> str:
    """Minimal layout hint when the planner omits arch_description (better than wrong residual template)."""
    at = str(spec.get("arch_type") or "mlp")
    h = int(spec.get("hidden_dim") or 1024)
    nl = int(spec.get("n_layers") or 2)
    drop = spec.get("dropout", 0.3)
    act = spec.get("activation", "gelu")
    norm = spec.get("normalization", "layer_norm")
    if at == "linear_probe":
        return "Single Dense(num_classes, sigmoid) on raw embeddings."
    if at == "transformer_block":
        return (
            f"Dense({h}) stem from emb_dim. Reshape (1, {h}), one MultiHeadAttention + FFN + {norm}, "
            f"residual, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    if at == "mixture_of_experts":
        return (
            f"Dense({h}) stem. {nl} expert Dense({h}) branches, Concatenate, Dense({h}) compress, "
            f"residual Add, {norm}, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    if at == "dense_connections":
        return (
            f"Dense({h}) stem. {nl} dense blocks: Concatenate growth then Dense({h}) compress each step, "
            f"{act}, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    return (
        f"{at}: Dense({h}) stem from emb_dim. {nl} blocks with {act}, {norm}, Dropout({drop}), "
        f"Dense(num_classes, sigmoid). Follow standard shape-safe residual/gated patterns for this family."
    )


def _normalize_experiment_item(raw: dict) -> dict | None:
    if not isinstance(raw, dict) or not str(raw.get("arch_type") or "").strip():
        return None
    out = _birdnet_fill_defaults(dict(raw))
    desc = str(raw.get("arch_description") or "").strip()
    if desc:
        out["arch_description"] = desc
    else:
        out["arch_description"] = _synthesize_arch_description(out)
        out["_arch_description_synthesized"] = True
    return out


def _is_valid_experiment_spec(spec: dict) -> bool:
    return bool(str(spec.get("arch_type") or "").strip()) and bool(
        str(spec.get("arch_description") or "").strip()
    )
BIRDNET_CODER_SYSTEM_PROMPT = (
    """You are a Python ML engineer building TF/Keras classification heads on frozen 1024-d BirdNET embeddings.

"""
    + BIRDNET_TASK_CONTEXT
    + """

The researcher proposes an experimental STRATEGY (architecture family, key hyperparameters, hypothesis).
arch_type may be any descriptive snake_case label (including names not in the example list); arch_description
is authoritative. YOU implement the exact Keras head in build_head — not a fixed template. Follow arch_description closely.
Advanced designs are encouraged when specified, especially multi_tower_ensemble (3–5 diverse parallel towers
on the same embedding, fuse per-tower sigmoid logits with Average or Concatenate→Dense(num_classes, sigmoid)).
Goal: accurate, well-calibrated per-window species probabilities for long jungle soundscape recordings.

You must output TWO functions in a single ```python``` code block:
  1. build_head(emb_dim, num_classes) → tf.keras.Model
  2. get_training_config() → dict with learning_rate, batch_size, optimizer, epochs, patience, blend_weight

HARD RULES (the harness depends on these):
- tf is already imported — do NOT add ANY imports
- No top-level code, no main(), no class definitions — only the two functions
- build_head MUST return tf.keras.Model(inp, out)
- The FINAL layer MUST be Dense(num_classes, activation="sigmoid")  (multi-label)
- Do NOT use tf.keras.layers.Lambda (breaks model save/load)
- Do NOT slice tensors with [:, a:b] (breaks the functional API) — use separate Dense layers instead

*** MANDATORY STEM (fixes 1024 vs 512 crashes) ***
emb_dim is 1024. Pick hidden_dim (e.g. 512) from the spec. IMMEDIATELY after Input:
  inp = tf.keras.layers.Input(shape=(emb_dim,))
  x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)   # REQUIRED — never Add() against raw inp
From this point on, the main trunk tensor x must stay shape (batch, hidden_dim) unless you explicitly project back.

*** CRITICAL — Add() / Multiply() SHAPE RULE ***
tf.keras.layers.Add()([a, b]) and Multiply()([a, b]) CRASH if shapes differ (e.g. (1024,) vs (512,), or (2048,) vs (512,)).
- Use ONE hidden_dim everywhere inside blocks; both Add() inputs must be (batch, hidden_dim).
- Project the skip path: skip = tf.keras.layers.Dense(hidden_dim)(skip) before Add().
- Add() takes exactly TWO same-shaped tensors — never Add()([x, a, b]) with three different branches unless all three are Dense(hidden_dim) first.
- After Concatenate, width grows — you CANNOT Add() the concat to x. Use Dense(hidden_dim) on the concat output first.

OTHER SHAPE GOTCHAS:
- MultiHeadAttention needs 3D: Reshape((1, hidden_dim)) → MHA → Reshape((hidden_dim,)).
- Concatenate(axis=-1) stacks widths (512+512=1024). Follow with Dense(hidden_dim) before any Add() with x.
- Do NOT Concatenate expert outputs then Multiply with a router — use the MoE pattern below.

REFERENCE PATTERNS (copy the shape logic; adapt layer counts):

# Stem (always)
inp = tf.keras.layers.Input(shape=(emb_dim,))
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)

# Residual block — x and h both hidden_dim
h = tf.keras.layers.Dense(hidden_dim)(x)
h = tf.keras.layers.LayerNormalization()(h)
h = tf.keras.layers.Activation("gelu")(h)
h = tf.keras.layers.Dropout(dropout)(h)
h = tf.keras.layers.Dense(hidden_dim)(h)
x = tf.keras.layers.Add()([x, h])

# Mixture of experts (K=4) — concat experts, compress (shape-safe; no router multiply bugs)
num_experts = 4
expert_outs = [tf.keras.layers.Dense(hidden_dim, activation="gelu")(x) for _ in range(num_experts)]
moe = tf.keras.layers.Concatenate()(expert_outs)              # (batch, K*hidden_dim)
moe = tf.keras.layers.Dense(hidden_dim, activation="gelu")(moe)  # learn mixture weights
x = tf.keras.layers.Add()([x, moe])

# DenseNet-style dense connection — concat then compress
dense_in = tf.keras.layers.Concatenate()([x, h])
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(dense_in)

# Multi-scale — concat branches then project
b1 = tf.keras.layers.Dense(256, activation="gelu")(x)
b2 = tf.keras.layers.Dense(512, activation="gelu")(x)
merged = tf.keras.layers.Concatenate()([b1, b2])
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(merged)

# Gated (GLU) block
v = tf.keras.layers.Dense(hidden_dim, activation="linear")(x)
g = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")(x)
gated = tf.keras.layers.Multiply()([v, g])
x = tf.keras.layers.Add()([x, gated])

# Attention block
x_3d = tf.keras.layers.Reshape((1, hidden_dim))(x)
attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=hidden_dim // 4)(x_3d, x_3d)
attn = tf.keras.layers.Reshape((hidden_dim,))(attn)
x = tf.keras.layers.Add()([x, attn])

# Classifier head (after trunk is stable)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

# Multi-tower ensemble (e.g. 5 towers) — shape-safe; each tower: stem branch → Dense(num_classes, sigmoid)
inp = tf.keras.layers.Input(shape=(emb_dim,))
stem = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)
def _tower_residual(x):
    h = tf.keras.layers.Dense(hidden_dim, activation="gelu")(x)
    h = tf.keras.layers.Dense(hidden_dim)(h)
    return tf.keras.layers.Add()([x, h])
def _tower_gated(x):
    v = tf.keras.layers.Dense(hidden_dim, activation="linear")(x)
    g = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")(x)
    return tf.keras.layers.Multiply()([v, g])
t1 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(_tower_residual(stem))
t2 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(_tower_gated(stem))
t3 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(stem)  # linear-ish tower
# ... add t4, t5 with attention / bottleneck as needed ...
out = tf.keras.layers.Average()([t1, t2, t3])  # all (batch, num_classes) — OK

For mixture_of_experts: K experts → Concatenate → Dense(hidden_dim) → Add with x (do NOT Multiply softmax router against concat experts).
For dense_connections: Concatenate([x, h]) → Dense(hidden_dim) replaces x (do NOT Add concat to x).
For multi_tower_ensemble: each tower must output (batch, num_classes) before Average/Concat fusion; never Add() towers at hidden_dim then one shared classifier unless you design it that way explicitly.

Keep build_head shape-safe. Prefer a working simpler model over a broken exotic one."""
)


def _spec_to_coder_prompt(spec: dict) -> str:
    arch_type        = spec.get("arch_type", "residual_mlp")
    arch_description = spec.get("arch_description", "(no description provided)")
    hypothesis       = spec.get("hypothesis", "")
    reasoning        = spec.get("reasoning", "")
    strategy         = spec.get("strategy", "explore")

    arch_keys = ("hidden_dim", "proj_dim", "n_layers", "dropout", "activation", "normalization")
    arch_cfg  = {k: spec[k] for k in arch_keys if k in spec}

    training_keys = ("learning_rate", "batch_size", "optimizer", "epochs", "patience", "blend_weight")
    training_cfg  = {k: spec[k] for k in training_keys if k in spec}

    return (
        f"Researcher's experimental proposal:\n"
        f"  arch_type:  {arch_type}\n"
        f"  strategy:   {strategy}\n"
        f"  hypothesis: {hypothesis}\n"
        f"  reasoning:  {reasoning}\n\n"
        f"Architecture description (use as design guidance — implement faithfully but make your own concrete choices):\n"
        f"  {arch_description}\n\n"
        f"Suggested architecture hyperparameters (these are HINTS — adapt as needed for shape safety):\n"
        f"{json.dumps(arch_cfg, indent=2)}\n\n"
        f"Training config to return from get_training_config() (use these values verbatim unless they would break training):\n"
        f"{json.dumps(training_cfg, indent=2)}\n\n"
        f"{_arch_type_hint(arch_type)}"
        f"Implement build_head(emb_dim, num_classes) freely in idiomatic Keras functional API, "
        f"obeying all HARD RULES, the MANDATORY STEM, and Add() SHAPE RULE. "
        f"Return BOTH functions in a single ```python``` code block."
    )


def _extract_code(response: str) -> str | None:
    match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        first = candidate.splitlines()[0].strip().lower() if candidate else ""
        if first in ("python", "py", ""):
            candidate = "\n".join(candidate.splitlines()[1:]).strip() if first else candidate
        return candidate or None
    return None


def _validate_perch_code(code: str) -> list[str]:
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "build_head" not in names:
        issues.append("Missing: build_head(emb_dim, num_classes)")
    if "get_training_config" not in names:
        issues.append("Missing: get_training_config()")
    return issues


def _dry_run_build_head(
    code: str, emb_dim: int = 1024, num_classes: int = 234
) -> list[str]:
    """Instantiate build_head with dummy shapes — catches Add/Concat bugs in seconds."""
    import tensorflow as tf

    namespace: dict = {"tf": tf}
    try:
        exec(code, namespace)  # noqa: S102 — trusted coder slot in isolated process
    except Exception as e:
        return [f"exec failed: {type(e).__name__}: {e}"]
    build_head = namespace.get("build_head")
    if build_head is None:
        return ["build_head not defined after exec"]
    try:
        model = build_head(emb_dim, num_classes)
        if not isinstance(model, tf.keras.Model):
            return ["build_head must return tf.keras.Model(inp, out)"]
        model(tf.zeros((2, emb_dim), dtype=tf.float32), training=False)
    except Exception as e:
        return [f"build_head shape error: {type(e).__name__}: {e}"]
    return []


_ARCH_TYPE_SHAPE_HINTS: dict[str, str] = {
    "mixture_of_experts": (
        "MoE: K parallel Dense(hidden_dim)(x), Concatenate expert outputs, "
        "Dense(hidden_dim) to compress, Add with x. Do NOT softmax-multiply router against concat."
    ),
    "dense_connections": (
        "DenseNet: h = Dense(hidden_dim)(x); dense_in = Concatenate([x, h]); "
        "x = Dense(hidden_dim)(dense_in). Never Add() concat tensor to x directly."
    ),
    "multi_scale_mlp": (
        "Parallel Dense branches → Concatenate → Dense(hidden_dim) before Add with trunk."
    ),
    "bottleneck_mlp": (
        "Dense(wide)→Dense(narrow)→Dense(hidden_dim) before any Add() with x; x must stay hidden_dim."
    ),
    "residual_mlp": (
        "Stem Dense(hidden_dim)(inp) first; each block ends with Dense(hidden_dim) before Add([x, h])."
    ),
    "attention_mlp": "Project to hidden_dim, Reshape (1, H) for MHA, Reshape (H,) back, Add with x.",
    "transformer_block": "Stem to hidden_dim; MHA on (batch, 1, hidden_dim).",
    "multi_tower_ensemble": (
        "Shared stem Dense(hidden_dim)(inp). Build 3–5 parallel towers (different topologies). "
        "Each tower ends with Dense(num_classes, sigmoid). Fuse with Average([t1,t2,...]) or "
        "Concatenate(towers)→Dense(num_classes, sigmoid). All tower outputs must be (batch, num_classes)."
    ),
}

def _arch_type_hint(arch_type: str) -> str:
    hint = _ARCH_TYPE_SHAPE_HINTS.get(arch_type)
    if not hint:
        return ""
    return f"\nShape hint for {arch_type}: {hint}\n"


# Known-good fallback: used when the coder exhausts all retries.
# Implements the safe-default residual MLP so no iteration is ever wasted.
# Simplest possible head: single linear layer (pure linear probe on Perch embeddings).
# Used as (a) mandatory iteration-0 baseline and (b) fallback if the coder fails all retries.
_SAFE_DEFAULT_SLOT_CODE = '''
def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(inp)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 20,
        "patience": 5,
        "blend_weight": 0.2,
    }
'''.strip()

_BASELINE_SPEC = {
    "arch_type":        "linear_probe",
    "arch_description": "Single Dense(num_classes, sigmoid) — pure linear probe on raw 1024-d Perch embeddings. No hidden layers.",
    "hidden_dim":       0,
    "n_layers":         0,
    "dropout":          0.0,
    "activation":       "sigmoid",
    "normalization":    "none",
    "learning_rate":    1e-3,
    "batch_size":       256,
    "optimizer":        "adam",
    "epochs":           20,
    "patience":         5,
    "blend_weight":     0.2,
    "reasoning":        "Mandatory baseline — establishes the linear separability ceiling of raw Perch embeddings.",
    "hypothesis":       "Linear probe sets the floor; any non-linear head should beat this.",
    "strategy":         "baseline",
}
def generate_birdnet_code(
    coder_llm: LLMClient, spec: dict, temperature: float, max_retries: int = 5
) -> str | None:
    prompt = _spec_to_birdnet_coder_prompt(spec)
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        print(f"  [Coder] Attempt {attempt}/{max_retries}...")
        response = coder_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": BIRDNET_CODER_SYSTEM_PROMPT},
                {"role": "user",   "content": current_prompt},
            ],
            temperature=temperature,
        )

        if _llm_response_failed(response):
            print(f"  [Coder] LLM error: {str(response)[:150]}")
            break

        code = _extract_code(response)
        if not code:
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        issues = _validate_birdnet_code(code) if code else ["No code found in response."]
        if not issues:
            shape_issues = _dry_run_build_head(code)
            if shape_issues:
                issues = shape_issues
        if not issues:
            print("  [Coder] Code valid (shape check passed).")
            return code

        print(f"  [Coder] Issues: {issues}")
        missing_build_head = any("build_head" in i for i in issues)
        build_head_hint = (
            "\n\nCRITICAL: build_head(emb_dim, num_classes) is MISSING from your output. "
            "You MUST define it. It must accept (emb_dim: int, num_classes: int) and return "
            "a tf.keras.Model. Minimal valid example:\n"
            "def build_head(emb_dim, num_classes):\n"
            "    inp = tf.keras.layers.Input(shape=(emb_dim,))\n"
            "    x = tf.keras.layers.Dense(512, activation='gelu')(inp)\n"
            "    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)\n"
            "    return tf.keras.Model(inp, out)\n"
            "Now implement the full architecture from the spec AND include get_training_config()."
        ) if missing_build_head else ""
        shape_fix = ""
        if any("shape" in i.lower() or "incompatible" in i.lower() for i in issues):
            shape_fix = (
                "\n\nSHAPE FIX CHECKLIST:\n"
                "1. First layer after Input: Dense(hidden_dim)(inp) — emb_dim is 1024.\n"
                "2. Every Add()/Multiply(): both inputs are (batch, hidden_dim).\n"
                "3. After Concatenate: Dense(hidden_dim) before Add with x.\n"
            )
        current_prompt = (
            "Your code had issues:\n" + "\n".join(f"- {i}" for i in issues) +
            shape_fix +
            build_head_hint +
            f"\n\nOriginal spec:\n{_spec_to_birdnet_coder_prompt(spec)}\n\n"
            "Fix all issues and return BOTH functions in a single ```python``` code block."
        )

    return None


def _repair_birdnet_code(
    coder_llm: LLMClient, spec: dict, previous_code: str, error: str, temperature: float
) -> str | None:
    """Feed a runtime error back to the coder and ask for a fix."""
    error_tail = error[-1500:] if len(error) > 1500 else error

    # Detect the specific shape-mismatch pattern and inject a targeted hint
    shape_hint = ""
    if "incompatible shapes" in error.lower() or "elemwise_op_output_shape" in error:
        arch_type = spec.get("arch_type", "")
        arch_extra = _arch_type_hint(arch_type)
        shape_hint = (
            "\n*** SHAPE MISMATCH DETECTED — THIS IS THE SPECIFIC BUG IN YOUR CODE ***\n"
            "Add() or Multiply() crashed because two tensors have different feature sizes.\n"
            "Common fixes:\n"
            "  - Missing stem: x = Dense(hidden_dim)(inp) right after Input — inp is 1024-d.\n"
            "  - Residual without projection: Add([x, h]) where x is 1024 and h is 512.\n"
            "  - MoE bug: Concatenate experts (2048) then Multiply with router (512) — use Concat→Dense(hidden_dim) instead.\n"
            "  - DenseNet bug: Add([x, concat]) where concat is wider than x — use Dense(hidden_dim) on concat.\n"
            "THE FIX: Before every Add(), both tensors must be Dense(hidden_dim):\n"
            "    x = tf.keras.layers.Dense(hidden_dim)(inp)   # stem first\n"
            "    h = ... ; h = tf.keras.layers.Dense(hidden_dim)(h)\n"
            "    x = tf.keras.layers.Add()([x, h])\n"
            f"{arch_extra}"
        )

    prompt = (
        f"Your previously generated code failed at runtime with this error:\n"
        f"```\n{error_tail}\n```\n"
        f"{shape_hint}\n"
        f"Architecture spec:\n{_spec_to_birdnet_coder_prompt(spec)}\n\n"
        f"Your previous code:\n```python\n{previous_code}\n```\n\n"
        "Fix the runtime error. General causes to check:\n"
        "- Add() shape mismatch: both inputs must have the EXACT same shape — project with Dense() if needed\n"
        "- MultiHeadAttention requires 3D input: reshape (batch, emb_dim) → (batch, 1, emb_dim) first\n"
        "- Concatenate needs tensors with matching non-concatenation dimensions\n"
        "- Tensor slicing (x[:, :512]) breaks the functional API — use separate Dense layers\n"
        "- Lambda layers may fail on save — replace with explicit Keras layers\n"
        "- build_head must return tf.keras.Model(inp, out), not a layer or tensor\n\n"
        "Return ONLY the corrected code in a ```python``` code block. "
        "Both build_head(emb_dim, num_classes) and get_training_config() must be present."
    )
    response = coder_llm.generate_from_messages(
        messages=[
            {"role": "system", "content": BIRDNET_CODER_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
    )
    if response.startswith("Error communicating"):
        print(f"  [Coder] LLM error during repair: {response[:150]}")
        return None
    code = _extract_code(response)
    if not code:
        lines = response.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    issues = _validate_birdnet_code(code) if code else ["No code found in repair response."]
def _spec_to_birdnet_coder_prompt(spec: dict) -> str:
    arch_type        = spec.get("arch_type", "residual_mlp")
    arch_description = spec.get("arch_description", "(no description provided)")
    hypothesis       = spec.get("hypothesis", "")
    reasoning        = spec.get("reasoning", "")
    strategy         = spec.get("strategy", "explore")

    arch_keys = ("hidden_dim", "proj_dim", "n_layers", "dropout", "activation", "normalization")
    arch_cfg  = {k: spec[k] for k in arch_keys if k in spec}

    training_keys = ("learning_rate", "batch_size", "optimizer", "epochs", "patience", "blend_weight")
    training_cfg  = {k: spec[k] for k in training_keys if k in spec}

    return (
        f"Researcher's experimental proposal:\n"
        f"  arch_type:  {arch_type}\n"
        f"  strategy:   {strategy}\n"
        f"  hypothesis: {hypothesis}\n"
        f"  reasoning:  {reasoning}\n\n"
        f"Architecture description (use as design guidance — implement faithfully but make your own concrete choices):\n"
        f"  {arch_description}\n\n"
        f"Suggested architecture hyperparameters (these are HINTS — adapt as needed for shape safety):\n"
        f"{json.dumps(arch_cfg, indent=2)}\n\n"
        f"Training config to return from get_training_config() (use these values verbatim unless they would break training):\n"
        f"{json.dumps(training_cfg, indent=2)}\n\n"
        f"{_arch_type_hint(arch_type)}"
        f"Implement build_head(emb_dim, num_classes) freely in idiomatic Keras functional API, "
        f"obeying all HARD RULES, the MANDATORY STEM, and Add() SHAPE RULE. "
        f"Return BOTH functions in a single ```python``` code block."
    )


def _extract_code(response: str) -> str | None:
    match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        first = candidate.splitlines()[0].strip().lower() if candidate else ""
        if first in ("python", "py", ""):
            candidate = "\n".join(candidate.splitlines()[1:]).strip() if first else candidate
        return candidate or None
    return None


def _validate_perch_code(code: str) -> list[str]:
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "build_head" not in names:
        issues.append("Missing: build_head(emb_dim, num_classes)")
    if "get_training_config" not in names:
        issues.append("Missing: get_training_config()")
    return issues


def _dry_run_build_head(
    code: str, emb_dim: int = 1536, num_classes: int = 234
) -> list[str]:
    """Instantiate build_head with dummy shapes — catches Add/Concat bugs in seconds."""
    import tensorflow as tf

    namespace: dict = {"tf": tf}
    try:
        exec(code, namespace)  # noqa: S102 — trusted coder slot in isolated process
    except Exception as e:
        return [f"exec failed: {type(e).__name__}: {e}"]
    build_head = namespace.get("build_head")
    if build_head is None:
        return ["build_head not defined after exec"]
    try:
        model = build_head(emb_dim, num_classes)
        if not isinstance(model, tf.keras.Model):
            return ["build_head must return tf.keras.Model(inp, out)"]
        model(tf.zeros((2, emb_dim), dtype=tf.float32), training=False)
    except Exception as e:
        return [f"build_head shape error: {type(e).__name__}: {e}"]
    return []


_ARCH_TYPE_SHAPE_HINTS: dict[str, str] = {
    "mixture_of_experts": (
        "MoE: K parallel Dense(hidden_dim)(x), Concatenate expert outputs, "
        "Dense(hidden_dim) to compress, Add with x. Do NOT softmax-multiply router against concat."
    ),
    "dense_connections": (
        "DenseNet: h = Dense(hidden_dim)(x); dense_in = Concatenate([x, h]); "
        "x = Dense(hidden_dim)(dense_in). Never Add() concat tensor to x directly."
    ),
    "multi_scale_mlp": (
        "Parallel Dense branches → Concatenate → Dense(hidden_dim) before Add with trunk."
    ),
    "bottleneck_mlp": (
        "Dense(wide)→Dense(narrow)→Dense(hidden_dim) before any Add() with x; x must stay hidden_dim."
    ),
    "residual_mlp": (
        "Stem Dense(hidden_dim)(inp) first; each block ends with Dense(hidden_dim) before Add([x, h])."
    ),
    "attention_mlp": "Project to hidden_dim, Reshape (1, H) for MHA, Reshape (H,) back, Add with x.",
    "transformer_block": "Stem to hidden_dim; MHA on (batch, 1, hidden_dim).",
    "multi_tower_ensemble": (
        "Shared stem Dense(hidden_dim)(inp). Build 3–5 parallel towers (different topologies). "
        "Each tower ends with Dense(num_classes, sigmoid). Fuse with Average([t1,t2,...]) or "
        "Concatenate(towers)→Dense(num_classes, sigmoid). All tower outputs must be (batch, num_classes)."
    ),
}


def _build_birdnet_harness_prefix(
    train_cache: Path,
    val_cache: Path,
    head_train_max_samples: int | None = None,
    head_train_indices_path: Path | None = None,
) -> str:
    sub = _harness_subsample_block(head_train_max_samples, head_train_indices_path)
    return f'''
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

_PROJECT_ROOT = None
for _cand in Path(__file__).resolve().parents:
    if (_cand / "src").exists() and (_cand / "configs").exists():
        _PROJECT_ROOT = _cand
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

def _load_cache(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    if "X" in d.files:
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.float32)
    else:
        X = d["X_train"].astype(np.float32)
        y = d["y_train"].astype(np.float32)
    return X, y

_train_cache = Path(r"{train_cache}")
_val_cache   = Path(r"{val_cache}")
X_train, y_train = _load_cache(_train_cache)
X_val, y_val = _load_cache(_val_cache)
{sub}
EMB_DIM   = X_train.shape[1]
N_CLASSES = y_train.shape[1]

print(f"  Loaded train cache: {{_train_cache.name}}")
print(f"  Training head on:   X={{X_train.shape}}  y={{y_train.shape}}")
print(f"  Soundscape val:     X={{X_val.shape}}    y={{y_val.shape}}")
'''.strip()


BIRDNET_HARNESS_SUFFIX = r"""
# build_head(emb_dim, num_classes) and get_training_config() are defined in the slot code above.


def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()

    lr           = float(cfg.get("learning_rate", 8e-4))
    batch_size   = int(cfg.get("batch_size", 256))
    epochs       = int(cfg.get("epochs", 50))
    patience     = int(cfg.get("patience", 7))
    blend_weight = float(cfg.get("blend_weight", 0.2))
    val_split    = float(cfg.get("val_split", 0.1))
    opt_name     = str(cfg.get("optimizer", "adam"))

    # Train/val split on cached embeddings
    n_val   = max(1, int(len(X_train) * val_split))
    perm    = np.random.default_rng(42).permutation(len(X_train))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
    X_vl, y_vl = X_train[val_idx], y_train[val_idx]

    # Positive class weighting — handles severe species imbalance (200:1 neg/pos ratio)
    pos = y_tr.sum(axis=0).astype(np.float64)
    neg = len(y_tr) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
    pw = tf.constant(pos_weight)[tf.newaxis, :]

    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        return tf.reduce_mean(
            pw * y_true * (-tf.math.log(y_pred))
            + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
        )

    # Build head (Coder-generated architecture)
    head = build_head(EMB_DIM, N_CLASSES)

    # Compile
    if opt_name == "sgd_momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif opt_name == "adamw":
        try:
            opt = tf.keras.optimizers.AdamW(lr)
        except AttributeError:
            opt = tf.keras.optimizers.Adam(lr)
    else:
        opt = tf.keras.optimizers.Adam(lr)
    head.compile(optimizer=opt, loss=weighted_bce)

    # Train with early stopping and LR reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True, monitor="val_loss", verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(2, patience // 2), min_lr=1e-6, verbose=0
        ),
    ]
    head.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict on soundscape validation (head output only — no backbone logit blend)
    y_pred = head.predict(X_val, batch_size=batch_size, verbose=0).astype(np.float32)

    # Save trained head so the main loop can promote it if it's the new best
    head.save(str(Path(tempfile.gettempdir()) / "_trained_head.keras"))
    # Also save weights-only file (Keras-version-agnostic, used by Kaggle notebook)
    head.save_weights(str(Path(tempfile.gettempdir()) / "_trained_head.weights.h5"))

    # Save artifacts for the evaluator
    _tmp = Path(tempfile.gettempdir())
    np.save(str(_tmp / "_y_true.npy"), y_val)
    np.save(str(_tmp / "_y_pred.npy"), y_pred)
    print("EVAL_ARTIFACTS_SAVED")


if __name__ == "__main__":
    main()
""".strip()


def _build_birdnet_script(
    slot_code: str,
    train_cache: Path,
    val_cache: Path,
    head_train_max_samples: int | None = None,
    head_train_indices_path: Path | None = None,
) -> str:
    prefix = _build_birdnet_harness_prefix(
        train_cache, val_cache, head_train_max_samples, head_train_indices_path
    )
    return prefix + "\n\n" + slot_code + "\n\n" + BIRDNET_HARNESS_SUFFIX


def _build_final_retrain_script(
    best_code: str,
    mem_dir: Path,
    train_cache: Path,
    val_cache: Path,
) -> str:
    """Build final retrain script using the best iteration's coder-generated build_head + get_training_config."""
    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_MEM_DIR   = Path(r"{mem_dir}")

def _load(p):
    d = np.load(str(p), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)

X_tr, y_tr = _load(Path(r"{train_cache}"))
X_vl, y_vl = _load(Path(r"{val_cache}"))

X_full = np.concatenate([X_tr, X_vl], axis=0)
y_full = np.concatenate([y_tr, y_vl], axis=0)

EMB_DIM   = X_full.shape[1]
N_CLASSES = y_full.shape[1]
print(f"  Final retrain: X={{X_full.shape}}  y={{y_full.shape}}")

# --- Best iteration's build_head + get_training_config (coder-generated) ---
{best_code}
# ---------------------------------------------------------------------------

cfg        = get_training_config()
lr         = float(cfg.get("learning_rate", 8e-4))
batch_size = int(cfg.get("batch_size",    256))
epochs     = int(cfg.get("epochs",         50))
opt_name   = str(cfg.get("optimizer",   "adam"))

head = build_head(EMB_DIM, N_CLASSES)

pos = y_full.sum(axis=0).astype(np.float64)
neg = len(y_full) - pos
pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
pw = tf.constant(pos_weight)[tf.newaxis, :]

def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(
        pw * y_true * (-tf.math.log(y_pred))
        + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
    )

if opt_name == "sgd_momentum":
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
elif opt_name == "adamw":
    try:
        opt = tf.keras.optimizers.AdamW(lr)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(lr)
else:
    opt = tf.keras.optimizers.Adam(lr)
head.compile(optimizer=opt, loss=weighted_bce)

tf.keras.utils.set_random_seed(42)
head.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=1)

head.save(str(_MEM_DIR / "final_head.keras"))
head.save_weights(str(_MEM_DIR / "final_head.weights.h5"))
print("FINAL_RETRAIN_DONE")
""".strip()


def _build_pseudo_refine_script(
    best_code: str,
    mem_dir: Path,
    train_cache: Path,
    pseudo_cache: Path,
    *,
    val_cache: Path | None = None,
    init_weights: Path | None = None,
    fine_tune_lr: float = 2e-4,
    epochs: int = 15,
    val_split: float = 0.1,
    sample_weight_supervised: float = 1.0,
    sample_weight_labeled_val: float = 1.0,
    sample_weight_pseudo: float = 0.5,
) -> str:
    """Fine-tune the 1d head on focal train + optional labeled val + pseudo windows."""
    init_block = ""
    if init_weights is not None and Path(init_weights).exists():
        init_block = f"""
_wpath = Path(r"{init_weights}")
if _wpath.suffix == ".keras" and _wpath.exists():
    head = tf.keras.models.load_model(str(_wpath), compile=False)
    print(f"  Warm-start from {{_wpath.name}}")
else:
    head = build_head(EMB_DIM, N_CLASSES)
    if _wpath.exists():
        head.load_weights(str(_wpath))
        print(f"  Warm-start weights from {{_wpath.name}}")
"""
    else:
        init_block = "\nhead = build_head(EMB_DIM, N_CLASSES)\n"

    val_block = ""
    if val_cache is not None and Path(val_cache).exists():
        val_block = f"""
X_lv, y_lv = _load_focal(Path(r"{val_cache}"))
X_parts.append(X_lv)
y_parts.append(y_lv)
sw_parts.append(np.full(len(X_lv), {sample_weight_labeled_val}, dtype=np.float32))
n_labeled_val = len(X_lv)
"""

    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_MEM_DIR = Path(r"{mem_dir}")

def _load_focal(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    if "X" in d.files:
        X = d["X"]
    elif "X_train" in d.files:
        X = d["X_train"]
    else:
        raise KeyError(f"No X or X_train in {{npz_path}}")
    if "y" in d.files:
        y = d["y"]
    elif "y_train" in d.files:
        y = d["y_train"]
    else:
        raise KeyError(f"No y or y_train in {{npz_path}}")
    return X.astype(np.float32), y.astype(np.float32)

def _load_pseudo(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    if "X_pseudo" not in d.files or "y_pseudo" not in d.files:
        raise KeyError(
            f"Expected X_pseudo and y_pseudo in {{npz_path}}, got {{list(d.files)}}"
        )
    return d["X_pseudo"].astype(np.float32), d["y_pseudo"].astype(np.float32)

X_parts, y_parts, sw_parts = [], [], []
n_focal = n_labeled_val = 0

X_tr, y_tr = _load_focal(Path(r"{train_cache}"))
X_parts.append(X_tr)
y_parts.append(y_tr)
sw_parts.append(np.full(len(X_tr), {sample_weight_supervised}, dtype=np.float32))
n_focal = len(X_tr)
{val_block}
Xp, yp = _load_pseudo(Path(r"{pseudo_cache}"))
if len(Xp) > 0:
    X_parts.append(Xp)
    y_parts.append(yp)
    sw_parts.append(np.full(len(Xp), {sample_weight_pseudo}, dtype=np.float32))
    n_pseudo = len(Xp)
else:
    n_pseudo = 0
    print("  [BirdNET 1e] No pseudo windows in cache — fine-tuning on supervised (+ val) only")

X_all = np.concatenate(X_parts, axis=0)
y_all = np.concatenate(y_parts, axis=0)
sw_all = np.concatenate(sw_parts, axis=0)

EMB_DIM   = X_all.shape[1]
N_CLASSES = y_all.shape[1]
print(
    f"  Pseudo refine: focal={{n_focal}}  labeled_val={{n_labeled_val}}  "
    f"pseudo={{n_pseudo}}  total={{X_all.shape[0]}}"
)

# --- Locked head architecture (from stage 1b / 1d) ---
{best_code}
# ---------------------------------------------------------------------------
{init_block}

pos = y_all.sum(axis=0).astype(np.float64)
neg = len(y_all) - pos
pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
pw = tf.constant(pos_weight)[tf.newaxis, :]

def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(
        pw * y_true * (-tf.math.log(y_pred))
        + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
    )

lr = {fine_tune_lr}
batch_size = int(get_training_config().get("batch_size", 256))
epochs = {epochs}
val_split = {val_split}
opt_name = str(get_training_config().get("optimizer", "adam"))

if opt_name == "sgd_momentum":
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
elif opt_name == "adamw":
    try:
        opt = tf.keras.optimizers.AdamW(lr)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(lr)
else:
    opt = tf.keras.optimizers.Adam(lr)
head.compile(optimizer=opt, loss=weighted_bce)

tf.keras.utils.set_random_seed(42)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_loss", verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]
head.fit(
    X_all, y_all,
    sample_weight=sw_all,
    validation_split=val_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1,
)

head.save(str(_MEM_DIR / "final_head_pseudo.keras"))
head.save_weights(str(_MEM_DIR / "final_head_pseudo.weights.h5"))
print("PSEUDO_REFINE_DONE")
""".strip()


def _ranking_metric_from_config(config: dict) -> str:
    return str(config.get("meta_agent", {}).get("primary_metric", PRIMARY_META_METRIC))


def _ranking_value_from_metrics(metrics: dict | None) -> float | None:
    if not metrics or metrics.get("status") != "success":
        return None
    key = metrics.get("ranking_metric", PRIMARY_META_METRIC)
    if key == "macro_roc_auc":
        return metrics.get("macro_roc_auc")
    return metrics.get("macro_average_precision")


def _format_iteration_metrics(metrics: dict | None) -> str:
    return format_metrics_dict(metrics, ranking_metric=PRIMARY_META_METRIC)


def _slug_slot_suffix(slot: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(slot).strip().lower()).strip("_")
    return (s[:24] or "slot")


def _promote_best_head(
    *,
    rank_val: float | None,
    metrics: dict | None,
    best_score_ever: float,
    iteration: int,
    spec: dict,
    slot_code: str,
    trained_head_path: Path,
    best_head_path: Path,
    mem_dir: Path,
    ranking_metric: str,
    quiet: bool = False,
) -> float:
    if rank_val is None or rank_val <= best_score_ever:
        return best_score_ever
    import shutil

    best_score_ever = float(rank_val)
    ap = metrics.get("macro_average_precision") if metrics else None
    auc = metrics.get("macro_roc_auc") if metrics else None
    med = metrics.get("median_per_class_auc") if metrics else None
    if trained_head_path.exists():
        shutil.copy2(str(trained_head_path), str(best_head_path))
        ws = Path(tempfile.gettempdir()) / "_trained_head.weights.h5"
        if ws.exists():
            shutil.copy2(str(ws), str(mem_dir / "best_head.weights.h5"))
        with open(mem_dir / "best_model_info.json", "w", encoding="utf-8") as f:
            json.dump({
                "ranking_metric": ranking_metric,
                "ranking_value": rank_val,
                "macro_average_precision": ap,
                "macro_roc_auc": auc,
                "median_per_class_auc": med,
                "iteration": iteration,
                "spec": spec,
            }, f, indent=2)
        yp = Path(tempfile.gettempdir()) / "_y_pred.npy"
        yt = Path(tempfile.gettempdir()) / "_y_true.npy"
        if yp.exists():
            shutil.copy2(str(yp), str(mem_dir / "best_val_preds.npy"))
        if yt.exists():
            shutil.copy2(str(yt), str(mem_dir / "y_val.npy"))
        (mem_dir / "best_head_code.py").write_text(slot_code, encoding="utf-8")
        if not quiet:
            print(f"  [Best] NEW BEST {_format_iteration_metrics(metrics)}")
    return best_score_ever


def run_birdnet_fixed_head_train(
    config: dict,
    fixed_cfg: dict,
    *,
    train_cache: Path,
    val_cache: Path,
    mem_dir: Path,
    code_dir: Path,
) -> dict | None:
    """Train a locked head on cached embeddings (stage 1c aug search). No LLM loop."""
    head_path = Path(fixed_cfg["head_code_path"])
    if not head_path.exists():
        raise FileNotFoundError(f"Fixed head code not found: {head_path}")

    head_code = head_path.read_text(encoding="utf-8")
    indices_path = fixed_cfg.get("head_train_indices_path")
    if indices_path:
        indices_path = Path(indices_path)
        if not indices_path.exists():
            raise FileNotFoundError(f"Head train indices not found: {indices_path}")
    else:
        indices_path = None

    py_exe = config.get("execution", {}).get("python_executable", "python3")
    timeout = config.get("execution", {}).get("timeout_seconds", 1800)
    ranking_metric = _ranking_metric_from_config(config)
    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator = Evaluator(row_id_column_name="row_id")

    label = fixed_cfg.get("label", "fixed_head_train")
    quiet = _cfg_quiet(config)
    if not quiet:
        print(f"\n  [Fixed train] {label}")
        print(f"  train cache → {train_cache.name}")
        if indices_path:
            print(f"  head indices → {indices_path.name}")

    script = _build_birdnet_script(
        head_code,
        train_cache,
        val_cache,
        head_train_indices_path=indices_path,
    )
    code_dir.mkdir(parents=True, exist_ok=True)
    script_path = code_dir / f"{label.replace('/', '_')[:60]}.py"
    script_path.write_text(script, encoding="utf-8")

    y_true_path = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path = Path(tempfile.gettempdir()) / "_y_pred.npy"
    trained_head_path = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path = mem_dir / "best_head.keras"

    result = executor.run_file(script_path)
    metrics = None
    if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
        if y_true_path.exists() and y_pred_path.exists():
            summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
            metrics = summary.metrics

    rank_val = _ranking_value_from_metrics(metrics)
    if not quiet:
        print(f"  [Fixed train] {_format_iteration_metrics(metrics)}")

    spec = dict(fixed_cfg.get("spec") or {})
    spec.setdefault("mode", "fixed_head_train")
    spec["aug_preset"] = fixed_cfg.get("aug_preset")

    if rank_val is not None:
        prior_best = -1.0
        info_path = mem_dir / "best_model_info.json"
        if info_path.exists():
            try:
                prev = json.loads(info_path.read_text(encoding="utf-8"))
                prior_best = float(
                    prev.get("ranking_value")
                    or prev.get("macro_average_precision", -1.0)
                )
            except (json.JSONDecodeError, TypeError, ValueError, OSError):
                prior_best = -1.0
        promotion_quiet = quiet or bool(fixed_cfg.get("fixed_1c_trial"))
        _promote_best_head(
            rank_val=rank_val,
            metrics=metrics,
            best_score_ever=prior_best,
            iteration=0,
            spec=spec,
            slot_code=head_code,
            trained_head_path=trained_head_path,
            best_head_path=best_head_path,
            mem_dir=mem_dir,
            ranking_metric=ranking_metric,
            quiet=promotion_quiet,
        )
    elif not result.success and not quiet:
        print(f"  [Fixed train] failed: {(result.stderr or '')[-600:]}")

    return metrics
def _execute_birdnet_iteration(
    iteration: int,
    *,
    researcher: BirdnetResearcher,
    coder_llm: LLMClient,
    coder_temp: float,
    executor: CodeExecutor,
    evaluator: Evaluator,
    memory: BirdnetExperimentMemory,
    train_cache: Path,
    val_cache: Path,
    head_train_max: int | None,
    code_dir: Path,
    spec: dict | None = None,
    slot_suffix: str = "",
) -> tuple[dict | None, str, dict | None]:
    """Coder→train for one spec (researcher may have planned several per round)."""
    if spec is None:
        spec = researcher.next_experiment()
    slot_code = generate_birdnet_code(coder_llm, spec, coder_temp)
    if slot_code is None:
        print("  [Coder] All AST retries exhausted — falling back to safe default residual MLP.")
        slot_code = _SAFE_DEFAULT_SLOT_CODE

    _MAX_EXEC_ATTEMPTS = 5
    metrics = None
    final_slot_code = slot_code
    y_true_path = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path = Path(tempfile.gettempdir()) / "_y_pred.npy"

    for exec_attempt in range(1, _MAX_EXEC_ATTEMPTS + 1):
        script = _build_birdnet_script(
            final_slot_code, train_cache, val_cache, head_train_max_samples=head_train_max
        )
        slot_tag = f"_{_slug_slot_suffix(slot_suffix)}" if slot_suffix else ""
        script_path = code_dir / f"iter_{iteration:03d}{slot_tag}_a{exec_attempt}.py"
        script_path.write_text(script, encoding="utf-8")

        print(f"  [Executor] Attempt {exec_attempt}/{_MAX_EXEC_ATTEMPTS} — {script_path.name} ...")
        result = executor.run_file(script_path)

        if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
            if y_true_path.exists() and y_pred_path.exists():
                summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
                metrics = summary.metrics
            break

        error_msg = (result.stderr or "")[-1500:]
        print(f"  [Executor] Attempt {exec_attempt} failed.")
        if error_msg:
            print(f"  [Error]  {error_msg[-600:]}")

        if exec_attempt == _MAX_EXEC_ATTEMPTS:
            print(f"  [Coder] All {_MAX_EXEC_ATTEMPTS} execution attempts exhausted — skipping slot.")
            break

        repaired = _repair_birdnet_code(coder_llm, spec, final_slot_code, error_msg, coder_temp)
        if repaired is None:
            print("  [Coder] Repair failed — skipping remaining attempts.")
            break
        final_slot_code = repaired

    return metrics, final_slot_code, spec


def _promote_if_best(
    *,
    rank_val: float | None,
    metrics: dict | None,
    spec: dict,
    iteration: int,
    slot_label: str,
    best_score_ever: float,
    trained_head_path: Path,
    best_head_path: Path,
    mem_dir: Path,
    code_dir: Path,
    slot_code: str,
    ranking_metric: str,
) -> float:
    if rank_val is None or rank_val <= best_score_ever:
        return best_score_ever
    import shutil

    best_score_ever = rank_val
    auc = metrics.get("macro_roc_auc") if metrics else None
    ap = metrics.get("macro_average_precision") if metrics else None
    med = metrics.get("median_per_class_auc") if metrics else None
    if trained_head_path.exists():
        shutil.copy2(str(trained_head_path), str(best_head_path))
        _weights_src = Path(tempfile.gettempdir()) / "_trained_head.weights.h5"
        if _weights_src.exists():
            shutil.copy2(str(_weights_src), str(mem_dir / "best_head.weights.h5"))
        with open(mem_dir / "best_model_info.json", "w") as _f:
            json.dump({
                "ranking_metric": ranking_metric,
                "ranking_value": rank_val,
                "macro_average_precision": ap,
                "macro_roc_auc": auc,
                "median_per_class_auc": med,
                "auc": auc,
                "iteration": iteration,
                "slot": slot_label,
                "spec": spec,
            }, _f, indent=2)
        _y_pred_tmp = Path(tempfile.gettempdir()) / "_y_pred.npy"
        _y_true_tmp = Path(tempfile.gettempdir()) / "_y_true.npy"
        if _y_pred_tmp.exists():
            shutil.copy2(str(_y_pred_tmp), str(mem_dir / "best_val_preds.npy"))
        if _y_true_tmp.exists():
            shutil.copy2(str(_y_true_tmp), str(mem_dir / "y_val.npy"))
        (mem_dir / "best_head_code.py").write_text(slot_code, encoding="utf-8")
        print(
            f"  [Best] NEW BEST {_format_iteration_metrics(metrics)} "
            f"(slot={slot_label}) — head saved to {best_head_path.name}"
        )
    return best_score_ever
def _run_birdnet_refine_loop(
    config: dict,
    refine_cfg: dict,
    *,
    researcher: BirdnetResearcher,
    coder_llm: LLMClient,
    coder_temp: float,
    executor: CodeExecutor,
    evaluator: Evaluator,
    memory: BirdnetExperimentMemory,
    train_cache: Path,
    val_cache: Path,
    head_train_max: int | None,
    code_dir: Path,
    mem_dir: Path,
    ranking_metric: str,
    experiments_per_round: int = 1,
) -> None:
    """Adaptive exploit loop: initial training tries, +bonus on improve, hard cap."""
    initial = max(1, int(refine_cfg.get("initial_iterations", 5)))
    bonus = max(1, int(refine_cfg.get("bonus_iterations_on_improve", 5)))
    max_total = max(initial, int(refine_cfg.get("max_iterations_per_model", 25)))
    seed_score = float(refine_cfg.get("seed_score", -1.0))
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    refine_cfg = dict(refine_cfg)
    refine_cfg["ranking_metric"] = ranking_metric
    champion_spec, seed_score, winner = _persist_birdnet_refine_champion_artifacts(
        mem_dir, refine_cfg, ranking_metric=ranking_metric
    )
    refine_cfg["seed_spec"] = champion_spec
    refine_cfg["seed_score"] = seed_score
    refine_cfg["seed_macro_ap"] = winner.get("macro_average_precision")
    refine_cfg["seed_macro_auc"] = winner.get("macro_roc_auc")
    refine_cfg["seed_median_auc"] = winner.get("median_per_class_auc")
    researcher.seed_spec = champion_spec
    researcher.seed_score = seed_score
    researcher.locked_arch_type = str(locked)

    parent_dir = refine_cfg.get("parent_memory_dir")
    if parent_dir and memory.total() == 0:
        memory.seed_from_stage_1a(
            Path(parent_dir),
            arch_type=str(locked),
            aug_baseline=str(refine_cfg.get("aug_baseline", "?")),
            seed_score=seed_score,
            seed_spec=champion_spec,
        )
    memory.sync_refine_champion(
        arch_type=str(locked),
        aug_baseline=str(refine_cfg.get("aug_baseline", "?")),
        seed_score=seed_score,
        seed_spec=champion_spec,
        parent_memory_dir=str(parent_dir) if parent_dir else None,
        champion_source=str(winner.get("source", "?")),
    )

    seed_spec = champion_spec
    seed_ap = refine_cfg.get("seed_macro_ap")
    if seed_ap is None:
        seed_ap = seed_spec.get("macro_average_precision") or refine_cfg.get(
            "macro_average_precision"
        )
    seed_auc = refine_cfg.get("seed_macro_auc")
    if seed_auc is None:
        seed_auc = refine_cfg.get("macro_roc_auc")
    seed_med = refine_cfg.get("seed_median_auc")
    if seed_med is None:
        seed_med = refine_cfg.get("median_per_class_auc")

    print(f"\n  REFINE CAMPAIGN — locked arch_type: {locked}")
    print(f"  Aug baseline: {refine_cfg.get('aug_baseline', '?')}")
    seed_metrics = {
        "status": "success",
        "macro_average_precision": seed_ap,
        "macro_roc_auc": seed_auc,
        "median_per_class_auc": seed_med,
        "ranking_metric": ranking_metric,
    }
    print(f"  Champion to beat: {format_metrics_dict(seed_metrics, ranking_metric=ranking_metric)}")
    epr_note = (
        f", {experiments_per_round} experiments per planner call"
        if experiments_per_round > 1
        else ""
    )
    print(
        f"  Budget: {initial} initial training tries{epr_note}, +{bonus} on each improvement, "
        f"max {max_total} total training runs"
    )
    print(f"  Champion spec → {mem_dir / REFINE_CHAMPION_SPEC_FILE}")

    best_score_ever = max(seed_score, -1.0)
    _prior = memory.best_runs(1)
    if _prior:
        best_score_ever = max(best_score_ever, memory._ranking_value(_prior[0]))

    tries_left = initial
    total_done = 0
    run_index = 0
    planner_round = 0
    trained_head_path = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path = mem_dir / "best_head.keras"

    while total_done < max_total and tries_left > 0:
        planner_round += 1
        print(f"\n{'─'*60}")
        if experiments_per_round > 1:
            print(
                f"  REFINE PLANNER ROUND {planner_round}  "
                f"(training runs {total_done}/{max_total}, queue {tries_left})"
            )
        else:
            print(
                f"  REFINE {planner_round}  (done {total_done}/{max_total}, "
                f"queue {tries_left} remaining)"
            )
        print(f"{'─'*60}")

        specs = researcher.next_experiments()
        for slot_i, spec in enumerate(specs, 1):
            if tries_left <= 0 or total_done >= max_total:
                break

            run_index += 1
            total_done += 1
            tries_left -= 1
            slot_label = str(spec.get("slot") or f"s{slot_i}")
            if experiments_per_round > 1:
                print(f"\n  ▸ Try {slot_i}/{len(specs)}: {slot_label}")

            metrics, slot_code, spec = _execute_birdnet_iteration(
                run_index,
                spec=spec,
                slot_suffix=slot_label,
                researcher=researcher,
                coder_llm=coder_llm,
                coder_temp=coder_temp,
                executor=executor,
                evaluator=evaluator,
                memory=memory,
                train_cache=train_cache,
                val_cache=val_cache,
                head_train_max=head_train_max,
                code_dir=code_dir,
            )

            rank_val = _ranking_value_from_metrics(metrics)
            print(f"  [Result] [{slot_label}] {_format_iteration_metrics(metrics)}")
            memory.log(spec=spec, metrics=metrics, code=slot_code)

            if rank_val is not None and rank_val > best_score_ever:
                prev = best_score_ever
                best_score_ever = _promote_best_head(
                    rank_val=rank_val,
                    metrics=metrics,
                    best_score_ever=best_score_ever,
                    iteration=run_index,
                    spec=spec,
                    slot_code=slot_code,
                    trained_head_path=trained_head_path,
                    best_head_path=best_head_path,
                    mem_dir=mem_dir,
                    ranking_metric=ranking_metric,
                )
                bonus_add = min(bonus, max_total - total_done)
                tries_left += bonus_add
                print(
                    f"  [Refine] Improved {ranking_metric} {prev:.5f} → {best_score_ever:.5f} "
                    f"| +{bonus_add} bonus tries (queue={tries_left})"
                )
            elif rank_val is not None:
                print(
                    f"  [Refine] No improvement (best {ranking_metric}={best_score_ever:.5f})"
                )

        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] {memory._format_run_score(best[0])}")

    print(
        f"\n  [Refine] Finished: {total_done} training runs, "
        f"best {ranking_metric}={best_score_ever:.5f}"
    )


def _persist_birdnet_refine_champion_artifacts(
    mem_dir: Path, refine_cfg: dict, *, ranking_metric: str
) -> tuple[dict, float, dict]:
    """
    Resolve best model from experiment memories, write refine_champion_spec.json,
    and copy winning head artifacts into the refine memory dir.
    """
    import shutil

    mem_dir.mkdir(parents=True, exist_ok=True)
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    winner = _resolve_refine_champion(mem_dir, refine_cfg, ranking_metric=ranking_metric)
    seed_spec = dict(winner.get("spec") or {})
    seed_score = float(winner.get("ranking_value", refine_cfg.get("seed_score", -1.0)))
    source = str(winner.get("source", "?"))
    winner_mem = Path(winner.get("memory_dir") or mem_dir)

    if winner_mem.is_dir() and winner_mem != mem_dir:
        info_path = winner_mem / "best_model_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                for k, v in (info.get("spec") or {}).items():
                    if v is not None and k not in seed_spec:
                        seed_spec[k] = v
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists():
                shutil.copy2(src, mem_dir / name)
    elif winner_mem.is_dir():
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists() and not (mem_dir / name).exists():
                shutil.copy2(src, mem_dir / name)

    seed_spec["arch_type"] = locked
    if not seed_spec.get("arch_description"):
        seed_spec["arch_description"] = _synthesize_arch_description(seed_spec)

    payload = {
        "locked_arch_type": locked,
        "aug_baseline": refine_cfg.get("aug_baseline"),
        "champion_score": seed_score,
        "ranking_metric": ranking_metric,
        "source": source,
        "winner_memory_dir": str(winner_mem),
        "parent_memory_dir": refine_cfg.get("parent_memory_dir"),
        "macro_average_precision": winner.get("macro_average_precision"),
        "macro_roc_auc": winner.get("macro_roc_auc"),
        "median_per_class_auc": winner.get("median_per_class_auc"),
        "spec": seed_spec,
    }
    (mem_dir / REFINE_CHAMPION_SPEC_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"  [Refine] Champion from {source}: {ranking_metric}={seed_score:.5f} "
        f"({locked})"
    )
    return seed_spec, seed_score, winner
def _resolve_birdnet_refine_champion(
    mem_dir: Path,
    refine_cfg: dict,
    *,
    ranking_metric: str,
) -> dict:
    """
    Scan refine + parent experiment memories and pick the best successful run
  (same locked arch_type). Beats relying only on the 1a handoff score.
    """
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )
    candidates: list[dict] = []

    handoff_spec = dict(refine_cfg.get("seed_spec") or {})
    handoff_score = float(refine_cfg.get("seed_score", -1.0))
    if handoff_spec:
        candidates.append(
            {
                "spec": handoff_spec,
                "ranking_value": handoff_score,
                "macro_average_precision": refine_cfg.get("seed_macro_ap"),
                "macro_roc_auc": refine_cfg.get("seed_macro_auc"),
                "median_per_class_auc": refine_cfg.get("seed_median_auc"),
                "source": "stage_1a_handoff",
                "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
            }
        )

    seen_dirs: set[str] = set()
    for raw in (refine_cfg.get("parent_memory_dir"), str(mem_dir)):
        if not raw:
            continue
        d = Path(raw)
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        best = _best_run_from_memory_dir(
            d, ranking_metric=ranking_metric, locked_arch_type=str(locked)
        )
        if best is None:
            continue
        spec = dict(best.get("spec") or {})
        val = float(best.get("_ranking_value", _ranking_value_from_memory_entry(best, ranking_metric)))
        candidates.append(
            {
                "spec": spec,
                "ranking_value": val,
                "macro_average_precision": best.get("macro_average_precision"),
                "macro_roc_auc": best.get("macro_roc_auc"),
                "median_per_class_auc": best.get("median_per_class_auc"),
                "source": f"memory:{d.name}",
                "memory_dir": best.get("_source_memory_dir", str(d)),
            }
        )

    if not candidates:
        return {
            "spec": handoff_spec or {"arch_type": locked},
            "ranking_value": handoff_score,
            "source": "default",
            "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
        }

    winner = max(candidates, key=lambda c: float(c.get("ranking_value", -1.0)))
    return winner


def _load_refine_champion_spec(mem_dir: Path) -> dict | None:
    """Canonical champion written at refine start (full spec + metadata)."""
    mem_dir = Path(mem_dir)
    for name in (REFINE_CHAMPION_SPEC_FILE, LEGACY_CHAMPION_SPEC_FILE):
        path = mem_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        spec = dict(payload.get("spec") or {})
        if payload.get("locked_arch_type"):
            spec["arch_type"] = payload["locked_arch_type"]
        return spec or None
    return None


def _mem_dir(config: dict) -> Path:
    return Path((config.get("birdnet") or {})["memory_dir"])


def _cache_dir(config: dict) -> Path:
    bn = config.get("birdnet") or {}
    return Path(bn.get("cache_dir", ROOT / "logs" / "meta_agent" / "birdnet_cache"))


def _code_dir(config: dict) -> Path:
    bn = config.get("birdnet") or {}
    return Path(bn.get("code_dir", _mem_dir(config) / "codes"))


def _val_cache_path(config: dict) -> Path:
    bn = config.get("birdnet") or {}
    return Path(bn.get("val_cache_path", _cache_dir(config) / "val_emb.npz"))


def _train_cache_path(config: dict, preset: str) -> Path:
    custom = config.get("train_cache_path")
    if custom:
        return Path(custom)
    return _cache_dir(config) / f"train_emb_{preset}.npz"


def _save_birdnet_train_cache(
    out_path: Path, X: np.ndarray, y: np.ndarray, *, preset: str, sample_frac: float
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        meta_json=np.array(
            [
                json.dumps(
                    {
                        "backbone": "birdnet",
                        "aug_preset": preset,
                        "sample_frac": sample_frac,
                        "n_samples": len(X),
                        "embedding_dim": int(X.shape[1]),
                    }
                )
            ],
            dtype=object,
        ),
    )


def _ensure_birdnet_val_cache(config: dict, val_path: Path) -> None:
    val_path.parent.mkdir(parents=True, exist_ok=True)
    if val_path.exists() and not config.get("force_rebuild_cache"):
        return
    init_birdnet()
    _, species_to_idx = _build_species_map()
    from birdnet_agent import NPZ_VAL, build_val_cache

    if not NPZ_VAL.exists() or config.get("force_rebuild_cache"):
        build_val_cache(species_to_idx)
    d = np.load(str(NPZ_VAL), allow_pickle=True)
    np.savez_compressed(
        val_path,
        X=d["X_val"].astype(np.float32),
        y=d["y_val"].astype(np.float32),
        row_ids=d["row_ids"],
    )


def setup_birdnet_caches(config: dict) -> tuple[Path, Path]:
    preset = str(config.get("meta_aug_preset", "medium"))
    for d in (_mem_dir(config), _cache_dir(config), _code_dir(config)):
        d.mkdir(parents=True, exist_ok=True)
    train_path = _train_cache_path(config, preset)
    val_path = _val_cache_path(config)
    _ensure_birdnet_val_cache(config, val_path)
    if not train_path.exists() or config.get("force_rebuild_cache"):
        X_train, y_train, _, _ = ensure_caches(config)
        frac = float(config.get("train_sample_frac", 0.5))
        _save_birdnet_train_cache(train_path, X_train, y_train, preset=preset, sample_frac=frac)
    return train_path, val_path


def _cfg_quiet(config: dict) -> bool:
    return bool(config.get("birdnet_quiet") or (config.get("birdnet") or {}).get("quiet_trial"))


def _resolve_experiments_per_round(config: dict, *, refine_mode: bool) -> int:
    if refine_mode:
        br = config.get("birdnet_refine") or {}
        if br.get("experiments_per_researcher_call") is not None:
            return int(br["experiments_per_researcher_call"])
    return int(config.get("researcher", {}).get("batch_size", 3))


def run_birdnet_explore(config: dict) -> None:
    train_cache, val_cache = setup_birdnet_caches(config)
    mem_dir, code_dir = _mem_dir(config), _code_dir(config)
    ranking_metric = _ranking_metric_from_config(config)
    researcher_llm = LLMClient(
        provider=config.get("llm", {}).get("provider", "ollama"),
        model=config.get("researcher", {}).get("model", "deepseek-r1:8b"),
        timeout_seconds=_resolve_researcher_timeout_seconds(config),
    )
    coder_llm = LLMClient(provider=config.get("llm", {}).get("provider", "ollama"), model=config.get("llm", {}).get("model", "qwen2.5-coder:7b"))
    memory = BirdnetExperimentMemory(mem_dir, ranking_metric=ranking_metric)
    researcher = BirdnetResearcher(researcher_llm, memory, temperature=float(config.get("researcher", {}).get("temperature", 0.2)),
        batch_size=_resolve_experiments_per_round(config, refine_mode=False))
    executor = CodeExecutor(python_executable=config.get("execution", {}).get("python_executable", "python3"),
        timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 1800)))
    evaluator = Evaluator(row_id_column_name="row_id")
    max_iterations = int(config.get("max_iterations", 3))
    head_train_max = config.get("head_train_max_samples")
    if head_train_max is not None:
        head_train_max = int(head_train_max)
    print("\n" + "=" * 60 + "\n  BirdNET staged explore (1024-d embeddings)\n" + "=" * 60)
    best = memory.best_runs(1)
    best_score = memory._ranking_value(best[0]) if best else -1.0
    trained = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head = mem_dir / "best_head.keras"
    for it in range(1, max_iterations + 1):
        print(f"\n--- planner round {it}/{max_iterations} ---")
        for slot_i, spec in enumerate(researcher.next_experiments(), 1):
            metrics, code, spec = _execute_birdnet_iteration(it, spec=spec, slot_suffix=str(spec.get("slot", slot_i)),
                researcher=researcher, coder_llm=coder_llm, coder_temp=float(config.get("llm", {}).get("temperature", 0.2)),
                executor=executor, evaluator=evaluator, memory=memory, train_cache=train_cache, val_cache=val_cache,
                head_train_max=head_train_max, code_dir=code_dir)
            memory.log(spec=spec, metrics=metrics, code=code)
            rv = _ranking_value_from_metrics(metrics)
            print(f"  {_format_iteration_metrics(metrics)}")
            best_score = _promote_if_best(rank_val=rv, metrics=metrics, spec=spec, iteration=it, slot_label=str(spec.get("slot", slot_i)),
                best_score_ever=best_score, trained_head_path=trained, best_head_path=best_head, mem_dir=mem_dir, code_dir=code_dir,
                slot_code=code, ranking_metric=ranking_metric)


def dispatch_birdnet_staged(config: dict) -> None:
    if config.get("birdnet_build_cache_only"):
        setup_birdnet_caches(config)
        return
    if (config.get("birdnet_refine") or {}).get("enabled"):
        train_cache, val_cache = setup_birdnet_caches(config)
        mem_dir, code_dir = _mem_dir(config), _code_dir(config)
        refine_cfg = dict(config.get("birdnet_refine") or {})
        ranking_metric = _ranking_metric_from_config(config)
        researcher_llm = LLMClient(provider=config.get("llm", {}).get("provider", "ollama"), model=config.get("researcher", {}).get("model", "deepseek-r1:8b"),
            timeout_seconds=_resolve_researcher_timeout_seconds(config))
        coder_llm = LLMClient(provider=config.get("llm", {}).get("provider", "ollama"), model=config.get("llm", {}).get("model", "qwen2.5-coder:7b"))
        memory = BirdnetExperimentMemory(mem_dir, ranking_metric=ranking_metric)
        locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get("arch_type", "residual_mlp")
        researcher = BirdnetResearcher(researcher_llm, memory, refine_mode=True, locked_arch_type=str(locked),
            seed_spec=dict(refine_cfg.get("seed_spec") or {}), seed_score=refine_cfg.get("seed_score"),
            batch_size=int(refine_cfg.get("experiments_per_researcher_call") or 3))
        executor = CodeExecutor(python_executable=config.get("execution", {}).get("python_executable", "python3"),
            timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 1800)))
        evaluator = Evaluator(row_id_column_name="row_id")
        hmax = config.get("head_train_max_samples")
        _run_birdnet_refine_loop(config, refine_cfg, researcher=researcher, coder_llm=coder_llm,
            coder_temp=float(config.get("llm", {}).get("temperature", 0.2)), executor=executor, evaluator=evaluator,
            memory=memory, train_cache=train_cache, val_cache=val_cache, head_train_max=int(hmax) if hmax else None,
            code_dir=code_dir, mem_dir=mem_dir, ranking_metric=ranking_metric, experiments_per_round=researcher.batch_size)
        return
    if (config.get("birdnet_fixed_train") or {}).get("enabled"):
        train_cache, val_cache = setup_birdnet_caches(config)
        run_birdnet_fixed_head_train(config, config.get("birdnet_fixed_train") or {}, train_cache=train_cache,
            val_cache=val_cache, mem_dir=_mem_dir(config), code_dir=_code_dir(config))
        return
    if config.get("birdnet_staged"):
        run_birdnet_explore(config)
