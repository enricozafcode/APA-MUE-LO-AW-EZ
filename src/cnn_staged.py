"""
Perch-style staged CNN pipeline: researcher + coder (LLM architecture code),
explore → refine → augmentation search → full final train → pseudo-label refine.

Entry: ``dispatch_cnn_staged(config)`` from ``cnn_agent.agent_loop`` when ``cnn_staged`` is set.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from memory import ExperimentMemory

try:
    from .cnn_agent import (
        GENERATION_SYSTEM_PROMPT,
        SAFE_BASELINE_SLOT_CODE,
        _append_eval_wrapper,
        _locked_cnn_aug,
        _ranking_metric_from_config,
        _ranking_value_from_metrics,
        assemble_script,
        extract_python_code,
        run_experiment_until_success,
        validate_slot_code,
    )
    from .cnn_focal_cache import ensure_focal_train_cache
    from .cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR, ensure_soundscape_mel_cache
    from .soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict
except ImportError:
    from cnn_agent import (
        GENERATION_SYSTEM_PROMPT,
        SAFE_BASELINE_SLOT_CODE,
        _append_eval_wrapper,
        _locked_cnn_aug,
        _ranking_metric_from_config,
        _ranking_value_from_metrics,
        assemble_script,
        extract_python_code,
        run_experiment_until_success,
        validate_slot_code,
    )
    from cnn_focal_cache import ensure_focal_train_cache
    from cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR, ensure_soundscape_mel_cache
    from soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict

ROOT = Path(__file__).resolve().parents[1]
CNN_BATCH_SLOTS = ("tweak", "explore", "free")
REFINE_CHAMPION_SPEC_FILE = "refine_champion_spec.json"
STAGED_RESULTS_FILE = "staged_results.json"

CNN_SEARCH_SPACE = {
    "arch_type": [
        "shallow_cnn",
        "deep_cnn",
        "residual_cnn",
        "separable_cnn",
        "multi_scale_cnn",
        "dilated_cnn",
        "attention_pool_cnn",
        "bottleneck_cnn",
    ],
    "depth": [2, 3, 4, 5, 6, 8],
    "filters_base": [16, 32, 64, 128],
    "filter_pattern": ["doubling", "fixed", "pyramid"],
    "pooling_type": ["global_avg", "global_max", "flatten"],
    "classifier_hidden_units": [0, 128, 256, 512],
    "dropout": [0.0, 0.2, 0.3, 0.5],
    "batch_norm": [True, False],
    "residuals": [True, False],
    "learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32, 64],
    "optimizer": ["adam", "sgd_momentum"],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "epochs": [3, 5, 10, 15, 20],
    "n_mels": [64, 128],
    "n_frames": [128, 256],
}

CNN_RESEARCHER_SYSTEM = """You are an expert ML researcher designing BirdCLEF CNN models on log-mel spectrograms.
You propose architectures as JSON — you NEVER write Python code.

Goal: discover diverse CNN topologies (not just hyperparameter nudging). Each experiment must include
arch_description: 3–6 sentences describing the Conv2D stack, pooling, classifier head, and compile settings.

Custom arch_type names are allowed (snake_case). The coder implements from arch_description + hypers.

Ranking metric is macro_average_precision on labeled train_soundscapes (noisy jungle audio, rare species).

Output ONLY valid JSON — start with { and end with }."""

CNN_EXPLORE_ADDENDUM = """
STAGE 1a — EXPLORE CNN ARCHITECTURE SPACE:
- Cover different structural families (depth, separable convs, residuals, multi-scale branches, attention pooling).
- Tweaking a promising family is OK; repeating identical failed configs is not.
- strategy "explore" = new family or major structural change; "exploit" = refine within a family.
"""

CNN_BATCH_ADDENDUM = """
BATCH — return exactly {batch_size} experiments in ONE JSON:
{{"planner_note": "...", "experiments": [{{"slot": "tweak", ...}}, ...]}}
Slots (batch_size=3): tweak | explore | free — each needs arch_description.
Required keys per experiment: slot, arch_type, arch_description, depth, filters_base, filter_pattern,
pooling_type, classifier_hidden_units, dropout, batch_norm, residuals, learning_rate, batch_size,
optimizer, weight_decay, epochs, n_mels, n_frames, reasoning, hypothesis, strategy.
"""

CNN_REFINE_ADDENDUM = """
REFINE MODE — locked arch_type={locked}. All experiments must use arch_type="{locked}" and strategy="exploit".
Vary layout/hypers within the family only. Short JSON.
"""

CNN_CODER_SYSTEM = (
    GENERATION_SYSTEM_PROMPT
    + "\n\nYou implement the researcher's arch_description as a full Keras CNN on mel input "
    "(batch, n_mels, n_frames, 1). Use only tf.keras layers. "
    "get_training_config() must return epochs, batch_size, learning_rate, optimizer, n_mels, n_frames, "
    "val_split, and optional aug_prob/aug_noise_std/aug_time_mask/aug_freq_mask/aug_preset.\n"
    "Keras 3 rules:\n"
    "- Prefer Sequential with Input(shape=...) as the first layer.\n"
    "- For residual/skip connections use the Functional API (Input → branches → Add → head), "
    "NOT model.add(layers.Add()([shortcut, model])).\n"
    "- If unsure about residuals, set residuals=False and use a plain Conv block stack."
)


def _llm_failed(response: str) -> bool:
    return not response or response.startswith("Error communicating")


def _fill_cnn_defaults(spec: dict) -> dict:
    defaults = {
        "arch_type": "shallow_cnn",
        "arch_description": (
            "Three Conv2D blocks (32→64→128 filters) with ReLU, MaxPool after each, "
            "GlobalAveragePooling, Dropout 0.3, Dense(num_classes, sigmoid). Adam lr=1e-3."
        ),
        "depth": 3,
        "filters_base": 32,
        "filter_pattern": "doubling",
        "pooling_type": "global_avg",
        "classifier_hidden_units": 256,
        "dropout": 0.3,
        "batch_norm": True,
        "residuals": False,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "epochs": 10,
        "n_mels": 64,
        "n_frames": 128,
        "reasoning": "Safe default CNN.",
        "hypothesis": "Baseline mel-CNN should train.",
        "strategy": "explore",
    }
    out = dict(defaults)
    out.update(spec)
    return out


def _spec_to_coder_prompt(spec: dict) -> str:
    lines = [
        "Implement this BirdCLEF CNN experiment:",
        f"arch_type: {spec.get('arch_type')}",
        f"arch_description:\n{spec.get('arch_description', '')}",
        "",
        "Hyperparameters for get_training_config():",
    ]
    for k in (
        "depth", "filters_base", "filter_pattern", "pooling_type",
        "classifier_hidden_units", "dropout", "batch_norm", "residuals",
        "learning_rate", "batch_size", "optimizer", "weight_decay", "epochs",
        "n_mels", "n_frames",
    ):
        if k in spec:
            lines.append(f"  {k}: {spec[k]}")
    locked = _locked_cnn_aug(spec) if isinstance(spec.get("_config"), dict) else {}
    if not locked and spec.get("aug_preset"):
        lines.append(f"  aug_preset: {spec['aug_preset']}")
    lines.append(
        "\nReturn ONLY one ```python``` block with get_training_config() and "
        "build_model(input_shape, num_classes). Final layer: sigmoid, loss: binary_crossentropy."
    )
    return "\n".join(lines)


def generate_cnn_slot_code(
    coder_llm: LLMClient,
    spec: dict,
    temperature: float,
    max_retries: int = 5,
    *,
    seed_slot_code: str | None = None,
) -> str | None:
    prompt = _spec_to_coder_prompt(spec)
    if seed_slot_code:
        prompt += (
            "\n\nRefine mode: start from this working 1a slot. "
            "Keep build_model structure stable; only adjust hyperparameters in get_training_config() "
            "unless the arch_description requires a minimal safe fix.\n"
            f"```python\n{seed_slot_code.strip()[:12000]}\n```"
        )
    current = prompt
    for attempt in range(1, max_retries + 1):
        print(f"  [Coder] Attempt {attempt}/{max_retries}...")
        response = coder_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": CNN_CODER_SYSTEM},
                {"role": "user", "content": current},
            ],
            temperature=temperature,
        )
        if _llm_failed(response):
            print(f"  [Coder] LLM error: {str(response)[:150]}")
            break
        code = extract_python_code(response)
        if not code and response.strip():
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()
        issues = validate_slot_code(code) if code else ["No code found."]
        if not issues:
            print("  [Coder] Code validated.")
            return code
        print(f"  [Coder] Issues: {issues}")
        current = (
            "Fix these issues:\n" + "\n".join(f"- {i}" for i in issues)
            + f"\n\nOriginal request:\n{prompt}\n\nReturn corrected ```python``` only."
        )
    return None


def _parse_json_root(text: str) -> Any:
    _open, _close = "<" + "think" + ">", "</" + "think" + ">"
    cleaned = re.sub(
        re.escape(_open) + r"[\s\S]*?" + re.escape(_close),
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m2 = re.search(r"\{[\s\S]*\}", cleaned)
        if m2:
            try:
                return json.loads(m2.group())
            except json.JSONDecodeError:
                pass
    return None


def _normalize_experiment(item: dict) -> dict:
    spec = _fill_cnn_defaults(item)
    if not str(spec.get("arch_description", "")).strip():
        spec["arch_description"] = (
            f"{spec['arch_type']}: {spec['depth']} conv blocks, filters_base={spec['filters_base']}, "
            f"{spec['pooling_type']} pooling, classifier {spec['classifier_hidden_units']} units."
        )
    return spec


class CnnResearcher:
    def __init__(
        self,
        llm: LLMClient,
        memory: ExperimentMemory,
        temperature: float = 0.6,
        *,
        refine_mode: bool = False,
        locked_arch_type: str | None = None,
        seed_spec: dict | None = None,
        batch_size: int = 3,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.locked_arch_type = locked_arch_type
        self.seed_spec = seed_spec or {}
        self.batch_size = max(1, int(batch_size))

    def next_experiments(self) -> list[dict]:
        history = self.memory.researcher_context()
        best = self.memory.best_runs(1)
        total = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"
        batch_size = self.batch_size

        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "shallow_cnn")
            system = CNN_RESEARCHER_SYSTEM + "\n\n" + CNN_REFINE_ADDENDUM.format(locked=locked)
            user = (
                f"{history}\n\nSearch space hints:\n{json.dumps(CNN_SEARCH_SPACE, indent=2)}\n\n"
                f"Refine runs: {total} | best: {best_str}\n"
                f"Propose exactly {batch_size} refine experiments (arch_type={locked}). "
                'JSON: {{"experiments": [...]}}'
            )
        elif batch_size > 1:
            system = (
                CNN_RESEARCHER_SYSTEM + "\n\n" + CNN_EXPLORE_ADDENDUM + "\n\n"
                + CNN_BATCH_ADDENDUM.format(batch_size=batch_size)
            )
            user = (
                f"{history}\n\nSearch space:\n{json.dumps(CNN_SEARCH_SPACE, indent=2)}\n\n"
                f"Total runs: {total} | best: {best_str}\n"
                f"Propose exactly {batch_size} experiments (slots: {', '.join(CNN_BATCH_SLOTS[:batch_size])}). "
                'JSON: {{"planner_note":"...", "experiments":[...]}}'
            )
        else:
            system = CNN_RESEARCHER_SYSTEM + "\n\n" + CNN_EXPLORE_ADDENDUM
            user = (
                f"{history}\n\nSearch space:\n{json.dumps(CNN_SEARCH_SPACE, indent=2)}\n\n"
                f"Total: {total} | best: {best_str}\nOne experiment JSON with arch_description."
            )

        print(f"\n  [Researcher] Planning {batch_size} experiment(s) ({total} in memory)...")
        response = self.llm.generate_from_messages(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        if _llm_failed(response):
            specs = self._fallback_batch(batch_size)
        else:
            specs = self._parse_batch(response, batch_size)
        if self.refine_mode and self.locked_arch_type:
            for i, spec in enumerate(specs):
                spec["arch_type"] = self.locked_arch_type
                spec["strategy"] = "exploit"
                spec.setdefault("slot", f"r{i + 1}")
        return specs

    def _parse_batch(self, response: str, batch_size: int) -> list[dict]:
        root = _parse_json_root(response)
        items: list[dict] = []
        if isinstance(root, dict):
            exps = root.get("experiments")
            if isinstance(exps, list):
                items = [e for e in exps if isinstance(e, dict)]
        elif isinstance(root, list):
            items = [e for e in root if isinstance(e, dict)]
        if not items and isinstance(root, dict) and root.get("arch_type"):
            items = [root]
        if not items:
            print("  [Researcher] Parse failed — using fallbacks.")
            return self._fallback_batch(batch_size)
        specs = [_normalize_experiment(x) for x in items[:batch_size]]
        while len(specs) < batch_size:
            specs.append(_normalize_experiment(self._fallback_batch(1)[0]))
        for i, spec in enumerate(specs):
            if not spec.get("slot"):
                spec["slot"] = CNN_BATCH_SLOTS[i] if i < len(CNN_BATCH_SLOTS) else f"s{i + 1}"
        return specs

    def _fallback_batch(self, batch_size: int) -> list[dict]:
        templates = [
            {"slot": "tweak", "arch_type": "deep_cnn", "strategy": "exploit"},
            {"slot": "explore", "arch_type": "residual_cnn", "strategy": "explore"},
            {"slot": "free", "arch_type": "separable_cnn", "strategy": "explore"},
        ]
        out = []
        for i in range(batch_size):
            t = templates[i % len(templates)]
            s = _fill_cnn_defaults(t)
            s["reasoning"] = "Researcher fallback."
            out.append(s)
        return out


def _resolve_dirs(config: dict) -> dict[str, Path]:
    root = ROOT
    cnn_cfg = config.get("cnn", {})
    logs = Path(cnn_cfg["logs_dir"]) if cnn_cfg.get("logs_dir") else root / "logs" / "meta_agent" / "cnn"
    mem_dir = Path(cnn_cfg["memory_dir"]) if cnn_cfg.get("memory_dir") else logs
    code_dir = Path(cnn_cfg["code_dir"]) if cnn_cfg.get("code_dir") else mem_dir / "codes"
    eval_dir = mem_dir / "eval_artifacts"
    for d in (mem_dir, code_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"logs": logs, "mem_dir": mem_dir, "code_dir": code_dir, "eval_dir": eval_dir}


def _cheap_training_overrides(config: dict) -> dict[str, Any]:
    from cnn_focal_cache import focal_clip_seed_from_config

    sc = config.get("search", {})
    cheap = sc.get("cheap", {})
    meta = config.get("meta_agent", {})
    max_samples = meta.get("arch_search_cnn_max_samples", cheap.get("max_samples", 2000))
    return {
        "max_samples": max_samples,
        "epochs": cheap.get("epochs", 3),
        "val_split": cheap.get("val_split", 0.2),
        "focal_clip_seed": focal_clip_seed_from_config(config),
    }


def _enforced_training_overrides(config: dict, spec: dict | None = None) -> dict[str, Any]:
    """Merge cheap caps + locked aug; refine also locks mel shape from 1a winner."""
    ov = dict(_cheap_training_overrides(config))
    locked = _locked_cnn_aug(config)
    if locked:
        ov.update({k: v for k, v in locked.items() if k != "aug_preset"})
        if locked.get("aug_preset"):
            ov["aug_preset"] = locked["aug_preset"]
    refine = config.get("cnn_refine") or {}
    if refine.get("enabled"):
        seed = dict(refine.get("seed_spec") or {})
        for key in ("n_mels", "n_frames", "sample_rate", "clip_seconds"):
            if key in seed:
                ov[key] = seed[key]
    if spec:
        # Spec aug fields from researcher (architecture knobs stay in generated code).
        for key in (
            "aug_prob",
            "aug_noise_std",
            "aug_time_mask",
            "aug_freq_mask",
            "aug_preset",
        ):
            if key in spec and key not in ov:
                ov[key] = spec[key]
    return ov


def merge_aug_into_slot(slot_code: str, aug_dict: dict) -> str:
    """Append get_training_config override for stage 1c / final."""
    aug_repr = repr(dict(aug_dict))
    return (
        slot_code.strip()
        + f"\n\n{_META_AUG_MARKER}\n"
        f"_META_AUG = {aug_repr}\n\n"
        f"def get_training_config():\n"
        f"    cfg = _ORIG_GET_TRAINING_CONFIG()\n"
        f"    cfg.update(_META_AUG)\n"
        f"    return cfg\n"
    )


def inject_aug_override(slot_code: str, aug_dict: dict) -> str:
    """Rename original get_training_config and add override."""
    base = _strip_slot_meta_overrides(slot_code)
    if "def get_training_config" not in base:
        return base
    renamed = re.sub(
        r"^def get_training_config\s*\(",
        "def _ORIG_GET_TRAINING_CONFIG(",
        base,
        count=1,
        flags=re.MULTILINE,
    )
    return merge_aug_into_slot(renamed, aug_dict)


def _append_results(mem_dir: Path, entry: dict) -> None:
    path = mem_dir / STAGED_RESULTS_FILE
    rows: list[dict] = []
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(entry)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _format_metrics(metrics: dict | None) -> str:
    return format_metrics_dict(metrics, ranking_metric=PRIMARY_META_METRIC)


def _promote_cnn_best(
    *,
    rank_val: float | None,
    metrics: dict | None,
    spec: dict,
    iteration: int,
    slot_label: str,
    best_score_ever: float,
    mem_dir: Path,
    slot_code: str,
    eval_dir: Path,
    run_id: str,
    ranking_metric: str,
) -> float:
    if rank_val is None or rank_val <= best_score_ever:
        return best_score_ever
    best_score_ever = float(rank_val)
    ap = (metrics or {}).get("macro_average_precision")
    auc = (metrics or {}).get("macro_roc_auc")
    med = (metrics or {}).get("median_per_class_auc")
    info = {
        "ranking_metric": ranking_metric,
        "ranking_value": rank_val,
        "macro_average_precision": ap,
        "macro_roc_auc": auc,
        "median_per_class_auc": med,
        "iteration": iteration,
        "slot": slot_label,
        "run_id": run_id,
        "spec": spec,
    }
    (mem_dir / "best_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (mem_dir / "best_model_slot.py").write_text(slot_code, encoding="utf-8")
    for suffix in ("_a5", "_a4", "_a3", "_a2", "_a1", ""):
        yp = eval_dir / f"y_pred_{run_id}{suffix}.npy"
        yt = eval_dir / f"y_true_{run_id}{suffix}.npy"
        if yp.exists():
            shutil.copy2(yp, mem_dir / "best_val_preds.npy")
        if yt.exists():
            shutil.copy2(yt, mem_dir / "y_val.npy")
        if yp.exists():
            break
    print(f"  [Best] NEW BEST {_format_metrics(metrics)} (slot={slot_label})")
    return best_score_ever


def _execute_cnn_slot(
    *,
    iteration: int,
    spec: dict,
    slot_label: str,
    coder_llm: LLMClient,
    coder_temp: float,
    executor: CodeExecutor,
    evaluator: Evaluator,
    code_dir: Path,
    eval_dir: Path,
    train_overrides: dict,
    config: dict,
    max_attempts: int = 5,
) -> tuple[dict | None, str, dict]:
    spec = dict(spec)
    spec.update(train_overrides)
    locked_aug = _locked_cnn_aug(config)
    if locked_aug:
        spec.update({k: v for k, v in locked_aug.items() if k != "aug_preset"})
        if locked_aug.get("aug_preset"):
            spec["aug_preset"] = locked_aug["aug_preset"]
    seed_slot: str | None = None
    refine = config.get("cnn_refine") or {}
    if refine.get("enabled"):
        parent_slot = Path(refine.get("parent_memory_dir", "")) / "best_model_slot.py"
        if parent_slot.exists():
            seed_slot = parent_slot.read_text(encoding="utf-8")
    slot_code = generate_cnn_slot_code(
        coder_llm, spec, coder_temp, seed_slot_code=seed_slot
    )
    if not slot_code:
        return None, "", spec
    enforced = _enforced_training_overrides(config, spec)
    slot_code = _apply_slot_overrides(slot_code, enforced)
    print(
        "  [CNN] Enforced get_training_config(): "
        f"max_samples={enforced.get('max_samples')} epochs={enforced.get('epochs')} "
        f"n_mels={enforced.get('n_mels')} n_frames={enforced.get('n_frames')} "
        f"aug_preset={enforced.get('aug_preset')} "
        f"aug_prob={enforced.get('aug_prob')} "
        f"aug_noise_std={enforced.get('aug_noise_std')} "
        f"aug_time_mask={enforced.get('aug_time_mask')} "
        f"aug_freq_mask={enforced.get('aug_freq_mask')}",
        flush=True,
    )
    run_id = f"iter_{iteration:03d}_{slot_label}"
    slot_code, metrics, _result, _att = run_experiment_until_success(
        slot_code,
        run_id,
        code_dir,
        eval_dir,
        executor,
        evaluator,
        coder_llm,
        coder_temp,
        max_attempts=max_attempts,
        mel_cache_dir=_soundscape_mel_cache_dir(config),
        focal_cache_dir=_soundscape_mel_cache_dir(config),
        # Re-inject locked overrides into every LLM auto-fix. Without this,
        # an LLM fix on attempt 2+ would drop _META_OVERRIDES and let the
        # bare get_training_config() take effect, which is how stage 1b was
        # ending up with aug_prob=0/aug_noise_std=0/... despite "high" lock.
        reapply_overrides=lambda code: _apply_slot_overrides(code, enforced),
    )
    return metrics, slot_code, spec


def _soundscape_mel_cache_dir(config: dict) -> Path:
    custom = (config.get("cnn") or {}).get("soundscape_mel_cache_dir")
    return Path(custom) if custom else DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR


def _warm_soundscape_mel_cache(config: dict) -> None:
    cache_dir = _soundscape_mel_cache_dir(config)
    n_mels = int((config.get("cnn_augmentation") or {}).get("n_mels", 64))
    n_frames = int((config.get("cnn_augmentation") or {}).get("n_frames", 128))
    # Locked aug dict may omit n_mels; use search cheap defaults.
    cheap = (config.get("search") or {}).get("cheap") or {}
    n_mels = int(cheap.get("n_mels", n_mels))
    n_frames = int(cheap.get("n_frames", n_frames))
    print(f"  [CNN] Soundscape mel cache → {cache_dir} ({n_mels}x{n_frames})")
    ensure_soundscape_mel_cache(cache_dir, n_mels=n_mels, n_frames=n_frames)


def _focal_cache_params(config: dict) -> tuple[str, int | None, int, int]:
    from augmentation import normalize_baseline_aug_name

    cheap = (config.get("search") or {}).get("cheap") or {}
    preset = normalize_baseline_aug_name(
        config.get("meta_aug_preset")
        or (config.get("cnn_augmentation") or {}).get("aug_preset")
        or "medium"
    )
    max_samples = cheap.get("max_samples")
    trial = config.get("cnn_1c_trial") or {}
    if trial:
        preset = normalize_baseline_aug_name(
            trial.get("aug_preset")
            or (trial.get("aug_dict") or {}).get("aug_preset")
            or preset
        )
        max_samples = trial.get("max_samples", max_samples)
    n_mels = int(cheap.get("n_mels", 64))
    n_frames = int(cheap.get("n_frames", 128))
    return preset, max_samples, n_mels, n_frames


def _warm_focal_train_cache(config: dict) -> None:
    from cnn_focal_cache import focal_clip_seed_from_config

    preset, max_samples, n_mels, n_frames = _focal_cache_params(config)
    cache_dir = _soundscape_mel_cache_dir(config)
    clip_seed = focal_clip_seed_from_config(config)
    print(
        f"  [CNN] Focal train cache → {cache_dir} "
        f"preset={preset} max_samples={max_samples} ({n_mels}x{n_frames}) clip_seed={clip_seed}"
    )
    ensure_focal_train_cache(
        cache_dir,
        aug_preset=preset,
        max_samples=max_samples,
        n_mels=n_mels,
        n_frames=n_frames,
        clip_seed=clip_seed,
    )


def run_cnn_explore(config: dict) -> None:
    dirs = _resolve_dirs(config)
    mem_dir, code_dir, eval_dir = dirs["mem_dir"], dirs["code_dir"], dirs["eval_dir"]
    mel_cache_dir = _soundscape_mel_cache_dir(config)
    focal_cache_dir = mel_cache_dir
    _warm_soundscape_mel_cache(config)
    _warm_focal_train_cache(config)
    max_iterations = int(config.get("max_iterations", 5))
    train_overrides = _cheap_training_overrides(config)
    ranking_metric = _ranking_metric_from_config(config)

    provider = config.get("llm_researcher", {}).get("provider") or config["llm"]["provider"]
    researcher_model = (
        config.get("researcher", {}).get("model")
        or config.get("llm_researcher", {}).get("model")
        or config["llm"]["model"]
    )
    coder_model = config["llm"]["model"]
    researcher_temp = float(config.get("researcher", {}).get("temperature", 0.2))
    coder_temp = float(config.get("llm", {}).get("temperature", 0.2))
    timeout = float(config.get("execution", {}).get("timeout_seconds", 1800))
    batch_size = int(
        config.get("researcher", {}).get("batch_size", 3)
    )

    researcher_llm = LLMClient(
        provider=provider,
        model=researcher_model,
        timeout_seconds=float(
            config.get("meta_agent", {}).get("researcher_timeout_seconds", 600)
        ),
    )
    coder_timeout = float(
        config.get("llm", {}).get("timeout_seconds")
        or config.get("researcher", {}).get("timeout_seconds", 600)
    )
    coder_llm = LLMClient(
        provider=provider, model=coder_model, timeout_seconds=coder_timeout
    )
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=timeout,
    )
    print(f"  [CNN] Experiment subprocess timeout={timeout:.0f}s  coder LLM timeout={coder_timeout:.0f}s")
    evaluator = Evaluator(row_id_column_name="row_id")
    memory = ExperimentMemory(mem_dir, ranking_metric=ranking_metric)
    researcher = CnnResearcher(
        researcher_llm, memory, researcher_temp, batch_size=batch_size
    )

    preset = config.get("meta_aug_preset", "?")
    print("=" * 60)
    print(f"  CNN STAGED — Explore (1a) | preset={preset} | rounds={max_iterations}")
    print(f"  {batch_size} coder runs per researcher call")
    print("=" * 60)

    prior = memory.best_runs(1)
    best_score_ever = memory._ranking_value(prior[0]) if prior else -1.0

    if memory.total() == 0:
        print("\n  ITERATION 0 — Safe CNN baseline")
        run_id = "iter_000_baseline"
        baseline_code = _apply_slot_overrides(
            SAFE_BASELINE_SLOT_CODE, _enforced_training_overrides(config)
        )
        slot_code, metrics, _r, _a = run_experiment_until_success(
            baseline_code,
            run_id,
            code_dir,
            eval_dir,
            executor,
            evaluator,
            coder_llm,
            coder_temp,
            max_attempts=1,
            use_llm_fixes=False,
            mel_cache_dir=mel_cache_dir,
            focal_cache_dir=focal_cache_dir,
        )
        spec = _fill_cnn_defaults({"arch_type": "shallow_cnn", "strategy": "baseline"})
        memory.log(spec=spec, metrics=metrics, code=slot_code or SAFE_BASELINE_SLOT_CODE)
        rv = _ranking_value_from_metrics(metrics)
        best_score_ever = _promote_cnn_best(
            rank_val=rv,
            metrics=metrics,
            spec=spec,
            iteration=0,
            slot_label="baseline",
            best_score_ever=best_score_ever,
            mem_dir=mem_dir,
            slot_code=slot_code or SAFE_BASELINE_SLOT_CODE,
            eval_dir=eval_dir,
            run_id=f"{run_id}_a1",
            ranking_metric=ranking_metric,
        )
        _append_results(mem_dir, {
            "run_id": run_id,
            "success": metrics is not None and metrics.get("status") == "success",
            "ranking_value": rv,
            "slot_code": slot_code,
        })

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─' * 60}\n  PLANNER ROUND {iteration}/{max_iterations}\n{'─' * 60}")
        specs = researcher.next_experiments()
        for slot_i, spec in enumerate(specs, 1):
            slot_label = str(spec.get("slot") or f"s{slot_i}")
            print(f"\n  ▸ Slot {slot_i}/{len(specs)}: {slot_label}")
            metrics, slot_code, spec = _execute_cnn_slot(
                iteration=iteration,
                spec=spec,
                slot_label=slot_label,
                coder_llm=coder_llm,
                coder_temp=coder_temp,
                executor=executor,
                evaluator=evaluator,
                code_dir=code_dir,
                eval_dir=eval_dir,
                train_overrides=train_overrides,
                config=config,
            )
            print(f"  [Result] [{slot_label}] {_format_metrics(metrics)}")
            memory.log(spec=spec, metrics=metrics, code=slot_code or "")
            rv = _ranking_value_from_metrics(metrics)
            run_id = f"iter_{iteration:03d}_{slot_label}"
            _append_results(mem_dir, {
                "run_id": run_id,
                "success": rv is not None,
                "ranking_value": rv,
                "macro_average_precision": (metrics or {}).get("macro_average_precision"),
                "search_type": "cnn_staged_explore",
                "slot_code": slot_code,
            })
            best_score_ever = _promote_cnn_best(
                rank_val=rv,
                metrics=metrics,
                spec=spec,
                iteration=iteration,
                slot_label=slot_label,
                best_score_ever=best_score_ever,
                mem_dir=mem_dir,
                slot_code=slot_code or "",
                eval_dir=eval_dir,
                run_id=f"{run_id}_a1",
                ranking_metric=ranking_metric,
            )
        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] {memory._format_run_score(best[0])}")

    print(f"\n{'=' * 60}\n  CNN explore done → {mem_dir}\n{'=' * 60}")


def run_cnn_refine(config: dict) -> None:
    refine_cfg = config.get("cnn_refine") or {}
    dirs = _resolve_dirs(config)
    mem_dir, code_dir, eval_dir = dirs["mem_dir"], dirs["code_dir"], dirs["eval_dir"]
    train_overrides = _cheap_training_overrides(config)
    ranking_metric = _ranking_metric_from_config(config)

    locked = str(refine_cfg.get("locked_arch_type") or "shallow_cnn")
    seed_spec = dict(refine_cfg.get("seed_spec") or {})
    seed_spec.setdefault("arch_type", locked)
    seed_score = float(refine_cfg.get("seed_score", -1.0))

    initial = max(1, int(refine_cfg.get("initial_iterations", 6)))
    bonus = max(1, int(refine_cfg.get("bonus_iterations_on_improve", 6)))
    max_total = max(initial, int(refine_cfg.get("max_iterations_per_model", 30)))
    batch_size = int(refine_cfg.get("experiments_per_researcher_call", 3))

    provider = config.get("llm_researcher", {}).get("provider") or config["llm"]["provider"]
    researcher_model = config.get("researcher", {}).get("model") or config["llm_researcher"]["model"]
    coder_model = config["llm"]["model"]
    researcher_temp = float(config.get("researcher", {}).get("temperature", 0.2))
    coder_temp = float(config.get("llm", {}).get("temperature", 0.2))
    timeout = float(config.get("execution", {}).get("timeout_seconds", 1800))

    researcher_llm = LLMClient(provider=provider, model=researcher_model, timeout_seconds=600)
    coder_timeout = float(
        config.get("llm", {}).get("timeout_seconds")
        or config.get("researcher", {}).get("timeout_seconds", 600)
    )
    coder_llm = LLMClient(
        provider=provider, model=coder_model, timeout_seconds=coder_timeout
    )
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=timeout,
    )
    print(f"  [CNN] Experiment subprocess timeout={timeout:.0f}s  coder LLM timeout={coder_timeout:.0f}s")
    evaluator = Evaluator(row_id_column_name="row_id")
    memory = ExperimentMemory(mem_dir, ranking_metric=ranking_metric)

    champion_path = mem_dir / REFINE_CHAMPION_SPEC_FILE
    champion_path.write_text(json.dumps(seed_spec, indent=2), encoding="utf-8")
    if (Path(refine_cfg["parent_memory_dir"]) / "best_model_slot.py").exists() and not (
        mem_dir / "best_model_slot.py"
    ).exists():
        shutil.copy2(
            Path(refine_cfg["parent_memory_dir"]) / "best_model_slot.py",
            mem_dir / "best_model_slot.py",
        )

    researcher = CnnResearcher(
        researcher_llm,
        memory,
        researcher_temp,
        refine_mode=True,
        locked_arch_type=locked,
        seed_spec=seed_spec,
        batch_size=batch_size,
    )

    print("=" * 60)
    print(f"  CNN STAGED — Refine (1b) | locked={locked} | budget {initial}+{bonus}≤{max_total}")
    print("=" * 60)
    _warm_soundscape_mel_cache(config)
    _warm_focal_train_cache(config)

    best_score_ever = seed_score
    training_rounds = 0
    iteration = 0

    while training_rounds < max_total:
        iteration += 1
        rounds_left = max_total - training_rounds
        n_slots = min(batch_size, rounds_left)
        if training_rounds < initial:
            phase = f"initial ({training_rounds + 1}/{initial})"
        else:
            phase = f"bonus ({training_rounds - initial + 1})"
        print(f"\n{'─' * 60}\n  REFINE ROUND {iteration} — {phase}\n{'─' * 60}")
        specs = researcher.next_experiments()[:n_slots]
        improved = False
        for slot_i, spec in enumerate(specs, 1):
            slot_label = str(spec.get("slot") or f"r{slot_i}")
            metrics, slot_code, spec = _execute_cnn_slot(
                iteration=iteration,
                spec=spec,
                slot_label=slot_label,
                coder_llm=coder_llm,
                coder_temp=coder_temp,
                executor=executor,
                evaluator=evaluator,
                code_dir=code_dir,
                eval_dir=eval_dir,
                train_overrides=train_overrides,
                config=config,
            )
            memory.log(spec=spec, metrics=metrics, code=slot_code or "")
            rv = _ranking_value_from_metrics(metrics)
            training_rounds += 1
            if rv is not None and rv > best_score_ever:
                improved = True
                best_score_ever = _promote_cnn_best(
                    rank_val=rv,
                    metrics=metrics,
                    spec=spec,
                    iteration=iteration,
                    slot_label=slot_label,
                    best_score_ever=best_score_ever,
                    mem_dir=mem_dir,
                    slot_code=slot_code or "",
                    eval_dir=eval_dir,
                    run_id=f"iter_{iteration:03d}_{slot_label}_a1",
                    ranking_metric=ranking_metric,
                )
            if training_rounds >= max_total:
                break
        if training_rounds >= initial and not improved:
            print("  [Refine] No improvement this round — stopping bonus phase.")
            break
        if training_rounds >= initial and improved:
            if training_rounds + bonus > max_total:
                continue
        if training_rounds >= initial and training_rounds >= initial + bonus and not improved:
            break

    print(f"\n{'=' * 60}\n  CNN refine done | best={best_score_ever:.5f}\n{'=' * 60}")


_META_OVERRIDE_MARKER = "# --- META OVERRIDES ---"
_META_AUG_MARKER = "# --- META AUG OVERRIDE ---"


def _strip_slot_meta_overrides(slot_code: str) -> str:
    """Remove injected override wrappers so _apply_slot_overrides is idempotent."""
    code = slot_code
    for marker in (_META_OVERRIDE_MARKER, _META_AUG_MARKER):
        idx = code.find(marker)
        if idx >= 0:
            code = code[:idx]
    # LLM fixes sometimes copy the wrapper; drop delegate-only stubs.
    code = re.sub(
        r"\ndef _ORIG_GET_TRAINING_CONFIG\(\):\s*\n"
        r"\s*cfg = _ORIG_GET_TRAINING_CONFIG\(\)\s*\n"
        r"\s*cfg\.update\([^)]+\)\s*\n"
        r"\s*return cfg\s*",
        "\n",
        code,
        flags=re.MULTILINE,
    )
    code = re.sub(
        r"\ndef get_training_config\(\):\s*\n"
        r"\s*cfg = _ORIG_GET_TRAINING_CONFIG\(\)\s*\n"
        r"\s*cfg\.update\([^)]+\)\s*\n"
        r"\s*return cfg\s*",
        "\n",
        code,
        flags=re.MULTILINE,
    )
    code = re.sub(
        r"^def _ORIG_GET_TRAINING_CONFIG\s*\(",
        "def get_training_config(",
        code,
        flags=re.MULTILINE,
    )
    return code.rstrip()


def _apply_slot_overrides(slot_code: str, overrides: dict) -> str:
    """Single get_training_config() override after renaming the original."""
    base = _strip_slot_meta_overrides(slot_code)
    if "def get_training_config" not in base:
        return base
    renamed = re.sub(
        r"^def get_training_config\s*\(",
        "def _ORIG_GET_TRAINING_CONFIG(",
        base,
        count=1,
        flags=re.MULTILINE,
    )
    ov = repr(dict(overrides))
    return (
        renamed.strip()
        + f"\n\n{_META_OVERRIDE_MARKER}\n"
        f"_META_OVERRIDES = {ov}\n\n"
        f"def get_training_config():\n"
        f"    cfg = _ORIG_GET_TRAINING_CONFIG()\n"
        f"    cfg.update(_META_OVERRIDES)\n"
        f"    return cfg\n"
    )


def run_cnn_1c_trial(config: dict) -> dict:
    """Single augmentation trial with locked architecture (meta-agent 1c)."""
    trial = config.get("cnn_1c_trial") or {}
    _warm_soundscape_mel_cache(config)
    _warm_focal_train_cache(config)
    dirs = _resolve_dirs(config)
    mem_dir, code_dir, eval_dir = dirs["mem_dir"], dirs["code_dir"], dirs["eval_dir"]
    locked_path = Path(trial["locked_slot_path"])
    if not locked_path.exists():
        raise FileNotFoundError(f"Locked slot missing: {locked_path}")
    slot_code = locked_path.read_text(encoding="utf-8")
    aug_dict = dict(trial.get("aug_dict") or {})
    train_overrides = {
        "max_samples": trial.get("max_samples", 2000),
        "epochs": trial.get("epochs", config.get("search", {}).get("cheap", {}).get("epochs", 3)),
        "val_split": trial.get("val_split", 0.2),
        "focal_clip_seed": _cheap_training_overrides(config)["focal_clip_seed"],
    }
    # Capture the full aug+training override dict so we can re-inject it
    # if the LLM auto-fix returns a bare slot during a retry.
    _full_overrides = {**aug_dict, **train_overrides}
    slot_code = _apply_slot_overrides(slot_code, _full_overrides)

    provider = config["llm"]["provider"]
    coder_timeout = float(
        config.get("llm", {}).get("timeout_seconds")
        or config.get("researcher", {}).get("timeout_seconds", 600)
    )
    coder_llm = LLMClient(
        provider=provider, model=config["llm"]["model"], timeout_seconds=coder_timeout
    )
    coder_temp = float(config["llm"].get("temperature", 0.2))
    timeout = float(config.get("execution", {}).get("timeout_seconds", 1800))
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=timeout,
    )
    print(f"  [CNN] Experiment subprocess timeout={timeout:.0f}s  coder LLM timeout={coder_timeout:.0f}s")
    evaluator = Evaluator(row_id_column_name="row_id")
    trial_id = str(trial.get("trial_id", "trial"))
    metrics, slot_code, _ = _execute_locked_slot(
        slot_code=slot_code,
        run_id=trial_id,
        code_dir=code_dir,
        eval_dir=eval_dir,
        executor=executor,
        evaluator=evaluator,
        coder_llm=coder_llm,
        coder_temp=coder_temp,
        max_attempts=int(trial.get("max_attempts", 3)),
        mel_cache_dir=_soundscape_mel_cache_dir(config),
        focal_cache_dir=_soundscape_mel_cache_dir(config),
        reapply_overrides=lambda code: _apply_slot_overrides(code, _full_overrides),
    )
    rv = _ranking_value_from_metrics(metrics)
    entry = {
        "trial_id": trial_id,
        "aug_preset": trial.get("aug_preset"),
        "aug_dict": aug_dict,
        "success": rv is not None,
        "ranking_value": rv,
        "macro_average_precision": (metrics or {}).get("macro_average_precision"),
        "metrics": metrics,
    }
    out_path = mem_dir / f"1c_{trial_id}.json"
    out_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    if rv is not None:
        info_path = mem_dir / "best_model_info.json"
        prev = -1.0
        if info_path.exists():
            try:
                prev = float(json.loads(info_path.read_text()).get("ranking_value", -1))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if rv >= prev:
            _promote_cnn_best(
                rank_val=rv,
                metrics=metrics,
                spec={"trial_id": trial_id, "aug_preset": trial.get("aug_preset")},
                iteration=0,
                slot_label=trial_id,
                best_score_ever=prev,
                mem_dir=mem_dir,
                slot_code=slot_code,
                eval_dir=eval_dir,
                run_id=f"{trial_id}_a1",
                ranking_metric=_ranking_metric_from_config(config),
            )
    return entry


def inject_training_cap(slot_code: str, overrides: dict) -> str:
    """Force max_samples/epochs/val_split for fast search trials."""
    return _apply_slot_overrides(slot_code, overrides)


def _execute_locked_slot(
    *,
    slot_code: str,
    run_id: str,
    code_dir: Path,
    eval_dir: Path,
    executor: CodeExecutor,
    evaluator: Evaluator,
    coder_llm: LLMClient,
    coder_temp: float,
    max_attempts: int,
    mel_cache_dir: Path | None = None,
    focal_cache_dir: Path | None = None,
    reapply_overrides=None,
) -> tuple[dict | None, str, int]:
    slot_code, metrics, _r, att = run_experiment_until_success(
        slot_code,
        run_id,
        code_dir,
        eval_dir,
        executor,
        evaluator,
        coder_llm,
        coder_temp,
        max_attempts=max_attempts,
        use_llm_fixes=max_attempts > 1,
        mel_cache_dir=mel_cache_dir,
        focal_cache_dir=focal_cache_dir or mel_cache_dir,
        reapply_overrides=reapply_overrides,
    )
    return metrics, slot_code, att


def run_cnn_final_train(config: dict) -> dict:
    """Full-data CNN training (stage 1d)."""
    from cnn_focal_cache import focal_clip_seed_from_config

    ft = config.get("cnn_final_train") or {}
    dirs = _resolve_dirs(config)
    mem_dir = dirs["mem_dir"]
    code_dir = dirs["code_dir"]
    locked_path = Path(ft["locked_slot_path"])
    slot_code = locked_path.read_text(encoding="utf-8")
    aug_dict = dict(ft.get("aug_dict") or {})
    aug_preset = str(aug_dict.get("aug_preset") or "high")
    sc_final = config.get("search", {}).get("final", {})
    max_samples = ft.get("max_samples", sc_final.get("max_samples"))
    clip_seed = focal_clip_seed_from_config(config)
    cache_dir = _soundscape_mel_cache_dir(config)
    cheap = (config.get("search") or {}).get("cheap") or {}
    n_mels = int(cheap.get("n_mels", 64))
    n_frames = int(cheap.get("n_frames", 128))
    ensure_focal_train_cache(
        cache_dir,
        aug_preset=aug_preset,
        max_samples=int(max_samples) if max_samples is not None else None,
        n_mels=n_mels,
        n_frames=n_frames,
        clip_seed=clip_seed,
        force=bool(ft.get("rebuild_focal_cache", False)),
    )

    # `.get("val_split", default)` only kicks in when the key is missing — a
    # key that is *present but None* (which happens when meta_agent forwards
    # an empty stage_1d.val_split) would otherwise propagate None into the
    # generated training script and crash on float(None). Resolve to a real
    # number here so the slot override is always usable.
    _epochs = ft.get("epochs")
    if _epochs is None:
        _epochs = sc_final.get("epochs", 15)
    _val_split = ft.get("val_split")
    if _val_split is None:
        _val_split = sc_final.get("val_split", 0.1)
    train_cfg = {
        "max_samples": int(max_samples) if max_samples is not None else None,
        "epochs": int(_epochs),
        "val_split": float(_val_split),
        "focal_clip_seed": clip_seed,
    }
    slot_code = _apply_slot_overrides(slot_code, {**aug_dict, **train_cfg})

    model_path = Path(ft["model_save_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    script = assemble_script(
        slot_code,
        is_final=True,
        model_save_path=str(model_path),
        focal_cache_dir=cache_dir,
    )
    script_path = code_dir / "final_train.py"
    script_path.write_text(script, encoding="utf-8")

    timeout = config.get("execution", {}).get("timeout_seconds", 1800)
    if ft.get("final_timeout_seconds"):
        timeout = int(ft["final_timeout_seconds"])
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=timeout,
    )
    print("=" * 60)
    print(f"  CNN STAGED — Final train (1d) → {model_path}")
    print(f"  timeout={timeout}s  streaming output (TRAIN_HEARTBEAT / TRAIN_BATCH / MODEL_SAVED)")
    print("=" * 60)
    # Stream stdout live — without this the parent buffers everything until
    # the subprocess exits, so a 2-hour final-train run appears completely
    # silent (and looks "stuck") even when it is progressing normally.
    result = executor.run_file(script_path, stream_output=True, label="1d")
    ok = result.success and "MODEL_SAVED" in (result.stdout or "")
    summary = {"success": ok, "model_path": str(model_path)}
    (mem_dir / "final_train_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not ok:
        tail = (result.stderr or result.stdout or "")[-800:]
        print(f"  [1d] Failed: {tail}")
    return summary


def run_cnn_pseudo_refine(config: dict) -> dict:
    """Pseudo-label refine (stage 1e) — requires cnn_pseudo cache + final model."""
    from cnn_pseudo import (
        build_cnn_pseudo_cache,
        build_cnn_pseudo_refine_script,
        pseudo_npz_is_empty,
    )

    cfg = config.get("cnn_pseudo_refine") or {}
    dirs = _resolve_dirs(config)
    mem_dir, code_dir = dirs["mem_dir"], dirs["code_dir"]
    pseudo_npz = Path(cfg.get("pseudo_npz", mem_dir.parent / "cnn_cache" / "pseudo_labels.npz"))
    teacher = Path(cfg.get("teacher_model", mem_dir / "final" / "model.keras"))
    slot_path = Path(cfg.get("locked_slot_path", mem_dir / "best_model_slot.py"))

    pseudo_stats: dict = {}
    if cfg.get("rebuild_pseudo_cache", True) or not pseudo_npz.exists():
        pseudo_stats = build_cnn_pseudo_cache(
            config=config,
            teacher_model_path=teacher,
            slot_code_path=slot_path,
            out_path=pseudo_npz,
            top1_threshold=float(cfg.get("top1_threshold", 0.55)),
            runnerup_max=float(cfg.get("runnerup_max", 0.35)),
            pseudo_label_weight=float(cfg.get("pseudo_label_weight", 0.8)),
            max_files=cfg.get("max_soundscape_files"),
            max_windows=cfg.get("max_pseudo_windows"),
            heartbeat_every=int(cfg.get("heartbeat_every", 10)),
        )
        if pseudo_stats.get("empty_pseudo"):
            print("\n  [CNN 1e] Continuing with supervised-only fine-tune (no pseudo windows).")
    save_path = Path(cfg.get("model_save_path", teacher.parent / "model_pseudo.keras"))
    _no_pseudo = pseudo_stats.get("empty_pseudo") or pseudo_npz_is_empty(pseudo_npz)
    if cfg.get("skip_fine_tune_without_pseudo", True) and _no_pseudo and teacher.exists():
        import shutil

        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(teacher, save_path)
        print(
            f"\n  [CNN 1e] No pseudo labels — skipped fine-tune; copied teacher → {save_path.name}"
        )
        summary = {
            "success": True,
            "pseudo_npz": str(pseudo_npz),
            "pseudo_stats": pseudo_stats,
            "skipped_fine_tune": True,
        }
        (mem_dir / "pseudo_refine_result.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    _mss = cfg.get("max_supervised_samples")
    _aug = dict(cfg.get("aug_dict") or {})
    script = build_cnn_pseudo_refine_script(
        slot_code_path=slot_path,
        teacher_model_path=teacher,
        pseudo_npz=pseudo_npz,
        sample_weight_supervised=float(cfg.get("sample_weight_supervised", 1.0)),
        sample_weight_pseudo=float(cfg.get("sample_weight_pseudo", 0.5)),
        epochs=int(cfg.get("fine_tune_epochs", 15)),
        learning_rate=float(cfg.get("fine_tune_lr", 2e-4)),
        model_save_path=str(cfg.get("model_save_path", teacher.parent / "model_pseudo.keras")),
        focal_cache_dir=_soundscape_mel_cache_dir(config),
        max_supervised_samples=int(_mss) if _mss is not None else None,
        aug_dict=_aug,
    )
    script_path = code_dir / "pseudo_refine.py"
    script_path.write_text(script, encoding="utf-8")
    timeout = int(cfg.get("refine_timeout_seconds", 7200))
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=timeout,
    )
    print("=" * 60)
    print("  CNN STAGED — Pseudo-label refine (1e)")
    print(f"  timeout={timeout}s  streaming output")
    print("=" * 60)
    # Same rationale as 1d — fine-tune runs are long and the buffered
    # `run_file` mode hides progress entirely.
    result = executor.run_file(script_path, stream_output=True, label="1e")
    ok = result.success and "PSEUDO_REFINE_DONE" in (result.stdout or "")
    summary = {"success": ok, "pseudo_npz": str(pseudo_npz), "pseudo_stats": pseudo_stats}
    (mem_dir / "pseudo_refine_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not ok:
        tail = (result.stderr or result.stdout or "")[-800:]
        print(f"  [1e] Failed: {tail}")
    return summary


def dispatch_cnn_staged(config: dict) -> None:
    """Route staged CNN subprocess modes."""
    if config.get("cnn_pseudo_refine"):
        run_cnn_pseudo_refine(config)
        return
    if config.get("cnn_final_train"):
        run_cnn_final_train(config)
        return
    if config.get("cnn_1c_trial"):
        run_cnn_1c_trial(config)
        return
    refine = config.get("cnn_refine") or {}
    if refine.get("enabled"):
        run_cnn_refine(config)
        return
    if config.get("cnn_explore") or config.get("max_iterations"):
        run_cnn_explore(config)
        return
    raise ValueError("cnn_staged: no recognized mode (explore/refine/1c/final/pseudo)")
