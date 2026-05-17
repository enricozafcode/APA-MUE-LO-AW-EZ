"""BirdNET meta-agent stages 1b–1e (parallel to Perch; calls birdnet_agent only)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from augmentation import get_audio_embedding_aug, get_aug_search_preset, list_aug_search_preset_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ma():
    import meta_agent as ma

    return ma


def collect_birdnet_1a_top_candidates(config: dict, top_k: int = 2) -> list[dict]:
    ma = _ma()
    metric = ma._meta_primary_metric(config)
    baselines = ma._baseline_names(config)
    if ma.ARCH_SEARCH_1A_RESULTS.exists():
        try:
            summary = json.loads(ma.ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            runs = (summary.get("tracks") or {}).get("birdnet", {}).get("runs") or []
            if runs:
                baselines = [r["aug_baseline"] for r in runs if r.get("aug_baseline")]
        except (json.JSONDecodeError, OSError):
            pass
    candidates: list[dict] = []
    for baseline in baselines:
        mem_dir = ma.META_LOGS / "birdnet" / baseline
        if not (mem_dir / "best_head_code.py").exists():
            continue
        champ = _load_champion(baseline, mem_dir, metric, ma)
        if champ:
            candidates.append(champ)
    candidates.sort(key=lambda c: float(c["ranking_value"]), reverse=True)
    picked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        key = (c["aug_baseline"], str(c.get("arch_type", "?")))
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= top_k:
            break
    return picked[:top_k]


def _load_champion(baseline: str, mem_dir: Path, metric: str, ma) -> dict | None:
    val = -1.0
    spec: dict = {}
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            val = float(info.get("ranking_value", -1))
            spec = dict(info.get("spec") or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if val < 0:
        sc = ma._soundscape_suite({}).score_birdnet_mem_dir(
            mem_dir, val_cache=ma.BIRDNET_SHARED_VAL_CACHE
        )
        if sc is None:
            return None
        val = float(sc.primary_value)
    return {
        "aug_baseline": baseline,
        "arch_type": spec.get("arch_type", "residual_mlp"),
        "memory_dir": str(mem_dir),
        "cache_dir": str(ma.META_LOGS / "birdnet_cache" / baseline),
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": val,
    }


def run_stage_1b_birdnet_refine(config: dict, suite) -> dict:
    ma = _ma()
    cfg = ma._birdnet_stage_cfg(config, "birdnet_stage_1b")
    if not cfg.get("enabled", False):
        return {}
    if ma._skip_if_completed(config, ma.BIRDNET_ARCH_1B_RESULTS, "BirdNET 1b"):
        try:
            return json.loads(ma.BIRDNET_ARCH_1B_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    top_k = max(1, int(cfg.get("top_k_models", 1)))
    metric = ma._meta_primary_metric(config)
    candidates = collect_birdnet_1a_top_candidates(config, top_k=top_k)
    if not candidates:
        return {"stage": "1b_birdnet_refine", "refine_runs": []}
    ma._print_stage_1b_champions(candidates, metric)
    refine_runs = []
    for i, cand in enumerate(candidates, 1):
        aug = cand["aug_baseline"]
        arch = str(cand.get("arch_type", "residual_mlp"))
        mem_dir = ma.META_LOGS / "birdnet" / "refine" / f"rank{i}_{aug}_{arch}"[:80]
        code_dir = mem_dir / "codes"
        mem_dir.mkdir(parents=True, exist_ok=True)
        batch = cfg.get("experiments_per_researcher_call") or config.get("researcher", {}).get(
            "batch_size", 3
        )
        override = {
            "meta_aug_preset": aug,
            "augmentation": get_audio_embedding_aug(aug),
            "birdnet_refine": {
                "enabled": True,
                "aug_baseline": aug,
                "locked_arch_type": arch,
                "seed_spec": cand["spec"],
                "seed_score": float(cand["ranking_value"]),
                "parent_memory_dir": cand["memory_dir"],
                "experiments_per_researcher_call": int(batch),
                "initial_iterations": int(cfg.get("initial_iterations", 4)),
                "bonus_iterations_on_improve": int(cfg.get("bonus_iterations_on_improve", 4)),
                "max_iterations_per_model": int(cfg.get("max_iterations_per_model", 20)),
            },
            "birdnet": {
                "memory_dir": str(mem_dir),
                "cache_dir": str(cand["cache_dir"]),
                "code_dir": str(code_dir),
                "val_cache_path": str(ma.BIRDNET_SHARED_VAL_CACHE),
            },
        }
        rc = ma._run_subprocess("birdnet_agent.py", override, config)
        refine_runs.append(
            {
                "rank": i,
                "aug_baseline": aug,
                "arch_type": arch,
                "memory_dir": str(mem_dir),
                "subprocess_rc": rc,
            }
        )
    scored = []
    for run in refine_runs:
        sc = suite.score_birdnet_mem_dir(
            Path(run["memory_dir"]), val_cache=ma.BIRDNET_SHARED_VAL_CACHE
        )
        ma._print_soundscape_score(f"BirdNET/refine/{run['arch_type']}", sc)
        entry = {**run, "score": sc.to_dict() if sc else None}
        scored.append(entry)
    winner = ma._pick_track_winner(scored, metric)
    summary = {
        "stage": "1b_birdnet_refine",
        "primary_metric": metric,
        "candidates": candidates,
        "refine_runs": scored,
        "winner": winner,
    }
    ma.BIRDNET_ARCH_1B_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_1c_birdnet_aug_search(config: dict, suite) -> dict:
    ma = _ma()
    cfg = ma._birdnet_stage_cfg(config, "birdnet_stage_1c")
    if not cfg.get("enabled", False):
        return {}
    if ma._skip_if_completed(config, ma.BIRDNET_ARCH_1C_RESULTS, "BirdNET 1c"):
        try:
            return json.loads(ma.BIRDNET_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if not ma.BIRDNET_ARCH_1B_RESULTS.exists():
        print("\n  [BirdNET 1c] Run 1b first.")
        return {}
    s1b = json.loads(ma.BIRDNET_ARCH_1B_RESULTS.read_text(encoding="utf-8"))
    winner = s1b.get("winner")
    if not winner:
        return {}
    head_path = Path(winner["memory_dir"]) / "best_head_code.py"
    if not head_path.exists():
        return {}
    metric = ma._meta_primary_metric(config)
    ma._enrich_perch_winner_metrics(winner, metric, suite)
    ma._print_stage_1c_locked_head(winner, metric)
    presets = list(cfg.get("aug_presets") or list_aug_search_preset_names())
    wb = winner.get("aug_baseline")
    if cfg.get("include_winner_aug_baseline", True) and wb and wb not in presets:
        presets = [wb] + presets
    baselines = ma._baseline_names(config)
    runs: list[dict] = []
    for preset in presets:
        try:
            aug_dict = get_aug_search_preset(preset)
        except KeyError:
            aug_dict = get_audio_embedding_aug(preset if preset in baselines else "medium")
        cache_dir = ma.META_LOGS / "birdnet_cache" / "aug_search" / preset
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{preset}_aug.json").write_text(json.dumps(aug_dict, indent=2), encoding="utf-8")
        mem_dir = ma.META_LOGS / "birdnet" / "aug_search" / preset
        mem_dir.mkdir(parents=True, exist_ok=True)
        embed_frac = float(ma._meta_cfg(config).get("arch_search_embed_sample_frac", 0.5))
        embed_cap = ma._embed_max_samples(config)
        override = {
            "meta_aug_preset": preset,
            "augmentation": aug_dict,
            "train_sample_frac": embed_frac,
            "force_rebuild_cache": True,
            "birdnet_fixed_train": {
                "enabled": True,
                "head_code_path": str(head_path),
                "aug_preset": preset,
                "spec": {"aug_preset": preset},
            },
            "birdnet": {
                "memory_dir": str(mem_dir),
                "cache_dir": str(cache_dir),
                "code_dir": str(mem_dir / "codes"),
                "val_cache_path": str(ma.BIRDNET_SHARED_VAL_CACHE),
            },
        }
        if embed_cap is not None:
            override["max_train_samples"] = embed_cap
        ma._run_subprocess("birdnet_agent.py", override, config, quiet=True)
        sc = suite.score_birdnet_mem_dir(mem_dir, val_cache=ma.BIRDNET_SHARED_VAL_CACHE)
        ma._print_soundscape_score(f"BirdNET/1c/{preset}", sc)
        runs.append(
            {
                "aug_preset": preset,
                "memory_dir": str(mem_dir),
                "cache_dir": str(cache_dir),
                "score": sc.to_dict() if sc else None,
            }
        )
    aug_winner = ma._pick_track_winner(runs, metric)
    summary = {
        "stage": "1c_birdnet_aug_search",
        "primary_metric": metric,
        "locked_head_code": str(head_path),
        "refine_winner_1b": winner,
        "runs": runs,
        "winner": aug_winner,
    }
    ma.BIRDNET_ARCH_1C_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_1d_birdnet_final_train(config: dict) -> dict:
    ma = _ma()
    cfg = ma._birdnet_stage_cfg(config, "birdnet_stage_1d")
    if not cfg.get("enabled", False) or not ma.BIRDNET_ARCH_1C_RESULTS.exists():
        return {"success": False}
    s1c = json.loads(ma.BIRDNET_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
    aug_winner = s1c.get("winner")
    refine_winner = s1c.get("refine_winner_1b")
    if not aug_winner or not refine_winner:
        return {"success": False}
    head_path = Path(s1c.get("locked_head_code", refine_winner["memory_dir"]))
    if not head_path.exists():
        head_path = Path(refine_winner["memory_dir"]) / "best_head_code.py"
    preset = aug_winner.get("aug_preset", "medium")
    final_cache = ma.META_LOGS / "birdnet_cache" / "final"
    final_cache.mkdir(parents=True, exist_ok=True)
    full_train = final_cache / f"train_emb_{preset}_full.npz"
    if cfg.get("rebuild_full_train_cache", True) or not full_train.exists():
        aug_dict = ma._load_aug_dict_for_1c_winner(aug_winner, ma._baseline_names(config))
        embed_cap = ma._embed_max_samples(config)
        d1d_override = {
            "meta_aug_preset": preset,
            "augmentation": aug_dict,
            "train_sample_frac": float(cfg.get("embed_sample_frac", 1.0)),
            "force_rebuild_cache": True,
            "birdnet_build_cache_only": True,
            "train_cache_path": str(full_train),
            "birdnet": {"cache_dir": str(final_cache), "val_cache_path": str(ma.BIRDNET_SHARED_VAL_CACHE)},
        }
        if embed_cap is not None:
            d1d_override["max_train_samples"] = embed_cap
        ma._run_subprocess("birdnet_agent.py", d1d_override, config)
    ma.BIRDNET_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = ma.BIRDNET_FINAL_DIR / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    head_code = head_path.read_text(encoding="utf-8")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from birdnet_staged import _build_final_retrain_script
    from code_executor import CodeExecutor

    script = _build_final_retrain_script(
        head_code, ma.BIRDNET_FINAL_DIR, full_train, ma.BIRDNET_SHARED_VAL_CACHE
    )
    script_path = code_dir / "final_retrain.py"
    script_path.write_text(script, encoding="utf-8")
    timeout = int(cfg.get("final_timeout_seconds") or 7200)
    executor = CodeExecutor(
        python_executable=config.get("execution", {}).get("python_executable", "python3"),
        timeout_seconds=timeout,
    )
    result = executor.run_file(script_path)
    ok = result.success and "FINAL_RETRAIN_DONE" in (result.stdout or "")
    shutil.copy2(head_path, ma.BIRDNET_FINAL_DIR / "best_head_code.py")
    if ok and cfg.get("copy_to_submission", True):
        sub = PROJECT_ROOT / "submission"
        sub.mkdir(parents=True, exist_ok=True)
        w = ma.BIRDNET_FINAL_DIR / "final_head.weights.h5"
        if w.exists():
            shutil.copy2(w, sub / "birdnet_final_head.weights.h5")
        keras_m = ma.BIRDNET_FINAL_DIR / "final_head.keras"
        if keras_m.exists():
            shutil.copy2(keras_m, sub / "birdnet_final_head.keras")
        shutil.copy2(head_path, sub / "birdnet_best_head_code.py")
    summary = {
        "stage": "1d_birdnet_final_train",
        "success": ok,
        "aug_preset": preset,
        "train_cache": str(full_train),
        "output_dir": str(ma.BIRDNET_FINAL_DIR),
        "locked_head_code": str(head_path),
    }
    ma.BIRDNET_ARCH_1D_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_1e_birdnet_pseudo_refine(config: dict, suite) -> dict:
    """Stage 1e: pseudo-label unlabeled soundscapes + fine-tune the 1d BirdNET head."""
    ma = _ma()
    cfg = ma._birdnet_stage_cfg(config, "birdnet_stage_1e")
    if not cfg.get("enabled", False):
        return {"success": False, "skipped": True}

    if ma._skip_if_completed(config, ma.BIRDNET_ARCH_1E_RESULTS, "BirdNET 1e"):
        try:
            return json.loads(ma.BIRDNET_ARCH_1E_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    teacher_keras = ma.BIRDNET_FINAL_DIR / "final_head.keras"
    teacher_weights = ma.BIRDNET_FINAL_DIR / "final_head.weights.h5"
    if not teacher_keras.exists() and not teacher_weights.exists():
        print("\n  [BirdNET 1e] No stage-1d final head — run birdnet_stage_1d first.")
        return {"stage": "1e_birdnet_pseudo_refine", "success": False}

    train_cache: Path | None = None
    preset = "medium"
    if ma.BIRDNET_ARCH_1D_RESULTS.exists():
        try:
            s1d = json.loads(ma.BIRDNET_ARCH_1D_RESULTS.read_text(encoding="utf-8"))
            if s1d.get("train_cache"):
                train_cache = Path(s1d["train_cache"])
            preset = str(s1d.get("aug_preset", preset))
        except (json.JSONDecodeError, OSError):
            pass
    if train_cache is None or not train_cache.exists():
        train_cache = ma.META_LOGS / "birdnet_cache" / "final" / f"train_emb_{preset}_full.npz"
    if not train_cache.exists():
        print(f"\n  [BirdNET 1e] Train cache missing: {train_cache}")
        return {"stage": "1e_birdnet_pseudo_refine", "success": False}

    head_code_path: Path | None = None
    if ma.BIRDNET_ARCH_1C_RESULTS.exists():
        try:
            s1c = json.loads(ma.BIRDNET_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
            locked = s1c.get("locked_head_code")
            if locked:
                head_code_path = Path(locked)
        except (json.JSONDecodeError, OSError):
            pass
    for candidate in (
        ma.BIRDNET_FINAL_DIR / "best_head_code.py",
        PROJECT_ROOT / "submission" / "birdnet_best_head_code.py",
    ):
        if candidate.exists():
            head_code_path = candidate
            break
    if head_code_path is None or not head_code_path.exists():
        print("\n  [BirdNET 1e] Missing best_head_code.py — cannot refine architecture.")
        return {"stage": "1e_birdnet_pseudo_refine", "success": False}

    top1 = float(cfg.get("top1_threshold", 0.55))
    runnerup = float(cfg.get("runnerup_max", 0.35))
    pl_weight = float(cfg.get("pseudo_label_weight", 0.8))
    sw_sup = float(cfg.get("sample_weight_supervised", 1.0))
    sw_val = float(cfg.get("sample_weight_labeled_val", sw_sup))
    sw_ps = float(cfg.get("sample_weight_pseudo", 0.5))
    rebuild = bool(cfg.get("rebuild_pseudo_cache", False))
    include_labeled_val = bool(cfg.get("include_labeled_val_in_refine", True))
    val_cache = (
        ma.BIRDNET_SHARED_VAL_CACHE
        if include_labeled_val and ma.BIRDNET_SHARED_VAL_CACHE.exists()
        else None
    )

    print("\n" + "=" * 60)
    print("  BIRDNET STAGED — Step 1e: Pseudo-label refine")
    print(f"  Thresholds: top1≥{top1}  runner-up<{runnerup}  soft_label×{pl_weight}")
    print(f"  Supervised cache: {train_cache.name}")
    if val_cache is not None:
        print(f"  Labeled val cache: {val_cache.name}")
    print("=" * 60)

    pseudo_stats: dict = {}
    if rebuild or not ma.BIRDNET_PSEUDO_LABELS_NPZ.exists():
        from birdnet_pseudo import build_birdnet_pseudo_label_cache

        teacher_path = teacher_keras if teacher_keras.exists() else teacher_weights
        pseudo_stats = build_birdnet_pseudo_label_cache(
            config=config,
            teacher_head_path=teacher_path,
            out_path=ma.BIRDNET_PSEUDO_LABELS_NPZ,
            soundscapes_dir=ma.TRAIN_SOUNDSCAPES,
            labels_csv=ma.SOUNDSCAPE_LABELS,
            top1_threshold=top1,
            runnerup_max=runnerup,
            pseudo_label_weight=pl_weight,
            train_cache=train_cache,
            embed_batch_size=int((config.get("birdnet") or {}).get("embed_batch_size", 32)),
            max_files=cfg.get("max_soundscape_files"),
        )
        if pseudo_stats.get("empty_pseudo"):
            print("\n  [BirdNET 1e] Continuing with supervised-only fine-tune (no pseudo windows).")
    else:
        d = np.load(str(ma.BIRDNET_PSEUDO_LABELS_NPZ), allow_pickle=True)
        pseudo_stats = {
            "n_accepted": int(d["n_accepted"]) if "n_accepted" in d.files else int(d["X_pseudo"].shape[0]),
            "out_path": str(ma.BIRDNET_PSEUDO_LABELS_NPZ),
            "reused_cache": True,
        }
        print(
            f"\n  [BirdNET 1e] Reusing pseudo cache → {ma.BIRDNET_PSEUDO_LABELS_NPZ.name} "
            f"({pseudo_stats['n_accepted']} windows)"
        )

    from birdnet_staged import _build_pseudo_refine_script
    from code_executor import CodeExecutor

    ma.BIRDNET_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = ma.BIRDNET_FINAL_DIR / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    head_code = head_code_path.read_text(encoding="utf-8")

    timeout = int(cfg.get("refine_timeout_seconds") or config.get("execution", {}).get("timeout_seconds", 1800))
    executor = CodeExecutor(
        python_executable=config.get("execution", {}).get("python_executable", "python3"),
        timeout_seconds=timeout,
    )

    init_weights = teacher_keras if teacher_keras.exists() else teacher_weights
    script = _build_pseudo_refine_script(
        head_code,
        ma.BIRDNET_FINAL_DIR,
        train_cache,
        ma.BIRDNET_PSEUDO_LABELS_NPZ,
        val_cache=val_cache,
        init_weights=init_weights,
        fine_tune_lr=float(cfg.get("fine_tune_lr", 2e-4)),
        epochs=int(cfg.get("fine_tune_epochs", 15)),
        val_split=float(cfg.get("val_split", 0.1)),
        sample_weight_supervised=sw_sup,
        sample_weight_labeled_val=sw_val,
        sample_weight_pseudo=sw_ps,
    )
    script_path = code_dir / "pseudo_refine.py"
    script_path.write_text(script, encoding="utf-8")

    score_1d: dict | None = None
    if teacher_keras.exists():
        for stale in ("final_head_pseudo.keras", "final_head_pseudo.weights.h5"):
            (ma.BIRDNET_FINAL_DIR / stale).unlink(missing_ok=True)
        sc_before = suite.score_birdnet_mem_dir(
            ma.BIRDNET_FINAL_DIR, val_cache=ma.BIRDNET_SHARED_VAL_CACHE
        )
        if sc_before:
            score_1d = sc_before.to_dict()
            ma._print_soundscape_score("BirdNET/1d (pre-pseudo head)", sc_before)

    print("\n  [BirdNET 1e] Fine-tuning head on supervised + pseudo embeddings…")
    result = executor.run_file(script_path)
    ok = result.success and "PSEUDO_REFINE_DONE" in (result.stdout or "")

    score_1e: dict | None = None
    if ok:
        pseudo_keras = ma.BIRDNET_FINAL_DIR / "final_head_pseudo.keras"
        pseudo_weights = ma.BIRDNET_FINAL_DIR / "final_head_pseudo.weights.h5"
        print(f"  [BirdNET 1e] Saved → {pseudo_keras.name}")
        if pseudo_keras.exists():
            shutil.copy2(pseudo_keras, ma.BIRDNET_FINAL_DIR / "best_head.keras")
        if pseudo_weights.exists():
            shutil.copy2(pseudo_weights, ma.BIRDNET_FINAL_DIR / "best_head.weights.h5")
        sc_after = suite.score_birdnet_mem_dir(
            ma.BIRDNET_FINAL_DIR, val_cache=ma.BIRDNET_SHARED_VAL_CACHE
        )
        if sc_after:
            score_1e = sc_after.to_dict()
            ma._print_soundscape_score("BirdNET/1e (pseudo-refined)", sc_after)
        if cfg.get("copy_to_submission", True):
            sub = PROJECT_ROOT / "submission"
            sub.mkdir(parents=True, exist_ok=True)
            if pseudo_keras.exists():
                shutil.copy2(pseudo_keras, sub / "birdnet_final_head_pseudo.keras")
                shutil.copy2(pseudo_keras, sub / "birdnet_final_head.keras")
            if pseudo_weights.exists():
                shutil.copy2(pseudo_weights, sub / "birdnet_final_head_pseudo.weights.h5")
                shutil.copy2(pseudo_weights, sub / "birdnet_final_head.weights.h5")
            shutil.copy2(head_code_path, sub / "birdnet_best_head_code.py")
            print(f"  [BirdNET 1e] Submission bundle updated → {sub}")
    else:
        print(f"  [BirdNET 1e] Pseudo refine failed: {(result.stderr or '')[-500:]}")

    summary = {
        "stage": "1e_birdnet_pseudo_refine",
        "success": ok,
        "train_cache": str(train_cache),
        "pseudo_cache": str(ma.BIRDNET_PSEUDO_LABELS_NPZ),
        "pseudo_stats": pseudo_stats,
        "thresholds": {"top1": top1, "runnerup_max": runnerup, "pseudo_label_weight": pl_weight},
        "output_dir": str(ma.BIRDNET_FINAL_DIR),
        "score_1d": score_1d,
        "score_1e": score_1e,
    }
    ma.BIRDNET_ARCH_1E_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
