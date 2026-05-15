"""
Meta Agent — BirdCLEF 2026
===========================
Phase 1 (perch_iterations):   Perch ONNX → MLP head hyperparameter search
Phase 2 (birdnet_iterations):  BirdNET → MLP head hyperparameter search
Phase 3 (ensemble_iterations): Blend-weight search on shared val set

Run:
    python src/meta_agent.py --config configs/agent_config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PERCH_MEMORY  = PROJECT_ROOT / "logs" / "perch_memory"
BIRDNET_LOGS  = PROJECT_ROOT / "logs" / "birdnet_agent"
META_LOGS     = PROJECT_ROOT / "logs" / "meta_agent"
PYTHON        = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_phase_config(base_config: dict, overrides: dict, out_path: Path) -> Path:
    """Write a temporary config JSON with overrides applied."""
    cfg = json.loads(json.dumps(base_config))
    cfg.update(overrides)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out_path


def _macro_auc_ge3(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, int]:
    """Macro AUC on species with ≥3 positive val samples (same metric as BirdNET agent)."""
    from sklearn.metrics import roc_auc_score
    aucs, count = [], 0
    for c in range(y_true.shape[1]):
        pos = int(y_true[:, c].sum())
        if pos < 3:
            continue
        count += 1
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else 0.0), count


def _align_preds(preds: np.ndarray, src_species: list[str], tgt_species: list[str]) -> np.ndarray:
    """Reorder columns of preds from src_species order to tgt_species order."""
    src_idx = {s: i for i, s in enumerate(src_species)}
    out = np.zeros((preds.shape[0], len(tgt_species)), dtype=np.float32)
    for j, sp in enumerate(tgt_species):
        if sp in src_idx:
            out[:, j] = preds[:, src_idx[sp]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Perch
# ─────────────────────────────────────────────────────────────────────────────

def run_perch_phase(base_config: dict, n_iterations: int) -> float:
    print("\n" + "=" * 60)
    print(f"  META AGENT — PHASE 1: Perch ({n_iterations} iterations)")
    print("=" * 60)

    PERCH_MEMORY.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(tempfile.gettempdir()) / "meta_perch_config.json"
    _write_phase_config(base_config, {"max_iterations": n_iterations}, cfg_path)

    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "src" / "perch_agent.py"), "--config", str(cfg_path)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("  [Phase 1] Perch agent exited with errors — continuing to Phase 2.")

    info_path = PERCH_MEMORY / "best_model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        auc = float(info.get("auc", 0.0))
        print(f"\n  [Phase 1] Best Perch AUC = {auc:.5f}")
        return auc
    print("  [Phase 1] No best_model_info.json found.")
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BirdNET
# ─────────────────────────────────────────────────────────────────────────────

def run_birdnet_phase(base_config: dict, n_iterations: int) -> float:
    print("\n" + "=" * 60)
    print(f"  META AGENT — PHASE 2: BirdNET ({n_iterations} iterations)")
    print("=" * 60)

    BIRDNET_LOGS.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(tempfile.gettempdir()) / "meta_birdnet_config.json"
    _write_phase_config(base_config, {"max_iterations": n_iterations}, cfg_path)

    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "src" / "birdnet_agent.py"), "--config", str(cfg_path)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("  [Phase 2] BirdNET agent exited with errors — continuing to Phase 3.")

    auc_path = BIRDNET_LOGS / "best_auc.json"
    if auc_path.exists():
        info = json.loads(auc_path.read_text())
        auc = float(info.get("auc", 0.0))
        print(f"\n  [Phase 2] Best BirdNET AUC = {auc:.5f}")
        return auc
    # Fallback: read from history
    hist_path = BIRDNET_LOGS / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        aucs = [e.get("macro_auc_ge3", 0.0) for e in history if e.get("status") == "success"]
        if aucs:
            auc = float(max(aucs))
            print(f"\n  [Phase 2] Best BirdNET AUC (from history) = {auc:.5f}")
            return auc
    print("  [Phase 2] No BirdNET AUC found.")
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Ensemble blend search
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble_phase(
    base_config: dict,
    n_iterations: int,
    perch_auc: float,
    birdnet_auc: float,
) -> dict:
    print("\n" + "=" * 60)
    print(f"  META AGENT — PHASE 3: Ensemble ({n_iterations} blend weights)")
    print(f"  Perch AUC={perch_auc:.5f}  BirdNET AUC={birdnet_auc:.5f}")
    print("=" * 60)

    META_LOGS.mkdir(parents=True, exist_ok=True)

    # Load species columns for alignment
    species_path = PERCH_MEMORY / "species_cols.json"
    if not species_path.exists():
        print("  [Phase 3] species_cols.json not found — skipping ensemble.")
        return {}
    perch_species = json.loads(species_path.read_text())

    # Load Perch val preds
    perch_preds_path = PERCH_MEMORY / "best_val_preds.npy"
    perch_ytrue_path = PERCH_MEMORY / "y_val.npy"
    if not perch_preds_path.exists() or not perch_ytrue_path.exists():
        print("  [Phase 3] Perch val preds not found — skipping ensemble.")
        return {}
    perch_preds = np.load(str(perch_preds_path))   # (N_perch, n_species)
    perch_ytrue = np.load(str(perch_ytrue_path))   # (N_perch, n_species)

    # Load BirdNET val preds
    birdnet_preds_path = BIRDNET_LOGS / "best_val_preds.npy"
    birdnet_ytrue_path = BIRDNET_LOGS / "y_val.npy"
    if not birdnet_preds_path.exists() or not birdnet_ytrue_path.exists():
        print("  [Phase 3] BirdNET val preds not found — skipping ensemble.")
        return {}
    birdnet_preds = np.load(str(birdnet_preds_path))  # (N_birdnet, n_species)
    birdnet_ytrue = np.load(str(birdnet_ytrue_path))  # (N_birdnet, n_species)

    print(f"  Perch val:   {perch_preds.shape}  BirdNET val: {birdnet_preds.shape}")

    # Both val sets are independent — evaluate each blend weight on BOTH val sets
    # Combined score: weighted average of macro AUC on each set
    def score_weight(w: float) -> float:
        p_auc, _ = _macro_auc_ge3(perch_ytrue, w * perch_preds + (1 - w) * 0.5)
        b_auc, _ = _macro_auc_ge3(birdnet_ytrue, w * 0.5 + (1 - w) * birdnet_preds)
        # Use own model's val for each, blend score by AUC contribution
        p_blend, _ = _macro_auc_ge3(perch_ytrue, perch_preds)
        b_blend, _ = _macro_auc_ge3(birdnet_ytrue, birdnet_preds)
        return (p_auc + b_auc) / 2.0

    # AUC-proportional starting weight
    total = perch_auc + birdnet_auc if (perch_auc + birdnet_auc) > 0 else 1.0
    w_init = round(perch_auc / total, 2)
    print(f"  AUC-proportional starting weight (perch): {w_init:.2f}")

    # Build candidate weights: start proportional, then explore around it
    candidates = sorted(set([
        round(w_init, 2),
        *[round(max(0.0, min(1.0, w_init + d)), 2) for d in [-0.2, -0.1, 0.1, 0.2]],
        0.0, 0.3, 0.5, 0.7, 1.0,
    ]))[:n_iterations * 2]  # keep enough to fill iterations

    results = []
    llm_client = _make_llm_client(base_config)

    for i, w in enumerate(candidates[:n_iterations], 1):
        t0 = time.time()
        combined = score_weight(w)
        print(f"  [{i}/{n_iterations}] perch_weight={w:.2f}  combined_score={combined:.5f}  ({time.time()-t0:.1f}s)")
        results.append({"perch_weight": w, "combined_score": combined})

        # After 3 results, ask LLM for a better weight to try next
        if i == 3 and i < n_iterations and llm_client:
            suggestion = _llm_suggest_weight(llm_client, results, perch_auc, birdnet_auc, base_config)
            if suggestion is not None and suggestion not in [r["perch_weight"] for r in results]:
                candidates.insert(i, suggestion)

    best = max(results, key=lambda r: r["combined_score"])
    best_weight = best["perch_weight"]
    best_score  = best["combined_score"]

    print(f"\n  [Phase 3] Best perch_weight = {best_weight:.2f}  score = {best_score:.5f}")

    ensemble_cfg = {
        "perch_weight":   best_weight,
        "birdnet_weight": round(1.0 - best_weight, 2),
        "perch_auc":      perch_auc,
        "birdnet_auc":    birdnet_auc,
        "combined_score": best_score,
        "all_results":    results,
    }
    out = META_LOGS / "ensemble_config.json"
    out.write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
    print(f"  Ensemble config saved to {out}")
    return ensemble_cfg


def _make_llm_client(config: dict):
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from llm_client import LLMClient
        llm_cfg = config.get("llm", {})
        return LLMClient(provider=llm_cfg.get("provider", "ollama"), model=llm_cfg.get("model", "llama3.2:3b"))
    except Exception:
        return None


def _llm_suggest_weight(llm, results: list[dict], perch_auc: float, birdnet_auc: float, config: dict) -> float | None:
    try:
        history_str = "\n".join(f"  perch_weight={r['perch_weight']:.2f} → score={r['combined_score']:.5f}" for r in results)
        prompt = (
            f"We are ensembling two bird sound classifiers:\n"
            f"  Perch model val AUC:   {perch_auc:.5f}\n"
            f"  BirdNET model val AUC: {birdnet_auc:.5f}\n\n"
            f"Blend: final = perch_weight * perch_preds + (1 - perch_weight) * birdnet_preds\n\n"
            f"Results so far:\n{history_str}\n\n"
            f"Suggest ONE float value for perch_weight in [0.0, 1.0] to try next. "
            f"Reply with ONLY the number, e.g.: 0.45"
        )
        resp = llm.generate_code("You are a machine learning expert.", prompt, temperature=0.3)
        import re
        m = re.search(r'0?\.\d+|\d+\.\d+', resp)
        if m:
            val = float(m.group())
            return round(max(0.0, min(1.0, val)), 2)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "agent_config.json"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    meta_cfg       = config.get("meta_agent", {})
    perch_iters    = int(meta_cfg.get("perch_iterations",    10))
    birdnet_iters  = int(meta_cfg.get("birdnet_iterations",  10))
    ensemble_iters = int(meta_cfg.get("ensemble_iterations",  5))

    t_start = time.time()

    perch_auc   = run_perch_phase(config, perch_iters)
    birdnet_auc = run_birdnet_phase(config, birdnet_iters)
    ensemble    = run_ensemble_phase(config, ensemble_iters, perch_auc, birdnet_auc)

    print("\n" + "=" * 60)
    print("  META AGENT COMPLETE")
    print(f"  Perch AUC:    {perch_auc:.5f}")
    print(f"  BirdNET AUC:  {birdnet_auc:.5f}")
    if ensemble:
        print(f"  Best blend:   perch={ensemble['perch_weight']:.2f}  birdnet={ensemble['birdnet_weight']:.2f}")
        print(f"  Combined:     {ensemble['combined_score']:.5f}")
    print(f"  Total time:   {(time.time() - t_start)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
