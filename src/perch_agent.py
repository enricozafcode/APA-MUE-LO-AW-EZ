"""
perch_agent.py — Autonomous Perch-based BirdCLEF agent.

Researcher/Coder architecture over Google Perch 1536-d embeddings (ONNX).

Pipeline:
  1. Auto-install onnxruntime + kagglehub if missing
  2. Auto-download Perch ONNX model + labels CSV via kagglehub
  3. Build embedding cache ONCE: AudioAugmenter → ONNX Perch → (X, S, y) .npz
     X = 1536-d embeddings, S = mapped Perch logit scores (234 species)
  4. Build soundscape validation cache ONCE from train_soundscapes_labels.csv
  5. Agent loop (max_iterations):
     a. Researcher reads memory → produces JSON spec
     b. Coder writes build_head(emb_dim, n_classes) + get_training_config()
     c. Harness loads caches, trains head, blends with Perch logit scores
     d. Evaluate macro ROC-AUC on soundscape windows
     e. Log to persistent memory

Run:
    python src/perch_agent.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from memory import ExperimentMemory


# ─────────────────────────────────────────────────────────────────────────────
# Dependency management
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_deps() -> None:
    """Install onnxruntime and kagglehub if not already available."""
    missing = []
    for pkg, imp in [("onnxruntime", "onnxruntime"), ("kagglehub", "kagglehub")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  [Setup] Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("  [Setup] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Model + label downloads
# ─────────────────────────────────────────────────────────────────────────────

def _find_or_download_onnx(dataset_slug: str) -> Path:
    """Download Perch ONNX model via kagglehub; return path to .onnx file."""
    import kagglehub
    print(f"  [Setup] Locating ONNX model ({dataset_slug})...")
    onnx_dir = Path(kagglehub.dataset_download(dataset_slug))
    onnx_files = sorted(onnx_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx file found in {onnx_dir}.\n"
            f"Make sure kagglehub is authenticated: kagglehub.login()"
        )
    print(f"  [Setup] ONNX model: {onnx_files[0]}")
    return onnx_files[0]


def _find_or_download_perch_labels(model_slug: str) -> Path:
    """Download Perch SavedModel (for labels.csv) via kagglehub."""
    import kagglehub
    print(f"  [Setup] Locating Perch labels ({model_slug})...")
    model_dir = Path(kagglehub.model_download(model_slug))
    label_files = sorted(model_dir.rglob("labels.csv"))
    if not label_files:
        raise FileNotFoundError(
            f"No labels.csv found in {model_dir}.\n"
            f"Make sure kagglehub is authenticated: kagglehub.login()"
        )
    print(f"  [Setup] Labels CSV: {label_files[0]}")
    return label_files[0]


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Perch session
# ─────────────────────────────────────────────────────────────────────────────

def _load_onnx_session(onnx_path: Path):
    """Load ONNX session; auto-detect embedding (1536-d) and logit outputs."""
    import onnxruntime as ort
    import numpy as np
    print("  [Perch] Loading ONNX session...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name

    # Smoke test to identify output indices
    dummy = np.zeros((1, 160_000), dtype=np.float32)
    outs = sess.run(None, {inp_name: dummy})
    emb_idx, logit_idx = None, None
    for i, arr in enumerate(outs):
        if arr.ndim == 2 and arr.shape[-1] == 1536:
            emb_idx = i
        elif arr.ndim == 2 and arr.shape[-1] > 5_000:
            logit_idx = i
    if emb_idx is None:
        raise RuntimeError("Could not find 1536-d embedding output in ONNX model")
    if logit_idx is None:
        raise RuntimeError("Could not find large logits output (>5000 classes) in ONNX model")
    print(f"  [Perch] Loaded. Embedding index={emb_idx}  Logits index={logit_idx}")
    return sess, inp_name, emb_idx, logit_idx


def _perch_embed_batch(sess, inp_name: str, emb_idx: int, logit_idx: int, waveforms):
    """Run a batch of 5-second waveforms through Perch → (embeddings, logits)."""
    import numpy as np
    PERCH_SAMPLES = 160_000
    if isinstance(waveforms, list):
        batch = np.stack(waveforms, axis=0).astype(np.float32)
    else:
        batch = np.asarray(waveforms, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch[None, :]
    if batch.shape[1] != PERCH_SAMPLES:
        fixed = np.zeros((batch.shape[0], PERCH_SAMPLES), dtype=np.float32)
        for i in range(len(batch)):
            n = min(batch.shape[1], PERCH_SAMPLES)
            fixed[i, :n] = batch[i, :n]
        batch = fixed
    outs = sess.run(None, {inp_name: batch})
    return outs[emb_idx].astype(np.float32), outs[logit_idx].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Direct logit mapping: Perch vocab (~14k) → BirdCLEF 234 species
# ─────────────────────────────────────────────────────────────────────────────

def _build_logit_mapping(labels_csv: Path, taxonomy_df, species_cols: list):
    """
    Map Perch's internal species indices to BirdCLEF species positions.
    Unmapped species get a genus-level proxy if one exists.
    Returns: (MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
    """
    import numpy as np
    import pandas as pd

    perch_labels = pd.read_csv(labels_csv).reset_index().rename(columns={"index": "perch_idx"})
    sci_col = next(
        (c for c in perch_labels.columns if any(k in c.lower() for k in ["sci", "name"])),
        perch_labels.columns[1],
    )
    perch_sci_to_idx = {
        str(row[sci_col]).strip(): int(row["perch_idx"])
        for _, row in perch_labels.iterrows()
    }
    NO_LABEL = len(perch_labels)

    tax_sci   = taxonomy_df.set_index("primary_label")["scientific_name"].to_dict()
    tax_class = taxonomy_df.set_index("primary_label")["class_name"].to_dict()

    BC_INDICES = np.array([
        perch_sci_to_idx.get(str(tax_sci.get(sp, "")).strip(), NO_LABEL)
        for sp in species_cols
    ], dtype=np.int32)

    MAPPED_MASK   = BC_INDICES != NO_LABEL
    MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
    UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

    # Genus-level proxy for classes not directly in Perch vocab
    PROXY_TAXA = {"Aves", "Amphibia", "Insecta", "Reptilia"}
    proxy_map: dict[int, list[int]] = {}
    for sp_idx in UNMAPPED_POS:
        sp  = species_cols[int(sp_idx)]
        cls = tax_class.get(sp, "")
        if cls not in PROXY_TAXA:
            continue
        sci = str(tax_sci.get(sp, "")).strip()
        genus = sci.split()[0] if sci else ""
        if not genus:
            continue
        pat = re.compile(rf"^{re.escape(genus)}\s")
        hits = [pidx for psci, pidx in perch_sci_to_idx.items() if pat.match(psci)]
        if hits:
            proxy_map[int(sp_idx)] = hits

    n = len(species_cols)
    print(
        f"  [Mapping] Direct: {MAPPED_MASK.sum()}/{n} | "
        f"Genus proxy: {len(proxy_map)}/{(~MAPPED_MASK).sum()} unmapped"
    )
    return MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL


def _apply_logit_mapping(logits, n_species: int, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL):
    """Convert Perch logits (B, vocab) → BirdCLEF species scores (B, 234)."""
    import numpy as np
    B = logits.shape[0]
    out = np.broadcast_to(
        logits.mean(axis=1, keepdims=True), (B, n_species)
    ).astype(np.float32).copy()
    out[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for sp_idx, bc_idxs in proxy_map.items():
        out[:, sp_idx] = logits[:, bc_idxs].mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_pad(audio_path: Path, sr: int = 32_000, clip_sec: float = 5.0):
    import numpy as np
    import librosa
    target = int(sr * clip_sec)
    wav, _ = librosa.load(str(audio_path), sr=sr, mono=True, duration=clip_sec)
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


def _enforce_length(wav, target: int):
    import numpy as np
    if len(wav) < target:
        return np.pad(wav, (0, target - len(wav)))
    return wav[:target]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache — built once, reused across all agent iterations
# ─────────────────────────────────────────────────────────────────────────────

def _build_train_cache(
    sess, inp_name, emb_idx, logit_idx,
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
    train_df, species_to_idx: dict, audio_dir: Path, n_species: int,
    cache_path: Path, max_samples=None, batch_size: int = 16,
    sr: int = 32_000, clip_sec: float = 5.0,
) -> None:
    """Embed all training audio clips with AudioAugmenter → save (X, S, y).npz."""
    import numpy as np
    from augmentation import AudioAugmenter

    aug_cfg = {
        "noise_injection": {"enabled": True,  "probability": 0.5, "noise_level": 0.007},
        "time_shift":      {"enabled": True,  "probability": 0.3, "shift_max_fraction": 0.2},
        "time_stretch":    {"enabled": False},
        "pitch_shift":     {"enabled": False},
    }
    augmenter = AudioAugmenter(aug_cfg)

    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename"      if "filename"      in train_df.columns else "filepath"
    rows = [(lb, fn) for lb, fn in train_df[[lcol, fcol]].dropna().values.tolist()
            if lb in species_to_idx]

    if max_samples and len(rows) > max_samples:
        import random; random.seed(42); random.shuffle(rows); rows = rows[:max_samples]

    print(f"  [Cache] Building train cache: {len(rows)} clips → {cache_path.name}")
    target = int(sr * clip_sec)
    X_parts, S_parts, y_parts = [], [], []
    batch_wavs, batch_labels = [], []

    def _flush():
        if not batch_wavs:
            return
        embs, logits = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, batch_wavs)
        scores = _apply_logit_mapping(logits, n_species, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
        X_parts.append(embs); S_parts.append(scores)
        y_parts.extend(batch_labels)

    for i, (label, fname) in enumerate(rows):
        p = audio_dir / fname
        if not p.exists():
            continue
        try:
            wav = _load_and_pad(p, sr, clip_sec)
            wav = augmenter.apply(wav, sr)
            wav = _enforce_length(wav, target)
        except Exception:
            continue

        vec = np.zeros(n_species, dtype=np.float32)
        vec[species_to_idx[label]] = 1.0
        batch_wavs.append(wav)
        batch_labels.append(vec)

        if len(batch_wavs) >= batch_size:
            _flush()
            batch_wavs.clear()
            batch_labels.clear()

        if (i + 1) % 1000 == 0 or (i + 1) == len(rows):
            print(f"    {i+1}/{len(rows)} files processed...", flush=True)

    _flush()
    if not X_parts:
        raise RuntimeError("No samples could be embedded for training cache.")

    X = np.concatenate(X_parts).astype(np.float32)
    S = np.concatenate(S_parts).astype(np.float32)
    y = np.stack(y_parts).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, S=S, y=y)
    print(f"  [Cache] Saved: X={X.shape}  S={S.shape}  y={y.shape}")


def _build_val_cache(
    sess, inp_name, emb_idx, logit_idx,
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
    soundscapes_dir: Path, labels_csv: Path,
    species_to_idx: dict, n_species: int,
    cache_path: Path, batch_size: int = 16,
    sr: int = 32_000, clip_sec: float = 5.0,
) -> bool:
    """Embed soundscape validation windows → save (X, S, y).npz. Returns True on success."""
    import numpy as np
    import pandas as pd
    import librosa

    if not labels_csv.exists():
        return False

    lab = pd.read_csv(labels_csv)

    def _tok(v):
        if pd.isna(v) or str(v).strip() == "":
            return set()
        return {t.strip() for t in str(v).split(";") if t.strip()}

    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set.union(*[_tok(v) for v in s]))
        .reset_index()
    )
    grp["start_sec"] = pd.to_timedelta(grp["start"]).dt.total_seconds().astype(float)
    grp["end_sec"]   = pd.to_timedelta(grp["end"]).dt.total_seconds().astype(int)

    target = int(sr * clip_sec)
    wavs, labels = [], []

    for _, row in grp.iterrows():
        fp = soundscapes_dir / row["filename"]
        if not fp.exists():
            continue
        try:
            offset = float(row["start_sec"])
            wav, _ = librosa.load(str(fp), sr=sr, mono=True, offset=offset, duration=clip_sec)
            wav = _enforce_length(wav.astype(np.float32), target)
        except Exception:
            continue
        vec = np.zeros(n_species, dtype=np.float32)
        for code in row["primary_label"]:
            j = species_to_idx.get(str(code))
            if j is not None:
                vec[j] = 1.0
        wavs.append(wav)
        labels.append(vec)

    if not wavs:
        return False

    print(f"  [Cache] Building val cache: {len(wavs)} soundscape windows → {cache_path.name}")
    X_parts, S_parts = [], []
    for start in range(0, len(wavs), batch_size):
        chunk = wavs[start:start + batch_size]
        embs, logits = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, chunk)
        scores = _apply_logit_mapping(logits, n_species, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
        X_parts.append(embs); S_parts.append(scores)

    X = np.concatenate(X_parts).astype(np.float32)
    S = np.concatenate(S_parts).astype(np.float32)
    y = np.stack(labels).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, S=S, y=y)
    print(f"  [Cache] Saved val: X={X.shape}  S={S.shape}  y={y.shape}")
    return True


def _build_focal_val_fallback(train_cache: Path, val_cache: Path) -> None:
    """Fallback: use last 10% of training cache as validation set."""
    import numpy as np
    d = np.load(str(train_cache))
    X, S, y = d["X"], d["S"], d["y"]
    n_val = max(50, int(len(X) * 0.1))
    np.savez_compressed(str(val_cache), X=X[-n_val:], S=S[-n_val:], y=y[-n_val:])
    print(f"  [Cache] Fallback val: {n_val} samples from end of train cache")


# ─────────────────────────────────────────────────────────────────────────────
# Perch Researcher — outer loop, reads history, decides head config
# ─────────────────────────────────────────────────────────────────────────────

def _extract_first_json_object(text: str) -> dict | None:
    """Find the first complete JSON object in text by counting braces."""
    start = text.find('{')
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PERCH_SEARCH_SPACE = {
    "n_blocks":      [1, 2, 3],
    "hidden_dim":    [512, 1024],
    "proj_dim":      [256, 512],
    "dropout_block": [0.1, 0.2, 0.3, 0.4],
    "dropout_final": [0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-2, 1e-3, 8e-4, 5e-4, 1e-4],
    "batch_size":    [128, 256, 512],
    "optimizer":     ["adam", "sgd_momentum"],
    "epochs":        [15, 25, 40],
    "patience":      [3, 5, 7],
    "perch_weight":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

PERCH_RESEARCHER_SYSTEM_PROMPT = """You are an expert ML researcher optimizing a BirdCLEF head model on top of frozen Google Perch embeddings.
Audio is already encoded as 1536-d vectors by Perch — you control only the classification head architecture.
Final predictions blend the head output with direct Perch logit scores:
  y_pred = perch_weight * perch_scores + (1 - perch_weight) * head_output

Reason carefully about:
- Which head depth (n_blocks), width (hidden_dim, proj_dim), and dropout improved AUC
- Whether a higher perch_weight (trusting Perch's own labels more) helps
- Which learning rates and batch sizes worked
- What has NOT been tried yet

You MUST respond with ONLY a single JSON object — no prose, no explanation, no markdown, no code fences.
Start your response with { and end with }.

Example response format:
{"n_blocks": 2, "hidden_dim": 1024, "proj_dim": 512, "dropout_block": 0.3, "dropout_final": 0.4, "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 25, "patience": 5, "perch_weight": 0.2, "reasoning": "First run, establishing baseline.", "hypothesis": "Standard residual MLP should generalize well.", "strategy": "explore"}

Required keys: n_blocks, hidden_dim, proj_dim, dropout_block, dropout_final, learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."""


class PerchResearcher:
    def __init__(self, llm: LLMClient, memory: ExperimentMemory, temperature: float = 0.6) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature

    def next_experiment(self) -> dict:
        history  = self.memory.researcher_context()
        best     = self.memory.best_runs(1)
        best_auc = best[0]["macro_roc_auc"] if best else None
        total    = self.memory.total()
        best_str = f"{best_auc:.5f}" if best_auc is not None else "none"

        user_prompt = (
            f"{history}\n\n"
            f"Search space:\n{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
            f"Total experiments: {total}\n"
            f"Best AUC: {best_str}\n\n"
            "Pick the next experiment. Respond with ONLY a JSON object — no prose, no markdown.\n"
            "Start with { and end with }. Include all keys: "
            "n_blocks, hidden_dim, proj_dim, dropout_block, dropout_final, learning_rate, "
            "batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."
        )

        print(f"\n  [Researcher] Analyzing {total} experiments, best AUC={best_str}...")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": PERCH_RESEARCHER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        spec = self._parse_spec(response)
        print(
            f"  [Researcher] Strategy: {spec.get('strategy', '?')} | "
            f"n_blocks={spec.get('n_blocks')} hidden={spec.get('hidden_dim')} "
            f"lr={spec.get('learning_rate')} perch_w={spec.get('perch_weight')}"
        )
        print(f"  [Researcher] Reasoning: {spec.get('reasoning', '')[:120]}")
        return spec

    def _parse_spec(self, response: str) -> dict:
        # Strip deepseek-r1 thinking tokens before searching for JSON
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Try ```json ... ``` block
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if m:
            try:
                return _perch_fill_defaults(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

        # Walk character by character to find the first complete JSON object
        spec = _extract_first_json_object(cleaned)
        if spec is not None:
            return _perch_fill_defaults(spec)

        print("  [Researcher] Warning: could not parse JSON, using safe defaults.")
        print(f"  [Researcher] Raw response (first 400 chars): {repr(cleaned[:400])}")
        return _perch_safe_defaults()


def _perch_safe_defaults() -> dict:
    return {
        "n_blocks":      2,
        "hidden_dim":    1024,
        "proj_dim":      512,
        "dropout_block": 0.3,
        "dropout_final": 0.4,
        "learning_rate": 8e-4,
        "batch_size":    256,
        "optimizer":     "adam",
        "epochs":        25,
        "patience":      5,
        "perch_weight":  0.2,
        "reasoning":     "Fallback defaults — researcher output could not be parsed.",
        "hypothesis":    "Baseline residual head config.",
        "strategy":      "explore",
    }


def _perch_fill_defaults(spec: dict) -> dict:
    defaults = _perch_safe_defaults()
    for k, v in defaults.items():
        spec.setdefault(k, v)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Coder — inner loop, writes TF/Keras head given spec
# ─────────────────────────────────────────────────────────────────────────────

PERCH_CODER_SYSTEM_PROMPT = """You are a Python ML engineer.
You receive a hyperparameter spec and must return ONLY a Python function get_training_config().

Rules:
- Define ONLY ONE function: get_training_config()
- get_training_config() returns a plain Python dict with all the values from the spec
- Do NOT define build_head(), main(), or any other function
- Do NOT import anything
- No top-level executable statements

Example:
```python
def get_training_config():
    return {
        "n_blocks": 2,
        "hidden_dim": 1024,
        "proj_dim": 512,
        "dropout_block": 0.3,
        "dropout_final": 0.4,
        "learning_rate": 0.001,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 25,
        "patience": 5,
        "perch_weight": 0.2,
    }
```"""


def _spec_to_coder_prompt(spec: dict) -> str:
    clean = {k: v for k, v in spec.items() if k not in ("reasoning", "hypothesis", "strategy")}
    return (
        f"Write get_training_config() that returns exactly this configuration:\n"
        f"{json.dumps(clean, indent=2)}\n\n"
        f"Return ONLY the function in a ```python``` code block. Nothing else."
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
    if "get_training_config" not in names:
        issues.append("Missing: get_training_config()")
    return issues


def generate_perch_code(
    coder_llm: LLMClient, spec: dict, temperature: float, max_retries: int = 3
) -> str | None:
    prompt = _spec_to_coder_prompt(spec)
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        print(f"  [Coder] Attempt {attempt}/{max_retries}...")
        response = coder_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": PERCH_CODER_SYSTEM_PROMPT},
                {"role": "user",   "content": current_prompt},
            ],
            temperature=temperature,
        )

        if response.startswith("Error communicating"):
            print(f"  [Coder] LLM error: {response[:150]}")
            break

        code = _extract_code(response)
        if not code:
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        issues = _validate_perch_code(code) if code else ["No code found in response."]
        if not issues:
            print("  [Coder] Code valid.")
            return code

        print(f"  [Coder] Issues: {issues}")
        current_prompt = (
            "Your code had issues:\n" + "\n".join(f"- {i}" for i in issues) +
            f"\n\nOriginal spec:\n{_spec_to_coder_prompt(spec)}\n\n"
            "Fix all issues and return the corrected code block."
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Script harness — wraps Coder's slot code into a runnable training script
# ─────────────────────────────────────────────────────────────────────────────

HARNESS_PREFIX = r"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

# Locate project root
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

# Load pre-built embedding caches
_CACHE_DIR = _PROJECT_ROOT / "logs" / "perch_cache"

def _load_cache(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    return d["X"].astype(np.float32), d["S"].astype(np.float32), d["y"].astype(np.float32)

X_train, S_train, y_train = _load_cache(_CACHE_DIR / "train_emb.npz")
X_val,   S_val,   y_val   = _load_cache(_CACHE_DIR / "val_emb.npz")

EMB_DIM   = X_train.shape[1]   # 1536
N_CLASSES = y_train.shape[1]   # 234

print(f"  Loaded train: X={X_train.shape}  y={y_train.shape}")
print(f"  Loaded val:   X={X_val.shape}    y={y_val.shape}")
""".strip()


HARNESS_SUFFIX = r"""
# Fixed reference architecture — always the same structure, controlled by get_training_config()
# This overrides any build_head() the Coder may have generated, ensuring Kaggle notebook matches.
def build_head(emb_dim, num_classes):
    cfg        = get_training_config()
    n_blocks   = int(cfg.get("n_blocks",      2))
    hidden_dim = int(cfg.get("hidden_dim",    1024))
    proj_dim   = int(cfg.get("proj_dim",      512))
    drop_block = float(cfg.get("dropout_block", 0.3))
    drop_final = float(cfg.get("dropout_final", 0.4))

    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x   = tf.keras.layers.BatchNormalization()(inp)
    x   = tf.keras.layers.Dense(hidden_dim)(x)
    x   = tf.keras.layers.LayerNormalization()(x)
    for _ in range(n_blocks):
        h = tf.keras.layers.Dense(hidden_dim)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("gelu")(h)
        h = tf.keras.layers.Dropout(drop_block)(h)
        h = tf.keras.layers.Dense(hidden_dim)(h)
        x = tf.keras.layers.Add()([x, h])
        x = tf.keras.layers.LayerNormalization()(x)
    x   = tf.keras.layers.Dense(proj_dim, activation="gelu")(x)
    x   = tf.keras.layers.Dropout(drop_final)(x)
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()

    lr           = float(cfg.get("learning_rate", 8e-4))
    batch_size   = int(cfg.get("batch_size", 256))
    epochs       = int(cfg.get("epochs", 50))
    patience     = int(cfg.get("patience", 7))
    perch_weight = float(cfg.get("perch_weight", 0.2))
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

    # Build head (Coder-generated, uncompiled)
    head = build_head(EMB_DIM, N_CLASSES)

    # Compile
    if opt_name == "sgd_momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
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

    # Predict on soundscape validation
    head_probs  = head.predict(X_val, batch_size=batch_size, verbose=0)
    perch_probs = 1.0 / (1.0 + np.exp(-S_val))   # sigmoid of Perch logit scores

    # Blend: perch_weight controls how much to trust Perch's own classifier
    y_pred = perch_weight * perch_probs + (1.0 - perch_weight) * head_probs

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


def _build_script(slot_code: str) -> str:
    return HARNESS_PREFIX + "\n\n" + slot_code + "\n\n" + HARNESS_SUFFIX


# ─────────────────────────────────────────────────────────────────────────────
# Final retrain on full data (train + val combined)
# ─────────────────────────────────────────────────────────────────────────────

def _build_final_retrain_script(best_spec: dict, mem_dir: Path, cache_dir: Path) -> str:
    spec_json = json.dumps(best_spec)
    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_CACHE_DIR = Path(r"{cache_dir}")
_MEM_DIR   = Path(r"{mem_dir}")

def _load(p):
    d = np.load(str(p), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)

X_tr, y_tr = _load(_CACHE_DIR / "train_emb.npz")
X_vl, y_vl = _load(_CACHE_DIR / "val_emb.npz")

X_full = np.concatenate([X_tr, X_vl], axis=0)
y_full = np.concatenate([y_tr, y_vl], axis=0)

EMB_DIM   = X_full.shape[1]
N_CLASSES = y_full.shape[1]
print(f"  Final retrain: X={{X_full.shape}}  y={{y_full.shape}}")

cfg        = {spec_json}
n_blocks   = int(cfg.get("n_blocks",      2))
hidden_dim = int(cfg.get("hidden_dim",    1024))
proj_dim   = int(cfg.get("proj_dim",      512))
drop_block = float(cfg.get("dropout_block", 0.3))
drop_final = float(cfg.get("dropout_final", 0.4))
lr         = float(cfg.get("learning_rate", 8e-4))
batch_size = int(cfg.get("batch_size",    256))
epochs     = int(cfg.get("epochs",         50))
opt_name   = str(cfg.get("optimizer",   "adam"))

inp = tf.keras.layers.Input(shape=(EMB_DIM,))
x   = tf.keras.layers.BatchNormalization()(inp)
x   = tf.keras.layers.Dense(hidden_dim)(x)
x   = tf.keras.layers.LayerNormalization()(x)
for _ in range(n_blocks):
    h = tf.keras.layers.Dense(hidden_dim)(x)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.keras.layers.Activation("gelu")(h)
    h = tf.keras.layers.Dropout(drop_block)(h)
    h = tf.keras.layers.Dense(hidden_dim)(h)
    x = tf.keras.layers.Add()([x, h])
    x = tf.keras.layers.LayerNormalization()(x)
x   = tf.keras.layers.Dense(proj_dim, activation="gelu")(x)
x   = tf.keras.layers.Dropout(drop_final)(x)
out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
head = tf.keras.Model(inp, out)

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
else:
    opt = tf.keras.optimizers.Adam(lr)
head.compile(optimizer=opt, loss=weighted_bce)

tf.keras.utils.set_random_seed(42)
head.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=1)

head.save(str(_MEM_DIR / "final_head.keras"))
head.save_weights(str(_MEM_DIR / "final_head.weights.h5"))
print("FINAL_RETRAIN_DONE")
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    logs_dir  = ROOT / "logs"
    code_dir  = logs_dir / "perch_agent_codes"
    cache_dir = logs_dir / "perch_cache"
    mem_dir   = logs_dir / "perch_memory"
    for d in [logs_dir, code_dir, cache_dir, mem_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Shortcut: skip search and only run final retrain on saved best spec
    if config.get("final_retrain_only", False):
        memory = ExperimentMemory(mem_dir)
        best_runs = memory.best_runs(1)
        if best_runs:
            best_spec = best_runs[0]["spec"]
            best_auc  = best_runs[0]["macro_roc_auc"]
            print(f"\n{'='*60}")
            print(f"  FINAL RETRAIN ONLY — best spec (val AUC={best_auc:.5f})")
            print(f"{'='*60}")
            py_exe   = config.get("execution", {}).get("python_executable", "python3")
            timeout  = config.get("execution", {}).get("timeout_seconds", 1800)
            executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
            final_script      = _build_final_retrain_script(best_spec, mem_dir, cache_dir)
            final_script_path = code_dir / "final_retrain.py"
            final_script_path.write_text(final_script, encoding="utf-8")
            result = executor.run_file(final_script_path)
            if result.success and "FINAL_RETRAIN_DONE" in (result.stdout or ""):
                print(f"  Final head saved → {mem_dir / 'final_head.weights.h5'}")
            else:
                print(f"  Final retrain failed: {(result.stderr or '')[-400:]}")
        else:
            print("  No memory found — cannot run final retrain only.")
        return

    perch_cfg      = config.get("perch", {})
    onnx_slug      = perch_cfg.get("onnx_dataset",       "rishikeshjani/perch-onnx-for-birdclef-2026")
    labels_slug    = perch_cfg.get("perch_labels_model",  "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu")
    max_samples    = perch_cfg.get("max_train_samples",   None)
    embed_bs       = perch_cfg.get("embed_batch_size",    16)
    force_rebuild  = perch_cfg.get("force_rebuild_cache", False)
    max_iterations = config.get("max_iterations", 10)

    researcher_model = config.get("researcher", {}).get("model",       "deepseek-r1:8b")
    coder_model      = config.get("llm",        {}).get("model",       "deepseek-r1:8b")
    provider         = config.get("llm",        {}).get("provider",    "ollama")
    researcher_temp  = config.get("researcher", {}).get("temperature", 0.6)
    coder_temp       = config.get("llm",        {}).get("temperature", 0.2)
    py_exe           = config.get("execution",  {}).get("python_executable", "python3")
    timeout          = config.get("execution",  {}).get("timeout_seconds",   1800)

    print("=" * 60)
    print("  BirdCLEF Perch Agent — Researcher / Coder Architecture")
    print("=" * 60)

    # ── Step 1: Install deps ──────────────────────────────────────────────
    _ensure_deps()

    # ── Step 2: Download ONNX model + Perch labels ────────────────────────
    onnx_path   = _find_or_download_onnx(onnx_slug)
    labels_path = _find_or_download_perch_labels(labels_slug)

    # ── Step 3: Load ONNX session ─────────────────────────────────────────
    sess, inp_name, emb_idx, logit_idx = _load_onnx_session(onnx_path)

    # ── Step 4: Load data ─────────────────────────────────────────────────
    import pandas as pd
    os.environ.setdefault("BIRDCLEF_DATA_DIR", str(ROOT / "data"))
    from data_io import (
        load_core_tables, resolve_birdclef_paths,
        species_columns_from_sample_submission, validate_required_files,
    )
    paths   = resolve_birdclef_paths()
    missing = validate_required_files(paths)
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")

    tables      = load_core_tables(paths)
    train_df    = tables["train"]
    sample_sub  = tables["sample_submission"]
    _tax = tables.get("taxonomy")
    taxonomy_df = _tax if _tax is not None else pd.read_csv(paths.taxonomy_csv)

    species_cols   = species_columns_from_sample_submission(sample_sub)
    species_to_idx = {s: i for i, s in enumerate(species_cols)}
    n_species      = len(species_cols)

    # ── Step 5: Build logit mapping ───────────────────────────────────────
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL = _build_logit_mapping(
        labels_path, taxonomy_df, species_cols
    )

    # ── Save species mapping for Kaggle notebook ──────────────────────────
    import numpy as _np_map
    bc_indices_full = _np_map.full(n_species, int(NO_LABEL), dtype=_np_map.int32)
    bc_indices_full[MAPPED_POS] = MAPPED_BC_IDX
    _np_map.save(str(mem_dir / "bc_indices.npy"), bc_indices_full)
    with open(mem_dir / "proxy_map.json", "w") as _f:
        json.dump({str(k): v for k, v in proxy_map.items()}, _f)
    with open(mem_dir / "species_cols.json", "w") as _f:
        json.dump(species_cols, _f)
    with open(mem_dir / "mapping_meta.json", "w") as _f:
        json.dump({"NO_LABEL": int(NO_LABEL), "n_species": n_species}, _f)

    # ── Step 6: Build embedding caches (once) ─────────────────────────────
    train_cache = cache_dir / "train_emb.npz"
    val_cache   = cache_dir / "val_emb.npz"

    if force_rebuild or not train_cache.exists():
        print("\n  [Setup] Building training embedding cache (runs once, ~30-60 min)...")
        _build_train_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            train_df, species_to_idx, paths.train_audio_dir, n_species,
            train_cache, max_samples, embed_bs,
        )
    else:
        import numpy as np
        d = np.load(str(train_cache))
        print(f"  [Cache] Train cache loaded: X={d['X'].shape}  y={d['y'].shape}")

    if force_rebuild or not val_cache.exists():
        print("\n  [Setup] Building validation embedding cache...")
        ok = _build_val_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            paths.train_soundscapes_dir, paths.train_soundscapes_labels_csv,
            species_to_idx, n_species, val_cache, embed_bs,
        )
        if not ok:
            print("  [Cache] Soundscape labels not found — using focal-clip fallback for val.")
            _build_focal_val_fallback(train_cache, val_cache)
    else:
        import numpy as np
        d = np.load(str(val_cache))
        print(f"  [Cache] Val cache loaded: X={d['X'].shape}  y={d['y'].shape}")

    # ── Step 7: Set up agent components ──────────────────────────────────
    researcher_llm = LLMClient(provider=provider, model=researcher_model)
    coder_llm      = LLMClient(provider=provider, model=coder_model)
    memory         = ExperimentMemory(mem_dir)
    researcher     = PerchResearcher(researcher_llm, memory, temperature=researcher_temp)
    executor       = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator      = Evaluator(row_id_column_name="row_id")

    print(f"\n  Researcher model : {researcher_model}")
    print(f"  Coder model      : {coder_model}")
    print(f"  Max iterations   : {max_iterations}")
    prior = memory.total()
    if prior:
        best     = memory.best_runs(1)
        best_auc = best[0]["macro_roc_auc"] if best else None
        best_str = f"{best_auc:.5f}" if best_auc is not None else "none"
        print(f"  Memory           : {prior} prior runs | best AUC={best_str}")
    else:
        print("  Memory           : fresh start")
    print("=" * 60)

    # ── Step 8: Agent loop ────────────────────────────────────────────────
    y_true_path        = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path        = Path(tempfile.gettempdir()) / "_y_pred.npy"
    trained_head_path  = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path     = mem_dir / "best_head.keras"
    _prior_best        = memory.best_runs(1)
    best_auc_ever      = _prior_best[0]["macro_roc_auc"] if _prior_best else -1.0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'─'*60}")

        spec      = researcher.next_experiment()
        slot_code = generate_perch_code(coder_llm, spec, coder_temp)

        if slot_code is None:
            print("  [Coder] Failed to generate valid code. Logging failure.")
            memory.log(spec=spec, metrics=None, code="")
            continue

        script      = _build_script(slot_code)
        script_path = code_dir / f"iter_{iteration:03d}.py"
        script_path.write_text(script, encoding="utf-8")

        print(f"  [Executor] Running {script_path.name} ...")
        result = executor.run_file(script_path)

        metrics = None
        if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
            if y_true_path.exists() and y_pred_path.exists():
                summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
                metrics = summary.metrics

        auc    = metrics.get("macro_roc_auc") if metrics else None
        status = f"AUC={auc:.5f}" if auc is not None else "FAILED"
        print(f"  [Result] {status}")
        if not result.success and result.stderr:
            # TF logs to stderr too — show the tail where the real Python error is
            print(f"  [Error]  {result.stderr[-600:]}")

        memory.log(spec=spec, metrics=metrics, code=slot_code)

        # Promote to best model if this run beat the previous best
        if auc is not None and auc > best_auc_ever:
            best_auc_ever = auc
            if trained_head_path.exists():
                import shutil
                shutil.copy2(str(trained_head_path), str(best_head_path))
                # Also copy weights-only file for Kaggle (no Keras version dependency)
                _weights_src = Path(tempfile.gettempdir()) / "_trained_head.weights.h5"
                if _weights_src.exists():
                    shutil.copy2(str(_weights_src), str(mem_dir / "best_head.weights.h5"))
                with open(mem_dir / "best_model_info.json", "w") as _f:
                    json.dump({"auc": auc, "iteration": iteration, "spec": spec}, _f, indent=2)
                # Save val preds for meta-agent ensemble phase
                _y_pred_tmp = Path(tempfile.gettempdir()) / "_y_pred.npy"
                _y_true_tmp = Path(tempfile.gettempdir()) / "_y_true.npy"
                if _y_pred_tmp.exists():
                    shutil.copy2(str(_y_pred_tmp), str(mem_dir / "best_val_preds.npy"))
                if _y_true_tmp.exists():
                    shutil.copy2(str(_y_true_tmp), str(mem_dir / "y_val.npy"))
                print(f"  [Best] NEW BEST AUC={auc:.5f} — head saved to {best_head_path.name}")

        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] AUC={best[0]['macro_roc_auc']:.5f}")

    print(f"\n{'='*60}")
    print("  DONE")
    best = memory.best_runs(3)
    for i, r in enumerate(best, 1):
        print(f"  #{i} AUC={r['macro_roc_auc']:.5f} | {r['reasoning'][:80]}")
    print(f"{'='*60}")

    # ── Final retrain on full data (train + val) with best spec ──────────────
    best_runs = memory.best_runs(1)
    if best_runs and not config.get("skip_final_training", False):
        best_spec = best_runs[0]["spec"]
        best_auc  = best_runs[0]["macro_roc_auc"]
        print(f"\n{'='*60}")
        print(f"  FINAL RETRAIN — full data (val AUC={best_auc:.5f})")
        print(f"{'='*60}")
        final_script      = _build_final_retrain_script(best_spec, mem_dir, cache_dir)
        final_script_path = code_dir / "final_retrain.py"
        final_script_path.write_text(final_script, encoding="utf-8")
        result = executor.run_file(final_script_path)
        if result.success and "FINAL_RETRAIN_DONE" in (result.stdout or ""):
            print(f"  Final head saved → {mem_dir / 'final_head.weights.h5'}")
        else:
            print(f"  Final retrain failed: {(result.stderr or '')[-400:]}")
    else:
        print("  No successful runs — skipping final retrain.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "agent_config.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config)


if __name__ == "__main__":
    main()
