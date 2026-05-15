"""
agent_loop.py — Researcher / Coder architecture.

OUTER LOOP: Researcher (reasoning model, e.g. deepseek-r1)
  - Reads full experiment history
  - Decides what to try next → outputs a compact spec (JSON)

INNER LOOP: Coder (code model, e.g. qwen2.5-coder)
  - Gets only the spec — no history, no memory
  - Writes get_training_config() + build_model() → executes → measures AUC

Memory: every result is logged to logs/experiment_memory.jsonl (append-only)

Run:
    python src/agent_loop.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from memory import ExperimentMemory
from researcher import Researcher

# ─────────────────────────────────────────────────────────────────────────────
# Coder system prompt — short and focused, no history
# ─────────────────────────────────────────────────────────────────────────────

CODER_SYSTEM_PROMPT = """You are a Python ML engineer.
You receive a hyperparameter specification and write working TensorFlow/Keras training code.
Return ONLY one ```python``` code block, nothing else.

Rules:
- Define exactly: get_training_config() and build_model(input_shape, num_classes)
- Optional: build_features(audio_path, sample_rate, clip_seconds, n_mels, n_frames)
- Do NOT define main()
- Final layer: Dense(num_classes, activation='sigmoid')
- Compile with binary_crossentropy
- No top-level executable statements (no print/fit/etc. outside functions)"""


def _spec_to_coder_prompt(spec: dict) -> str:
    """Convert researcher spec into a short, clean prompt for the Coder."""
    clean_spec = {k: v for k, v in spec.items()
                  if k not in ("reasoning", "hypothesis", "strategy")}
    return (
        f"Implement this BirdCLEF model configuration:\n"
        f"{json.dumps(clean_spec, indent=2)}\n\n"
        f"Write get_training_config() returning these values and "
        f"build_model(input_shape, num_classes) implementing the architecture. "
        f"Use the depth, filters_base, dropout, batch_norm, and residuals values."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Code generation + validation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code(response: str) -> str | None:
    # Try ```python ... ``` block first
    match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try plain ``` ... ``` block
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Skip if first line looks like a language tag
        first = candidate.splitlines()[0].strip().lower() if candidate else ""
        if first in ("python", "py", ""):
            candidate = "\n".join(candidate.splitlines()[1:]).strip() if first else candidate
        return candidate if candidate else None
    return None


def _validate_code(code: str) -> list[str]:
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "get_training_config" not in names:
        issues.append("Missing: get_training_config()")
    if "build_model" not in names:
        issues.append("Missing: build_model(input_shape, num_classes)")
    return issues


def generate_code(coder_llm: LLMClient, spec: dict, temperature: float, max_retries: int = 3) -> str | None:
    """Inner loop: Coder gets spec, returns validated Python code. No memory involved."""
    prompt = _spec_to_coder_prompt(spec)
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        print(f"  [Coder] Attempt {attempt}/{max_retries}...")
        response = coder_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": current_prompt},
            ],
            temperature=temperature,
        )
        code = _extract_code(response)
        if not code:
            # Try stripping markdown manually
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        # Detect LLM-level errors before attempting to parse as code
        if response.startswith("Error communicating"):
            print(f"  [Coder] LLM error: {response[:150]}")
            break

        issues = _validate_code(code) if code else ["No code found in response."]
        if not issues:
            print(f"  [Coder] Code valid.")
            return code

        print(f"  [Coder] Issues: {issues}")
        print(f"  [Coder] Response preview: {repr(response[:200])}")
        current_prompt = (
            f"Your code had issues:\n" + "\n".join(f"- {i}" for i in issues) +
            f"\n\nOriginal spec:\n{json.dumps(_spec_to_coder_prompt(spec))}\n\n"
            "Fix all issues and return the corrected code block."
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

HARNESS_PREFIX = """
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

_PROJECT_ROOT = None
for _cand in list(Path(__file__).resolve().parents):
    if (_cand / "src").exists() and (_cand / "configs").exists():
        _PROJECT_ROOT = _cand
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("BIRDCLEF_DATA_DIR", str(_PROJECT_ROOT / "data"))

from src.data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission, validate_required_files

def _load_mel(audio_path, sample_rate, clip_seconds, n_mels, n_frames):
    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    mel = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_r = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_r.astype(np.float32)
""".strip()

HARNESS_SUFFIX = """
def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()
    max_samples = cfg.get("max_samples", 1500)
    epochs      = cfg.get("epochs", 3)
    batch_size  = cfg.get("batch_size", 32)
    lr          = cfg.get("learning_rate", 1e-3)
    n_mels      = cfg.get("n_mels", 64)
    n_frames    = cfg.get("n_frames", 128)
    sr          = cfg.get("sample_rate", 32000)
    clip_sec    = cfg.get("clip_seconds", 5.0)
    val_split   = cfg.get("val_split", 0.2)

    paths = resolve_birdclef_paths()
    missing = validate_required_files(paths)
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    tables    = load_core_tables(paths)
    train_df  = tables["train"]
    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)
    sp2i = {s: i for i, s in enumerate(species_cols)}
    num_classes = len(species_cols)

    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename"  if "filename"  in train_df.columns else "filepath"
    mel_fn = _load_mel

    audio_dir = paths["train_audio"]
    rows = train_df[[lcol, fcol]].dropna().values.tolist()
    if max_samples and len(rows) > max_samples:
        import random; random.seed(42); random.shuffle(rows)
        rows = rows[:max_samples]

    X, y = [], []
    for label, fname in rows:
        if label not in sp2i:
            continue
        p = audio_dir / fname
        if not p.exists():
            continue
        try:
            mel = mel_fn(p, sr, clip_sec, n_mels, n_frames)
            vec = np.zeros(num_classes, dtype=np.float32)
            vec[sp2i[label]] = 1.0
            X.append(mel); y.append(vec)
        except Exception:
            continue
    if not X:
        raise RuntimeError("No samples loaded.")

    X = np.stack(X); y = np.stack(y)
    split = max(1, int(len(X) * val_split))
    X_val, y_val = X[:split], y[:split]
    X_tr,  y_tr  = X[split:],  y[split:]

    model = build_model(X_tr.shape[1:], num_classes)
    opt_name = cfg.get("optimizer", "adam")
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9) if opt_name == "sgd_momentum" else tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    has_val = len(X_val) > 0
    cb = []
    ckpt_path = "/tmp/_best_model.keras"
    if has_val:
        cb.append(tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0))

    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val) if has_val else None, callbacks=cb, verbose=1)

    if has_val and Path(ckpt_path).exists():
        model = tf.keras.models.load_model(ckpt_path)

    # Evaluate
    y_pred = model.predict(X_val if has_val else X_tr, batch_size=batch_size, verbose=0)
    y_true = y_val if has_val else y_tr

    import numpy as _np
    _np.save("/tmp/_y_true.npy", y_true)
    _np.save("/tmp/_y_pred.npy", y_pred)
    print("EVAL_ARTIFACTS_SAVED")

if __name__ == "__main__":
    main()
"""


def _build_script(slot_code: str) -> str:
    return HARNESS_PREFIX + "\n\n" + slot_code + "\n\n" + HARNESS_SUFFIX


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    root = ROOT
    logs_dir = root / "logs"
    code_dir = logs_dir / "agent_loop_codes"
    eval_dir = logs_dir / "agent_loop_eval"
    for d in [logs_dir, code_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Two separate LLM clients
    researcher_model = config.get("researcher", {}).get("model", "deepseek-r1:8b")
    coder_model      = config.get("llm", {}).get("model", "qwen2.5-coder:7b")
    provider         = config.get("llm", {}).get("provider", "ollama")
    researcher_temp  = config.get("researcher", {}).get("temperature", 0.6)
    coder_temp       = config.get("llm", {}).get("temperature", 0.2)
    max_iterations   = config.get("max_iterations", 10)
    py_exe           = config.get("execution", {}).get("python_executable", "python3")
    timeout          = config.get("execution", {}).get("timeout_seconds", 1800)

    researcher_llm = LLMClient(provider=provider, model=researcher_model)
    coder_llm      = LLMClient(provider=provider, model=coder_model)

    memory    = ExperimentMemory(logs_dir)
    researcher = Researcher(researcher_llm, memory, temperature=researcher_temp)
    executor  = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator = Evaluator(row_id_column_name="row_id")

    print("=" * 60)
    print("  BirdCLEF Agent — Researcher / Coder Architecture")
    print("=" * 60)
    print(f"  Researcher model : {researcher_model}")
    print(f"  Coder model      : {coder_model}")
    print(f"  Max iterations   : {max_iterations}")
    prior = memory.total()
    if prior:
        best = memory.best_runs(1)
        best_auc = best[0]["macro_roc_auc"] if best else None
        best_auc_str = f"{best_auc:.5f}" if best_auc is not None else "none"
        print(f"  Memory           : {prior} prior runs | best AUC={best_auc_str}")
    else:
        print("  Memory           : fresh start")
    print("=" * 60)

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'─'*60}")

        # ── OUTER LOOP: Researcher decides what to try ──
        spec = researcher.next_experiment()

        # ── INNER LOOP: Coder writes the code ──
        slot_code = generate_code(coder_llm, spec, coder_temp)
        if slot_code is None:
            print("  [Coder] Failed to generate valid code. Logging failure.")
            memory.log(spec=spec, metrics=None, code="")
            continue

        # Build and save full script
        script = _build_script(slot_code)
        script_path = code_dir / f"iter_{iteration:03d}.py"
        script_path.write_text(script, encoding="utf-8")

        # ── Execute ──
        print(f"  [Executor] Running {script_path.name} ...")
        result = executor.run_file(script_path)

        # ── Evaluate ──
        metrics = None
        if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
            y_true_path = Path("/tmp/_y_true.npy")
            y_pred_path = Path("/tmp/_y_pred.npy")
            if y_true_path.exists() and y_pred_path.exists():
                summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
                metrics = summary.metrics

        auc = metrics.get("macro_roc_auc") if metrics else None
        status = f"AUC={auc:.5f}" if auc else "FAILED"
        print(f"  [Result] {status}")
        if not result.success and result.stderr:
            print(f"  [Error]  {result.stderr[:300]}")

        # ── Log to memory ──
        memory.log(spec=spec, metrics=metrics, code=slot_code)

        # ── Print running best ──
        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] AUC={best[0]['macro_roc_auc']:.5f}")

    print(f"\n{'='*60}")
    print("  DONE")
    best = memory.best_runs(3)
    for i, r in enumerate(best, 1):
        print(f"  #{i} AUC={r['macro_roc_auc']:.5f} | {r['reasoning'][:80]}")
    print(f"{'='*60}")


def main() -> None:
    config_path = ROOT / "configs" / "agent_config.json"
    config = json.loads(config_path.read_text())
    run(config)


if __name__ == "__main__":
    main()
