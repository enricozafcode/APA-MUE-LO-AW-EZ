"""
Step 2: Final Training using precomputed cache (or falls back to live loading).
Run after precompute_spectrograms.py:

    .venv\Scripts\python.exe scripts\run_final_training.py
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from agent import assemble_script, _append_eval_wrapper
from code_executor import CodeExecutor
from evaluator import Evaluator

EPOCHS     = 15
VAL_SPLIT  = 0.1
MODEL_SAVE = str(root / "submission" / "model.keras")
CACHE_X    = root / "data" / "mel_cache" / "X_all.npy"
CACHE_Y    = root / "data" / "mel_cache" / "y_all.npy"

winner_path = root / "logs" / "winner_slot_code.py"
slot_code = winner_path.read_text(encoding="utf-8")
slot_code = slot_code.replace('"max_samples": 2000', '"max_samples": None')
slot_code = slot_code.replace('"epochs": 3',         f'"epochs": {EPOCHS}')
slot_code = slot_code.replace('"val_split": 0.2',    f'"val_split": {VAL_SPLIT}')

# If cache exists: inject fast numpy loading instead of audio loading
use_cache = CACHE_X.exists() and CACHE_Y.exists()

cache_injection = f"""
import os as _os
_os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
_os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")
_os.environ.setdefault("OMP_NUM_THREADS", "8")
"""
slot_code = cache_injection + slot_code

print("=" * 60)
print("  FINAL TRAINING")
print("=" * 60)
print(f"  Epochs     : {EPOCHS}")
print(f"  Max samples: ALL")
print(f"  Cache      : {'YES (fast!)' if use_cache else 'NO (live audio loading)'}")
print("=" * 60)

if use_cache:
    # Load cache and inject directly — skip harness audio loading entirely
    print(f"\nLoading precomputed spectrograms from cache...")
    X = np.load(CACHE_X)
    y = np.load(CACHE_Y)
    print(f"  X={X.shape}, y={y.shape}")

    # Write a standalone training script using the cached arrays
    script = f"""
import os, sys, json, numpy as np
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
from pathlib import Path
root = Path(r"{root}")
sys.path.insert(0, str(root / "src"))

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

X = np.load(r"{CACHE_X}")
y = np.load(r"{CACHE_Y}")
print(f"Data loaded: X={{X.shape}}, y={{y.shape}}")

# Split
split = int(len(X) * {1 - VAL_SPLIT})
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]
print(f"Train={{len(X_train)}}, Val={{len(X_val)}}")

# Build model
"""
    # Add the build_model function from slot_code
    script += slot_code + f"""

input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = build_model(input_shape, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy")
model.summary()

# Train
history = model.fit(X_train, y_train, epochs={EPOCHS}, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)

# Save
_mp = Path(r"{MODEL_SAVE}")
_mp.parent.mkdir(parents=True, exist_ok=True)
model.save(_mp)
print(f"MODEL_SAVED: {{_mp}}")

# Eval artifacts
import numpy as _np
_np.save(r"{root / 'logs' / 'eval_artifacts' / 'y_true_final_v2.npy'}", y_val)
_np.save(r"{root / 'logs' / 'eval_artifacts' / 'y_pred_final_v2.npy'}", model.predict(X_val, verbose=0))
print("EVAL_ARTIFACTS_SAVED")
"""
    script_path = root / "logs" / "final_run_v2.py"
    script_path.write_text(script, encoding="utf-8")
else:
    # Fall back to agent harness (live audio loading)
    slot_code_full = slot_code
    script = assemble_script(slot_code_full, is_final=True, model_save_path=MODEL_SAVE)
    eval_dir = root / "logs" / "eval_artifacts"
    eval_dir.mkdir(exist_ok=True)
    script = _append_eval_wrapper(script, "final_v2", eval_dir)
    script_path = root / "logs" / "final_run_v2.py"
    script_path.write_text(script, encoding="utf-8")

print(f"  Script -> {script_path.name}\n")

py_exe = str(root / ".venv" / "Scripts" / "python.exe")
executor = CodeExecutor(python_executable=py_exe, timeout_seconds=None)

print("Starting training...\n")
result = executor.run_file(script_path)

if result.success:
    print("\nSUCCESS!")
    if Path(MODEL_SAVE).exists():
        mb = Path(MODEL_SAVE).stat().st_size / (1024 * 1024)
        print(f"  model.keras saved ({mb:.1f} MB)")
    yt = root / "logs" / "eval_artifacts" / "y_true_final_v2.npy"
    yp = root / "logs" / "eval_artifacts" / "y_pred_final_v2.npy"
    if yt.exists() and yp.exists():
        evaluator = Evaluator(row_id_column_name="row_id")
        ev = evaluator.evaluate_from_files(yt, yp)
        m = ev.metrics
        if m.get("status") == "success":
            print(f"  Final AUC  : {m['macro_roc_auc']:.4f}")
            print(f"  Species    : {m['num_scored_columns']}/234")
            print(f"  Samples    : {m['num_samples']}")
else:
    print("\nFAILED!")
    print(result.stderr[-2000:])
