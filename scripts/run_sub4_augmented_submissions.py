#!/usr/bin/env python3
"""
Run final training from archived sub_4 model code for the v1.4 setup only.

Strategy:
  - online augmentation B (time/freq mask + mild noise + small time shift)
  - no mixup
  - include secondary_labels as additional positive targets (multi-hot labels)

Outputs:
  submission/submission_01/
    - model.keras
    - kaggle_inference.ipynb
    - best_model_code.py
    - best_params.json
    - train.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_training_config(slot_code: str) -> dict:
    ns: dict = {}
    exec(slot_code, ns)  # trusted local model code from submission archive
    fn = ns.get("get_training_config")
    if not callable(fn):
        return {}
    cfg = fn()
    return cfg if isinstance(cfg, dict) else {}


def _base_build_features_code() -> str:
    # Keep feature extraction deterministic; augmentation is applied online per epoch.
    return """
def build_features(audio_path, sample_rate, clip_seconds, n_mels, n_frames):
    import numpy as np
    import librosa
    import tensorflow as tf

    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy().astype(np.float32)
""".strip()


def _inject_online_augmentation_into_script(script: str) -> str:
    anchor = "history = model.fit(\n        X_train, y_train,"
    if anchor not in script:
        return script

    aug_block = """
    # Online augmentation: re-sampled every batch, every epoch.
    aug_prob = float(cfg.get("aug_probability", 0.6))
    noise_std = float(cfg.get("aug_noise_std", 0.015))
    time_shift_frac = float(cfg.get("aug_time_shift_max_fraction", 0.10))
    time_mask_frac = float(cfg.get("aug_time_mask_max_fraction", 0.08))
    freq_mask_frac = float(cfg.get("aug_freq_mask_max_fraction", 0.08))
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))

    class _AugSequence(tf.keras.utils.Sequence):
        def __init__(self, X, y, bs, seed=42):
            self.X = np.asarray(X, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32)
            self.bs = int(max(1, bs))
            self.rng = np.random.default_rng(seed)
            self.idx = np.arange(len(self.X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.bs))

        def on_epoch_end(self):
            self.rng.shuffle(self.idx)

        def _augment_batch(self, xb):
            xb = xb.copy()
            B, H, W, _C = xb.shape
            for i in range(B):
                if self.rng.random() < aug_prob and noise_std > 0.0:
                    xb[i, :, :, 0] += noise_std * self.rng.standard_normal((H, W), dtype=np.float32)
                if self.rng.random() < aug_prob and time_shift_frac > 0.0:
                    max_shift = int(max(1, round(W * time_shift_frac)))
                    shift = self.rng.integers(-max_shift, max_shift + 1)
                    xb[i, :, :, 0] = np.roll(xb[i, :, :, 0], shift, axis=1)
                if self.rng.random() < aug_prob and time_mask_frac > 0.0:
                    t = int(self.rng.integers(0, max(1, int(W * time_mask_frac)) + 1))
                    if t > 0:
                        t0 = int(self.rng.integers(0, max(1, W - t + 1)))
                        xb[i, :, t0:t0 + t, 0] = 0.0
                if self.rng.random() < aug_prob and freq_mask_frac > 0.0:
                    f = int(self.rng.integers(0, max(1, int(H * freq_mask_frac)) + 1))
                    if f > 0:
                        f0 = int(self.rng.integers(0, max(1, H - f + 1)))
                        xb[i, f0:f0 + f, :, 0] = 0.0
            return xb

        def __getitem__(self, i):
            sl = self.idx[i * self.bs:(i + 1) * self.bs]
            xb = self._augment_batch(self.X[sl])
            yb = self.y[sl].copy()
            if mixup_alpha > 0.0 and len(xb) > 1:
                perm = self.rng.permutation(len(xb))
                lam = self.rng.beta(mixup_alpha, mixup_alpha, size=(len(xb), 1, 1, 1)).astype(np.float32)
                xb2 = xb[perm]
                yb2 = yb[perm]
                xb = lam * xb + (1.0 - lam) * xb2
                yb = lam.reshape(len(xb), 1) * yb + (1.0 - lam.reshape(len(xb), 1)) * yb2
            return xb, yb

    train_seq = _AugSequence(X_train, y_train, batch_size, seed=42)
    print(
        f"ONLINE_AUG: prob={aug_prob} noise_std={noise_std} time_shift_frac={time_shift_frac} "
        f"time_mask_frac={time_mask_frac} freq_mask_frac={freq_mask_frac} mixup_alpha={mixup_alpha}"
    )
"""
    replacement = aug_block + """
    history = model.fit(
        train_seq,"""
    return script.replace(anchor, replacement, 1)


def _inject_secondary_labels_into_script(script: str) -> str:
    anchor = "yv[sp2i[label]] = 1.0"
    if anchor not in script:
        return script
    replacement = """yv[sp2i[label]] = 1.0
        # Add co-occurring species as additional positives when available.
        sec_label_weight = float(cfg.get("secondary_label_weight", 1.0))
        try:
            import ast as _ast
            sec = getattr(row, "secondary_labels", "[]")
            sec_list = _ast.literal_eval(str(sec)) if isinstance(sec, str) else []
            for sl in sec_list:
                sl = str(sl).strip()
                if sl and sl in sp2i:
                    yv[sp2i[sl]] = max(yv[sp2i[sl]], sec_label_weight)
        except Exception:
            pass"""
    return script.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sub_4 model with v1.4 augmentation + secondary labels.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--source-slot",
        type=Path,
        default=None,
        help="Path to source best_model_code.py (default: submission_archive/sub_4_29.04_v1.3/best_model_code.py).",
    )
    args = parser.parse_args()

    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.agent import (
        CodeExecutor,
        Evaluator,
        _append_eval_wrapper,
        _final_config_override_block,
        _final_execution_timeout_seconds,
        _generate_kaggle_notebook,
        assemble_script,
    )

    source_slot = (
        args.source_slot.resolve()
        if args.source_slot
        else (root / "submission_archive" / "sub_4_29.04_v1.3" / "best_model_code.py")
    )
    if not source_slot.exists():
        raise FileNotFoundError(f"Source slot code not found: {source_slot}")

    config_path = root / "configs" / "agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    search_cfg = config.get("search", {})
    final_cfg = search_cfg.get("final", {})

    slot_base = source_slot.read_text(encoding="utf-8")
    base_train_cfg = _extract_training_config(slot_base)
    params = {
        "n_mels": int(base_train_cfg.get("n_mels", 64)),
        "n_frames": int(base_train_cfg.get("n_frames", 128)),
    }

    logs_dir = root / "logs"
    eval_dir = logs_dir / "eval_artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=_final_execution_timeout_seconds(config),
    )
    evaluator = Evaluator(row_id_column_name="row_id")

    variants = [
        ("submission_01_sec_label_1_0", 1.0, "full secondary labels"),
        ("submission_02_sec_label_weighted_0_5", 0.5, "weighted secondary labels"),
    ]

    for folder_name, secondary_weight, variant_label in variants:
        mixup_alpha = 0.0
        out_dir = root / "submission" / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.keras"

        override_cfg = {
            "max_samples": final_cfg.get("max_samples"),
            "sample_rate": 32000,
            "clip_seconds": 5.0,
            "n_mels": params["n_mels"],
            "n_frames": params["n_frames"],
            "epochs": int(final_cfg.get("epochs", base_train_cfg.get("epochs", 15))),
            "batch_size": int(base_train_cfg.get("batch_size", 32)),
            "learning_rate": float(base_train_cfg.get("learning_rate", 1e-3)),
            "optimizer": base_train_cfg.get("optimizer", "adam"),
            "val_split": 0.0,
            "weight_decay": float(base_train_cfg.get("weight_decay", 0.0)),
            "classifier_hidden_units": int(base_train_cfg.get("classifier_hidden_units", 0)),
            "pooling_type": base_train_cfg.get("pooling_type", "global_avg"),
            "use_best_checkpoint": True,
            "aug_probability": 0.6,
            "aug_noise_std": 0.015,
            "aug_time_shift_max_fraction": 0.10,
            "aug_time_mask_max_fraction": 0.08,
            "aug_freq_mask_max_fraction": 0.08,
            "mixup_alpha": float(mixup_alpha),
            "secondary_label_weight": float(secondary_weight),
        }

        slot_code = "\n\n".join(
            [slot_base.strip(), _base_build_features_code(), _final_config_override_block(override_cfg)]
        )
        script = assemble_script(slot_code, is_final=True, model_save_path=str(model_path))
        script = _inject_secondary_labels_into_script(script)
        script = _inject_online_augmentation_into_script(script)
        run_id = f"final_aug_{folder_name}"
        script = _append_eval_wrapper(script, run_id, eval_dir)

        script_path = logs_dir / f"{run_id}.py"
        script_path.write_text(script, encoding="utf-8")
        (out_dir / "best_model_code.py").write_text(slot_code, encoding="utf-8")
        (out_dir / "best_params.json").write_text(json.dumps(override_cfg, indent=2), encoding="utf-8")

        print("=" * 60)
        print(f"Running {folder_name} ({variant_label}, weight={secondary_weight})")
        print("=" * 60)
        result = executor.run_file(script_path)
        (out_dir / "train.log").write_text((result.stdout or "") + "\n\n" + (result.stderr or ""), encoding="utf-8")

        if not result.success:
            print(f"FAILED: {folder_name} (see {out_dir / 'train.log'})")
            continue

        yt = eval_dir / f"y_true_{run_id}.npy"
        yp = eval_dir / f"y_pred_{run_id}.npy"
        if yt.exists() and yp.exists():
            ev = evaluator.evaluate_from_files(yt, yp)
            (out_dir / "final_eval_metrics.json").write_text(json.dumps(ev.metrics, indent=2), encoding="utf-8")
            auc = ev.metrics.get("macro_roc_auc")
            print(f"macro_roc_auc={auc}")

        _generate_kaggle_notebook(override_cfg, out_dir / "kaggle_inference.ipynb")
        print(f"DONE: {folder_name} -> {out_dir}")


if __name__ == "__main__":
    main()
