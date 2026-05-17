"""
Stage 1e CNN — pseudo-label unlabeled train_soundscapes and fine-tune the final model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .perch_pseudo import (
        labeled_soundscape_window_keys,
        unlabeled_soundscape_files,
        soundscape_windows_for_file,
    )
except ImportError:
    from perch_pseudo import (
        labeled_soundscape_window_keys,
        unlabeled_soundscape_files,
        soundscape_windows_for_file,
    )

SR = 32_000
CLIP_SEC = 5.0


def _load_species_cols(data_dir: Path) -> list[str]:
    sub = data_dir / "sample_submission.csv"
    df = pd.read_csv(sub)
    return [c for c in df.columns if c != "row_id"]


def build_cnn_pseudo_cache(
    *,
    config: dict,
    teacher_model_path: Path,
    slot_code_path: Path,
    out_path: Path,
    soundscapes_dir: Path | None = None,
    labels_csv: Path | None = None,
    top1_threshold: float = 0.55,
    runnerup_max: float = 0.35,
    pseudo_label_weight: float = 0.8,
    max_files: int | None = None,
) -> dict:
    """Predict on unlabeled windows; save mel tensors + soft multi-hot labels."""
    import librosa
    import tensorflow as tf

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(config.get("data_dir", root / "data"))
    soundscapes_dir = soundscapes_dir or data_dir / "train_soundscapes"
    labels_csv = labels_csv or data_dir / "train_soundscapes_labels.csv"
    species_cols = _load_species_cols(data_dir)
    n_classes = len(species_cols)

    slot_code = slot_code_path.read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(slot_code, "<slot>", "exec"), ns)  # noqa: S102
    cfg = ns["get_training_config"]()
    n_mels = int(cfg.get("n_mels", 64))
    n_frames = int(cfg.get("n_frames", 128))

    model = tf.keras.models.load_model(str(teacher_model_path))
    labeled_keys = labeled_soundscape_window_keys(labels_csv)
    files = unlabeled_soundscape_files(soundscapes_dir, labels_csv)
    if max_files is not None:
        files = files[: int(max_files)]

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    w_list: list[float] = []
    row_ids: list[str] = []

    def _wav_to_mel(wav: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SR, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
        return mel_resized.astype(np.float32)

    accepted = 0
    scanned = 0
    for fi, fp in enumerate(files, 1):
        windows = soundscape_windows_for_file(fp, labeled_keys=labeled_keys)
        if not windows:
            continue
        for wav, row_id, _end in windows:
            scanned += 1
            mel = _wav_to_mel(wav)
            x = mel[np.newaxis, ...]
            pred = model.predict(x, verbose=0)[0]
            order = np.argsort(pred)[::-1]
            top1, top2 = float(pred[order[0]]), float(pred[order[1]]) if len(order) > 1 else 0.0
            if top1 < top1_threshold or top2 >= runnerup_max:
                continue
            y_soft = np.zeros(n_classes, dtype=np.float32)
            y_soft[order[0]] = top1 * pseudo_label_weight
            X_list.append(mel)
            y_list.append(y_soft)
            w_list.append(float(config.get("cnn_pseudo_refine", {}).get("sample_weight_pseudo", 0.5)))
            row_ids.append(row_id)
            accepted += 1
        if fi % 5 == 0:
            print(f"  [pseudo] files {fi}/{len(files)} | accepted={accepted} scanned={scanned}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not X_list:
        X_empty = np.zeros((0, n_mels, n_frames, 1), dtype=np.float32)
        y_empty = np.zeros((0, n_classes), dtype=np.float32)
        np.savez_compressed(
            str(out_path),
            X_pseudo=X_empty,
            y_pseudo=y_empty,
            sample_weight=np.zeros(0, dtype=np.float32),
            row_id=np.array([], dtype=object),
            n_accepted=np.int32(0),
            n_scanned=np.int32(scanned),
            species_cols=np.array(species_cols, dtype=object),
        )
        stats = {
            "n_accepted": 0,
            "n_scanned": scanned,
            "out_path": str(out_path),
            "empty_pseudo": True,
        }
        print(
            f"  [CNN 1e] No pseudo windows passed thresholds ({scanned} scanned) — "
            f"wrote empty cache → {out_path.name}. Fine-tune will use supervised mels only."
        )
        return stats

    np.savez_compressed(
        str(out_path),
        X_pseudo=np.stack(X_list, axis=0),
        y_pseudo=np.stack(y_list, axis=0),
        sample_weight=np.array(w_list, dtype=np.float32),
        row_id=np.array(row_ids, dtype=object),
        n_accepted=np.int32(accepted),
        n_scanned=np.int32(scanned),
        species_cols=np.array(species_cols, dtype=object),
    )
    stats = {"n_accepted": accepted, "n_scanned": scanned, "out_path": str(out_path)}
    print(f"  [pseudo] Saved {accepted} windows → {out_path}")
    return stats


def build_cnn_pseudo_refine_script(
    *,
    slot_code_path: Path,
    teacher_model_path: Path,
    pseudo_npz: Path,
    sample_weight_supervised: float = 1.0,
    sample_weight_pseudo: float = 0.5,
    epochs: int = 15,
    learning_rate: float = 2e-4,
    val_split: float = 0.1,
    model_save_path: str,
) -> str:
    """Fine-tune CNN on focal mels + pseudo mels with warm-start from stage-1d teacher."""
    slot_code = slot_code_path.read_text(encoding="utf-8")
    init_block = f"""
    _teacher = Path(r"{teacher_model_path}")
    if _teacher.exists():
        model = tf.keras.models.load_model(str(_teacher), compile=False)
        print(f"  Warm-start from {{_teacher.name}}")
    else:
        model = build_model(input_shape, num_classes)
"""

    return f'''
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

_PROJECT_ROOT = Path(r"{Path(__file__).resolve().parents[1]}")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("BIRDCLEF_DATA_DIR", str(_PROJECT_ROOT / "data"))

from src.data_io import (
    load_core_tables, resolve_birdclef_paths,
    species_columns_from_sample_submission, validate_required_files,
)
def _wav_to_mel(wav, sample_rate, n_mels, n_frames):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_resized.astype(np.float32)

# --- Locked architecture ---
{slot_code.strip()}

_PSEUDO_NPZ = Path(r"{pseudo_npz}")
_SAVE = Path(r"{model_save_path}")
_FT_EPOCHS = {epochs}
_FT_LR = {learning_rate}
_VAL_SPLIT = {val_split}
_SW_SUP = {sample_weight_supervised}
_SW_PS = {sample_weight_pseudo}

def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()
    cfg["max_samples"] = None
    cfg["epochs"] = _FT_EPOCHS
    cfg["learning_rate"] = _FT_LR
    cfg["val_split"] = _VAL_SPLIT
    sample_rate = int(cfg.get("sample_rate", 32000))
    clip_seconds = float(cfg.get("clip_seconds", 5.0))
    n_mels = int(cfg.get("n_mels", 64))
    n_frames = int(cfg.get("n_frames", 128))
    batch_size = int(cfg.get("batch_size", 32))

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    train_df = tables["train"]
    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)
    n_classes = len(species_cols)
    sp2i = {{s: i for i, s in enumerate(species_cols)}}
    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"

    # Load focal mels (full train)
    X_list, y_list, sw_list = [], [], []
    for row in train_df.itertuples(index=False):
        label = str(getattr(row, lcol))
        if label not in sp2i:
            continue
        ap = paths.train_audio_dir / str(getattr(row, fcol))
        if not ap.exists():
            continue
        import librosa
        wav, _ = librosa.load(str(ap), sr=sample_rate, mono=True, duration=clip_seconds)
        tl = int(sample_rate * clip_seconds)
        if len(wav) < tl:
            wav = np.pad(wav, (0, tl - len(wav)))
        else:
            wav = wav[:tl]
        mel = _wav_to_mel(wav, sample_rate, n_mels, n_frames)
        y = np.zeros(n_classes, dtype=np.float32)
        y[sp2i[label]] = 1.0
        X_list.append(mel)
        y_list.append(y)
        sw_list.append(_SW_SUP)

    d = np.load(str(_PSEUDO_NPZ), allow_pickle=True)
    X_ps = d["X_pseudo"].astype(np.float32)
    y_ps = d["y_pseudo"].astype(np.float32)
    n_ps = int(X_ps.shape[0]) if X_ps.size else 0
    for i in range(n_ps):
        X_list.append(X_ps[i])
        y_list.append(y_ps[i])
        sw_list.append(_SW_PS)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    sw = np.array(sw_list, dtype=np.float32)
    n_sup = len(X_list) - n_ps
    if n_ps == 0:
        print(f"  Pseudo refine: supervised={{n_sup}} pseudo=0 total={{len(X)}} (supervised-only)")
    else:
        print(f"  Pseudo refine: supervised={{n_sup}} pseudo={{n_ps}} total={{len(X)}}")

    input_shape = (n_mels, n_frames, 1)
    num_classes = n_classes
{init_block}
    opt = tf.keras.optimizers.Adam(learning_rate=_FT_LR)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["AUC"])
    n = len(X)
    n_val = max(1, int(n * _VAL_SPLIT))
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    vi, ti = idx[:n_val], idx[n_val:]
    model.fit(
        X[ti], y[ti], sample_weight=sw[ti],
        validation_data=(X[vi], y[vi], sw[vi]),
        epochs=_FT_EPOCHS, batch_size=batch_size, verbose=1,
    )
    _SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(_SAVE)
    print("PSEUDO_REFINE_DONE")

if __name__ == "__main__":
    main()
'''
