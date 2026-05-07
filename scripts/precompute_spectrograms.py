"""
Step 1: Precompute mel-spectrograms in parallel and cache to disk.
Run once, then use run_final_training.py which loads the cache.

    .venv\Scripts\python.exe scripts\precompute_spectrograms.py
"""
import sys
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 32000
CLIP_SECONDS = 5.0
N_MELS       = 64
N_FRAMES     = 128
NUM_WORKERS  = 8
CACHE_DIR    = root / "data" / "mel_cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_X = CACHE_DIR / "X_all.npy"
CACHE_Y = CACHE_DIR / "y_all.npy"

def load_one(args):
    """Load one audio file and return mel-spectrogram + label vector."""
    audio_path, label_idx, secondary_idxs, n_species = args
    try:
        import librosa, numpy as np
        target_len = int(SAMPLE_RATE * CLIP_SECONDS)
        wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=512, power=2.0
        )
        import librosa.display
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Resize to (N_MELS, N_FRAMES)
        from PIL import Image
        img = Image.fromarray(mel_db).resize((N_FRAMES, N_MELS), Image.BILINEAR)
        mel_arr = np.array(img, dtype=np.float32)[..., np.newaxis]
        # Label
        y = np.zeros(n_species, dtype=np.float32)
        y[label_idx] = 1.0
        for si in secondary_idxs:
            y[si] = 1.0
        return mel_arr, y
    except Exception as e:
        return None, None


def main():
    if CACHE_X.exists() and CACHE_Y.exists():
        X = np.load(CACHE_X)
        print(f"Cache already exists: X={X.shape}. Delete {CACHE_DIR} to recompute.")
        return

    print("Loading metadata...")
    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    train_df = tables["train"]
    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)
    sp2i = {s: i for i, s in enumerate(species_cols)}
    n_species = len(species_cols)

    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"

    # Build task list
    tasks = []
    for row in train_df.itertuples(index=False):
        label = str(getattr(row, lcol))
        if label not in sp2i:
            continue
        ap = paths.train_audio_dir / str(getattr(row, fcol))
        if not ap.exists():
            continue
        try:
            import ast
            sec = getattr(row, "secondary_labels", "[]")
            sec_list = ast.literal_eval(str(sec)) if isinstance(sec, str) else []
            sec_idxs = [sp2i[s] for s in sec_list if s in sp2i]
        except Exception:
            sec_idxs = []
        tasks.append((ap, sp2i[label], sec_idxs, n_species))

    print(f"Files to process: {len(tasks)} across {NUM_WORKERS} workers")
    print("This will take ~20-40 minutes...")

    X_list, y_list = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(load_one, t): t for t in tasks}
        for fut in as_completed(futures):
            mel, y = fut.result()
            if mel is not None:
                X_list.append(mel)
                y_list.append(y)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} done ({len(X_list)} successful)")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    print(f"\nSaving cache: X={X.shape}, y={y.shape}")
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print(f"Saved to {CACHE_DIR}")
    print("Now run: .venv\\Scripts\\python.exe scripts\\run_final_training.py")


if __name__ == "__main__":
    main()
