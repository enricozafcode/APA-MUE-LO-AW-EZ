"""
BirdCLEF Track B — Kaggle inference script.

Run this notebook-style on Kaggle after uploading:
  - best_model.pth
  - label_encoder.pkl
  - model_config.json

It reads test_soundscapes/, splits each file into 5-second windows,
runs the trained CNN, and writes submission.csv.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths — auto-detects Kaggle vs. local
# ---------------------------------------------------------------------------
if Path("/kaggle/input").exists():
    # Kaggle environment
    BASE_DIR       = Path("/kaggle/input")
    DATA_DIR       = BASE_DIR / "birdclef-2026"          # adjust to competition slug
    MODEL_DIR      = BASE_DIR / "birdclef-model"         # your uploaded dataset
    OUTPUT_DIR     = Path("/kaggle/working")
else:
    # Local fallback for testing
    ROOT           = Path(__file__).resolve().parents[1]
    DATA_DIR       = ROOT / "data"
    MODEL_DIR      = ROOT / "logs"
    OUTPUT_DIR     = ROOT / "logs"

SOUNDSCAPES_DIR  = DATA_DIR / "test_soundscapes"
SUBMISSION_TMPL  = DATA_DIR / "sample_submission.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Model (must match train_scaffold.py)
# ---------------------------------------------------------------------------
class BirdCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Audio → mel spectrogram (same settings as training)
# ---------------------------------------------------------------------------
def audio_to_mel(audio: np.ndarray, cfg: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
    )
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    return mel


def load_chunk(audio: np.ndarray, start: int, target_len: int) -> np.ndarray:
    chunk = audio[start: start + target_len]
    if len(chunk) < target_len:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))
    return chunk


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config + label encoder
    with open(MODEL_DIR / "model_config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    num_classes = cfg["num_classes"]
    sr          = cfg["sample_rate"]
    duration    = cfg["duration"]
    target_len  = sr * duration

    # Load model
    model = BirdCNN(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    model.eval()
    print(f"Model loaded  ({num_classes} classes)")

    # Submission template — defines all required row_ids and species columns
    sub_template = pd.read_csv(SUBMISSION_TMPL)
    species_cols = sub_template.columns[1:].tolist()   # all 234 species
    trained_labels = [str(l) for l in le.classes_]     # classes model knows

    # Map trained label → submission column index
    label_to_col = {label: species_cols.index(label)
                    for label in trained_labels if label in species_cols}
    print(f"Mapped {len(label_to_col)}/{num_classes} trained classes to submission columns")

    # Build row_id → index mapping from template
    template_index = {row_id: i for i, row_id in enumerate(sub_template["row_id"])}

    # Collect all soundscape files
    soundscape_files = sorted(SOUNDSCAPES_DIR.glob("*.ogg"))
    if not soundscape_files:
        print("No .ogg files found in test_soundscapes/ — running on Kaggle will populate this.")
        print("Writing uniform-probability submission from template.")
        sub_template.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        return

    # Prepare output dataframe — start with zeros
    results = pd.DataFrame(0.0, index=sub_template.index, columns=sub_template.columns)
    results["row_id"] = sub_template["row_id"]

    for sf in soundscape_files:
        print(f"Processing {sf.name} ...")
        audio, _ = librosa.load(str(sf), sr=sr, mono=True)
        total_seconds = len(audio) // sr

        for end_sec in range(duration, total_seconds + duration, duration):
            start_sample = (end_sec - duration) * sr
            chunk = load_chunk(audio, start_sample, target_len)
            mel = audio_to_mel(chunk, cfg)

            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(mel_tensor), dim=1).cpu().numpy()[0]

            row_id = f"{sf.stem}_{end_sec}"
            if row_id not in template_index:
                continue

            row_idx = template_index[row_id]
            for label_idx, label in enumerate(trained_labels):
                col = label_to_col.get(label)
                if col is not None:
                    results.at[row_idx, col] = float(probs[label_idx])

    out_path = OUTPUT_DIR / "submission.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
