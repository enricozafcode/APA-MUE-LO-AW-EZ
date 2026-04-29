"""
BirdCLEF Track B — fixed training scaffold.

This script is the stable base that the agent runs each iteration.
Augmentation strategies are controlled via configs/agent_config.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "train_audio"
CONFIG_PATH = ROOT / "configs" / "agent_config.json"

sys.path.insert(0, str(ROOT / "src"))
from augmentation import build_augmenters  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with CONFIG_PATH.open() as f:
    CFG = json.load(f)

SAMPLE_RATE = 22050
DURATION = 5          # seconds per clip
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
MAX_SAMPLES = 3000    # cap for speed; set to None to use all data
NUM_CLASSES = 206
RANDOM_SEED = CFG.get("random_seed", 42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.augment = augment
        self.audio_aug, self.spec_aug = build_augmenters(CFG)
        self.mel_transform = librosa.feature.melspectrogram

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = AUDIO_DIR / row["filename"]

        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=DURATION, mono=True)
        except Exception:
            audio = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)

        # Pad / trim to fixed length
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Audio augmentation (training only)
        if self.augment:
            audio = self.audio_aug.apply(audio, SAMPLE_RATE)
            # Re-trim/pad after augmentation (time_stretch changes length)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # Normalize to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        # Spectrogram augmentation (training only)
        if self.augment:
            mel = self.spec_aug.apply(mel)

        # Shape: (1, n_mels, time)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        label = int(self.le.transform([row["primary_label"]])[0])
        return mel_tensor, label


# ---------------------------------------------------------------------------
# Model
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
# cmAP
# ---------------------------------------------------------------------------

def compute_cmap(all_labels: list, all_probs: list, num_classes: int) -> float:
    y_true = np.zeros((len(all_labels), num_classes), dtype=int)
    for i, label in enumerate(all_labels):
        y_true[i, label] = 1
    y_score = np.array(all_probs)
    # Only score classes that appear in y_true
    present = np.where(y_true.sum(axis=0) > 0)[0]
    if len(present) == 0:
        return 0.0
    return float(average_precision_score(y_true[:, present], y_score[:, present], average="macro"))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, num_classes):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    return compute_cmap(all_labels, all_probs, num_classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata
    df = pd.read_csv(DATA_DIR / "train.csv")
    df = df[["primary_label", "filename"]].dropna()

    # Keep only classes with enough samples
    counts = df["primary_label"].value_counts()
    df = df[df["primary_label"].isin(counts[counts >= 2].index)]

    # Optional cap for speed — sample evenly per class
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        per_class = max(1, MAX_SAMPLES // df["primary_label"].nunique())
        parts = [
            grp.sample(min(len(grp), per_class), random_state=RANDOM_SEED)
            for _, grp in df.groupby("primary_label")
        ]
        df = pd.concat(parts, ignore_index=True)

    le = LabelEncoder()
    le.fit(df["primary_label"])
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}  |  Samples: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["primary_label"], random_state=RANDOM_SEED)

    train_ds = BirdDataset(train_df, le, augment=True)
    val_ds   = BirdDataset(val_df,   le, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BirdCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Active augmentation info
    audio_aug, spec_aug = build_augmenters(CFG)
    print(f"Audio augmentations:       {audio_aug.active_strategies()}")
    print(f"Spectrogram augmentations: {spec_aug.active_strategies()}")

    best_cmap = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        cmap = eval_epoch(model, val_loader, device, num_classes)
        scheduler.step()
        print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}  cmAP={cmap:.4f}")
        if cmap > best_cmap:
            best_cmap = cmap

    print(f"\ncmAP: {best_cmap:.4f}")


if __name__ == "__main__":
    main()
