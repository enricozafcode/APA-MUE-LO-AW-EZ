import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Data ────────────────────────────────────────────────────────────────────
N_SAMPLES   = 200
N_MELS      = 64
TIME_FRAMES = 128
N_CLASSES   = 234
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 1e-3

# Synthetic data — shape (N, 1, N_MELS, TIME_FRAMES)
X = torch.tensor(np.random.randn(N_SAMPLES, 1, N_MELS, TIME_FRAMES), dtype=torch.float32)
y = torch.tensor((np.random.rand(N_SAMPLES, N_CLASSES) > 0.95), dtype=torch.float32)

split = int(0.8 * N_SAMPLES)
train_ds = TensorDataset(X[:split], y[:split])
val_ds   = TensorDataset(X[split:], y[split:])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── Model — MODIFY ONLY THIS CLASS ──────────────────────────────────────────
class BirdClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        # TODO: replace with your architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ── Training ─────────────────────────────────────────────────────────────────
model     = BirdClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

# ── Evaluation ───────────────────────────────────────────────────────────────
model.eval()
preds, targets = [], []
val_loss = 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        out = model(xb)
        val_loss += criterion(out, yb).item()
        preds.append(torch.sigmoid(out).numpy())
        targets.append(yb.numpy())

preds   = np.vstack(preds)
targets = np.vstack(targets)
val_loss /= len(val_loader)

try:
    roc_auc = roc_auc_score(targets, preds, average="macro")
except Exception:
    roc_auc = 0.0

print("METRICS:", json.dumps({"roc_auc": round(roc_auc, 4), "val_loss": round(val_loss, 4)}))
