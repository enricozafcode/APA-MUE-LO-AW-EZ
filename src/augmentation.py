"""
Audio and spectrogram augmentation for BirdCLEF Track B.

Each strategy can be individually enabled/disabled via the config dict
(loaded from agent_config.json -> "augmentation" key).

Meta-agent stage 1a uses three shared baselines (light / medium / high):
  - ``get_audio_embedding_aug(name)`` — BirdNET & Perch (audio + optional SNR mix)
  - ``get_cnn_baseline_aug(name)`` — CNN (soundscape SNR + audio aug before mel, then spec aug)
  - ``get_cnn_spectrogram_aug(name)`` — spectrogram-only knobs (subset of CNN baseline)

Usage:
    audio_aug = AudioAugmenter(config["augmentation"]["audio"])
    augmented_audio = audio_aug.apply(audio, sr)

    spec_aug = SpectrogramAugmenter(config["augmentation"]["spectrogram"])
    augmented_spec = spec_aug.apply(spec)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


BASELINE_AUG_NAMES: tuple[str, ...] = ("light", "medium", "high")


# ---------------------------------------------------------------------------
# Embedding-track baselines (BirdNET / Perch) — applied before encoder
# ---------------------------------------------------------------------------

AUDIO_EMBEDDING_BASELINES: dict[str, dict[str, Any]] = {
    "light": {
        "name": "light",
        "use_snr_mixing": True,
        "n_mix_views": 2,
        "mix_prob": 0.20,
        "snr_min_db": 8.0,
        "snr_max_db": 22.0,
        "audio": {
            "noise_injection": {"enabled": True, "probability": 0.30, "noise_level": 0.003},
            "time_shift": {"enabled": True, "probability": 0.35, "shift_max_fraction": 0.25},
            "gain_jitter": {"enabled": True, "probability": 0.30, "min_db": -4.0, "max_db": 4.0},
            "time_stretch": {"enabled": False},
            "pitch_shift": {"enabled": False},
        },
    },
    "medium": {
        "name": "medium",
        "use_snr_mixing": True,
        "n_mix_views": 2,
        "mix_prob": 0.35,
        "snr_min_db": 0.0,
        "snr_max_db": 15.0,
        "audio": {
            "noise_injection": {"enabled": True, "probability": 0.40, "noise_level": 0.005},
            "time_shift": {"enabled": True, "probability": 0.50, "shift_max_fraction": 0.40},
            "gain_jitter": {"enabled": True, "probability": 0.40, "min_db": -6.0, "max_db": 6.0},
            "time_stretch": {"enabled": False},
            "pitch_shift": {"enabled": False},
        },
    },
    "high": {
        "name": "high",
        "use_snr_mixing": True,
        "n_mix_views": 2,
        "mix_prob": 0.50,
        "snr_min_db": 0.0,
        "snr_max_db": 10.0,
        "audio": {
            "time_stretch": {"enabled": True, "probability": 0.45, "rate_min": 0.90, "rate_max": 1.10},
            "pitch_shift": {"enabled": True, "probability": 0.30, "steps_min": -2, "steps_max": 2},
            "noise_injection": {"enabled": True, "probability": 0.50, "noise_level": 0.008},
            "time_shift": {"enabled": True, "probability": 0.50, "shift_max_fraction": 0.50},
            "gain_jitter": {"enabled": True, "probability": 0.50, "min_db": -8.0, "max_db": 8.0},
        },
    },
}


# ---------------------------------------------------------------------------
# CNN-track baselines — applied on mel spectrograms in the harness
# ---------------------------------------------------------------------------

CNN_SPECTROGRAM_BASELINES: dict[str, dict[str, float | int]] = {
    "light": {
        "aug_prob": 0.50,
        "aug_noise_std": 0.003,
        "aug_time_mask": 8,
        "aug_freq_mask": 4,
    },
    "medium": {
        "aug_prob": 0.75,
        "aug_noise_std": 0.007,
        "aug_time_mask": 16,
        "aug_freq_mask": 8,
    },
    "high": {
        "aug_prob": 1.00,
        "aug_noise_std": 0.012,
        "aug_time_mask": 24,
        "aug_freq_mask": 12,
    },
}


def list_baseline_aug_names() -> list[str]:
    return list(BASELINE_AUG_NAMES)


def get_audio_embedding_aug(name: str) -> dict[str, Any]:
    """Full augmentation dict for embedding caches (BirdNET / Perch)."""
    key = str(name).strip().lower()
    if key not in AUDIO_EMBEDDING_BASELINES:
        raise KeyError(f"Unknown audio baseline {name!r}; choose from {list(BASELINE_AUG_NAMES)}")
    preset = AUDIO_EMBEDDING_BASELINES[key]
    return {k: v for k, v in preset.items() if k != "name"}


def get_cnn_spectrogram_aug(name: str) -> dict[str, float | int]:
    """Spectrogram augmentation knobs for the CNN harness / generate_slot_code."""
    key = str(name).strip().lower()
    if key not in CNN_SPECTROGRAM_BASELINES:
        raise KeyError(f"Unknown CNN baseline {name!r}; choose from {list(BASELINE_AUG_NAMES)}")
    return dict(CNN_SPECTROGRAM_BASELINES[key])


def get_cnn_baseline_aug(name: str) -> dict[str, Any]:
    """
    Full CNN stage-1a baseline: ``aug_preset`` drives audio + SNR (same as embed tracks),
    plus locked mel spectrogram augmentation fields.
    """
    key = str(name).strip().lower()
    if key not in BASELINE_AUG_NAMES:
        raise KeyError(f"Unknown baseline {name!r}; choose from {list(BASELINE_AUG_NAMES)}")
    return {"aug_preset": key, **get_cnn_spectrogram_aug(key)}


def describe_baseline(name: str) -> str:
    """One-line summary for logs."""
    key = str(name).strip().lower()
    audio = AUDIO_EMBEDDING_BASELINES.get(key, {})
    cnn = CNN_SPECTROGRAM_BASELINES.get(key, {})
    snr = "SNR" if audio.get("use_snr_mixing") else "no-SNR"
    return (
        f"{key}: embed[{snr}, views={audio.get('n_mix_views', 1)}] | "
        f"CNN[{snr}+spec, prob={cnn.get('aug_prob')}, tm={cnn.get('aug_time_mask')}, fm={cnn.get('aug_freq_mask')}]"
    )


# ---------------------------------------------------------------------------
# SNR mixing helpers (shared by BirdNET / Perch cache builders)
# ---------------------------------------------------------------------------

def mix_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal with noise at a given SNR (dB) using power-based scaling."""
    ps = np.mean(signal ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return np.clip(signal + scale * noise, -1.0, 1.0).astype(np.float32)


def load_random_soundscape_noise(
    rng: np.random.Generator,
    noise_pool: list[Path],
    *,
    sr: int,
    clip_sec: float,
) -> np.ndarray | None:
    """Load a random segment from a soundscape file as background noise."""
    if not noise_pool or not _LIBROSA_AVAILABLE:
        return None
    fp = noise_pool[int(rng.integers(0, len(noise_pool)))]
    try:
        dur = librosa.get_duration(path=str(fp))
        start_max = max(0.0, dur - clip_sec)
        offset = float(rng.uniform(0, start_max)) if start_max > 0 else 0.0
        n, _ = librosa.load(str(fp), sr=sr, mono=True, offset=offset, duration=clip_sec)
        nt = int(clip_sec * sr)
        n = n[:nt] if len(n) > nt else np.pad(n, (0, nt - len(n)))
        return n.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Audio-domain augmentations
# ---------------------------------------------------------------------------

class AudioAugmenter:
    """Applies enabled audio-domain augmentations in sequence."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _enabled(self, strategy: str) -> bool:
        return self.cfg.get(strategy, {}).get("enabled", False)

    def _prob(self, strategy: str) -> float:
        return self.cfg.get(strategy, {}).get("probability", 0.5)

    def _roll(self, strategy: str) -> bool:
        return np.random.rand() < self._prob(strategy)

    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not _LIBROSA_AVAILABLE:
            return audio
        c = self.cfg["time_stretch"]
        rate = np.random.uniform(c.get("rate_min", 0.9), c.get("rate_max", 1.1))
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not _LIBROSA_AVAILABLE:
            return audio
        c = self.cfg["pitch_shift"]
        n_steps = np.random.randint(c.get("steps_min", -2), c.get("steps_max", 2) + 1)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def noise_injection(self, audio: np.ndarray, sr: int) -> np.ndarray:
        c = self.cfg["noise_injection"]
        level = c.get("noise_level", 0.005)
        return audio + level * np.random.randn(len(audio)).astype(audio.dtype)

    def time_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        c = self.cfg["time_shift"]
        max_fraction = c.get("shift_max_fraction", 0.5)
        shift = int(np.random.uniform(-max_fraction, max_fraction) * len(audio))
        return np.roll(audio, shift)

    def gain_jitter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        c = self.cfg["gain_jitter"]
        gain_db = float(np.random.uniform(c.get("min_db", -6.0), c.get("max_db", 6.0)))
        return np.clip(audio * (10.0 ** (gain_db / 20.0)), -1.0, 1.0).astype(audio.dtype)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply all enabled strategies with their individual probabilities."""
        if self._enabled("time_stretch") and self._roll("time_stretch"):
            audio = self.time_stretch(audio, sr)
        if self._enabled("pitch_shift") and self._roll("pitch_shift"):
            audio = self.pitch_shift(audio, sr)
        if self._enabled("noise_injection") and self._roll("noise_injection"):
            audio = self.noise_injection(audio, sr)
        if self._enabled("time_shift") and self._roll("time_shift"):
            audio = self.time_shift(audio, sr)
        if self._enabled("gain_jitter") and self._roll("gain_jitter"):
            audio = self.gain_jitter(audio, sr)
        return audio

    def active_strategies(self) -> list[str]:
        return [
            s
            for s in ["time_stretch", "pitch_shift", "noise_injection", "time_shift", "gain_jitter"]
            if self._enabled(s)
        ]


# ---------------------------------------------------------------------------
# Spectrogram-domain augmentations
# ---------------------------------------------------------------------------

class SpectrogramAugmenter:
    """Applies enabled spectrogram-domain augmentations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _enabled(self, strategy: str) -> bool:
        return self.cfg.get(strategy, {}).get("enabled", False)

    def time_mask(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg["time_mask"]
        T = c.get("max_mask_size", 40)
        n = c.get("num_masks", 1)
        spec = spec.copy()
        for _ in range(n):
            t = np.random.randint(0, max(T, 1))
            t0 = np.random.randint(0, max(spec.shape[1] - t, 1))
            spec[:, t0 : t0 + t] = 0
        return spec

    def freq_mask(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg["freq_mask"]
        F = c.get("max_mask_size", 20)
        n = c.get("num_masks", 1)
        spec = spec.copy()
        for _ in range(n):
            f = np.random.randint(0, max(F, 1))
            f0 = np.random.randint(0, max(spec.shape[0] - f, 1))
            spec[f0 : f0 + f, :] = 0
        return spec

    def mixup(
        self,
        spec1: np.ndarray,
        label1: np.ndarray,
        spec2: np.ndarray,
        label2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg.get("mixup", {})
        alpha = c.get("alpha", 0.4)
        lam = np.random.beta(alpha, alpha)
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_spec, mixed_label

    def cutmix(
        self,
        spec1: np.ndarray,
        label1: np.ndarray,
        spec2: np.ndarray,
        label2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg.get("cutmix", {})
        alpha = c.get("alpha", 1.0)
        lam = np.random.beta(alpha, alpha)
        H, W = spec1.shape
        cut_h = int(H * np.sqrt(1 - lam))
        cut_w = int(W * np.sqrt(1 - lam))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        spec1 = spec1.copy()
        spec1[y1:y2, x1:x2] = spec2[y1:y2, x1:x2]
        actual_lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)
        mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        return spec1, mixed_label

    def apply(self, spec: np.ndarray) -> np.ndarray:
        if self._enabled("time_mask"):
            spec = self.time_mask(spec)
        if self._enabled("freq_mask"):
            spec = self.freq_mask(spec)
        return spec

    def active_strategies(self) -> list[str]:
        return [s for s in ["time_mask", "freq_mask", "mixup", "cutmix"] if self._enabled(s)]


def build_augmenters(config: dict) -> tuple[AudioAugmenter, SpectrogramAugmenter]:
    aug_cfg = config.get("augmentation", {})
    audio_aug = AudioAugmenter(aug_cfg.get("audio", {}))
    spec_aug = SpectrogramAugmenter(aug_cfg.get("spectrogram", {}))
    return audio_aug, spec_aug
