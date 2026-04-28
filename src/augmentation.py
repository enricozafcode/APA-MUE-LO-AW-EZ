"""
Audio and spectrogram augmentation for BirdCLEF Track B.

Each strategy can be individually enabled/disabled via the config dict
(loaded from agent_config.json -> "augmentation" key).

Usage:
    audio_aug = AudioAugmenter(config["augmentation"]["audio"])
    augmented_audio = audio_aug.apply(audio, sr)

    spec_aug = SpectrogramAugmenter(config["augmentation"]["spectrogram"])
    augmented_spec = spec_aug.apply(spec)

    # Mixup requires a batch — call at batch level:
    mixed_spec, mixed_label = spec_aug.mixup(spec1, label1, spec2, label2)
"""

from __future__ import annotations

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


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

    # --- individual strategies ---

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

    # --- main entry point ---

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
        return audio

    def active_strategies(self) -> list[str]:
        """Returns names of currently enabled strategies."""
        return [s for s in ["time_stretch", "pitch_shift", "noise_injection", "time_shift"]
                if self._enabled(s)]


# ---------------------------------------------------------------------------
# Spectrogram-domain augmentations
# ---------------------------------------------------------------------------

class SpectrogramAugmenter:
    """Applies enabled spectrogram-domain augmentations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _enabled(self, strategy: str) -> bool:
        return self.cfg.get(strategy, {}).get("enabled", False)

    # --- individual strategies ---

    def time_mask(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg["time_mask"]
        T = c.get("max_mask_size", 40)
        n = c.get("num_masks", 1)
        spec = spec.copy()
        for _ in range(n):
            t = np.random.randint(0, max(T, 1))
            t0 = np.random.randint(0, max(spec.shape[1] - t, 1))
            spec[:, t0:t0 + t] = 0
        return spec

    def freq_mask(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg["freq_mask"]
        F = c.get("max_mask_size", 20)
        n = c.get("num_masks", 1)
        spec = spec.copy()
        for _ in range(n):
            f = np.random.randint(0, max(F, 1))
            f0 = np.random.randint(0, max(spec.shape[0] - f, 1))
            spec[f0:f0 + f, :] = 0
        return spec

    def mixup(
        self,
        spec1: np.ndarray,
        label1: np.ndarray,
        spec2: np.ndarray,
        label2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mixes two spectrograms and their (soft) labels."""
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
        """Cuts a random rectangle from spec2 and pastes it into spec1."""
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

    # --- main entry point ---

    def apply(self, spec: np.ndarray) -> np.ndarray:
        """Apply per-sample strategies (time_mask, freq_mask) to a single spectrogram.
        Mixup/CutMix must be called explicitly at batch level."""
        if self._enabled("time_mask"):
            spec = self.time_mask(spec)
        if self._enabled("freq_mask"):
            spec = self.freq_mask(spec)
        return spec

    def active_strategies(self) -> list[str]:
        return [s for s in ["time_mask", "freq_mask", "mixup", "cutmix"]
                if self._enabled(s)]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_augmenters(config: dict) -> tuple[AudioAugmenter, SpectrogramAugmenter]:
    """
    Builds both augmenters from the full agent config.

    Args:
        config: full agent_config dict (must contain "augmentation" key)

    Returns:
        (AudioAugmenter, SpectrogramAugmenter)
    """
    aug_cfg = config.get("augmentation", {})
    audio_aug = AudioAugmenter(aug_cfg.get("audio", {}))
    spec_aug = SpectrogramAugmenter(aug_cfg.get("spectrogram", {}))
    return audio_aug, spec_aug
