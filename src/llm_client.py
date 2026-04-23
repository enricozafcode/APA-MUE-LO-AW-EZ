"""Client for communicating with a locally-hosted LLM via OpenAI-compatible API."""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI


class LLMClient:
    """Wraps the local LLM server (Ollama / LM Studio) with two focused methods."""

    def __init__(self, provider: str = "ollama", model: str = "deepseek-r1:8b") -> None:
        self.model_name = model
        base_url = (
            "http://localhost:11434/v1" if provider.lower() == "ollama"
            else "http://localhost:1234/v1"
        )
        self.client = OpenAI(base_url=base_url, api_key="local-dummy-key", timeout=1800.0)

    def _call(self, system: str, user: str, temperature: float) -> str:
        """Single point of contact with the LLM server."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Removes markdown code fences that some models add despite instructions."""
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    def generate_plan(
        self,
        data_summary: dict,
        instructions: str,
        temperature: float = 0.2,
    ) -> str:
        """
        Asks the LLM to propose a free-text experiment plan given a data summary.

        Returns a plain-text string describing what to try and why.
        """
        system = (
            "You are an expert ML researcher specialising in audio classification. "
            "Propose a concise, focused experiment plan. Be specific about model architecture, "
            "audio preprocessing, and training setup. Max 200 words."
        )
        user = (
            f"Data summary:\n{json.dumps(data_summary, indent=2)}\n\n"
            f"Instructions:\n{instructions}"
        )
        return self._call(system, user, temperature)

    def generate_code(
        self,
        plan: str,
        data_dir: Path,
        exp_dir: Path,
        temperature: float = 0.2,
    ) -> str:
        """
        Asks the LLM to generate the CNN model block only; the rest is a fixed skeleton.

        The script must write a metrics.json file to exp_dir before it exits.
        """
        metrics_path = exp_dir / "metrics.json"
        audio_dir = data_dir / "train_audio"
        train_csv = data_dir / "train.csv"

        system = (
            "You are an expert Python/Keras developer. "
            "Output ONLY raw Python code — no markdown fences, no explanations."
        )

        skeleton = (
            f"import time, json, os\n"
            f"import librosa\n"
            f"import numpy as np\n"
            f"import pandas as pd\n"
            f"from sklearn.model_selection import train_test_split\n"
            f"from sklearn.preprocessing import LabelEncoder\n"
            f"import tensorflow as tf\n"
            f"from tensorflow.keras import layers, models\n"
            f"\n"
            f"DATA_CSV   = r'{train_csv}'\n"
            f"AUDIO_DIR  = r'{audio_dir}'\n"
            f"METRICS_PATH = r'{metrics_path}'\n"
            f"N_MELS, HOP_LENGTH, TIME_STEPS = 64, 512, 216\n"
            f"DURATION, BATCH_SIZE, EPOCHS = 5.0, 16, 3\n"
            f"\n"
            f"start_time = time.time()\n"
            f"try:\n"
            f"    df = pd.read_csv(DATA_CSV).head(50)\n"
            f"    X, y_labels = [], []\n"
            f"    for _, row in df.iterrows():\n"
            f"        try:\n"
            f"            audio, sr = librosa.load(os.path.join(AUDIO_DIR, row['filename']), sr=22050, duration=DURATION)\n"
            f"            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)\n"
            f"            mel = librosa.power_to_db(mel, ref=np.max)  # shape: (N_MELS, T)\n"
            f"            if mel.shape[1] < TIME_STEPS:\n"
            f"                mel = np.pad(mel, ((0, 0), (0, TIME_STEPS - mel.shape[1])))\n"
            f"            else:\n"
            f"                mel = mel[:, :TIME_STEPS]\n"
            f"            X.append(mel.T)  # shape: (TIME_STEPS, N_MELS)\n"
            f"            y_labels.append(row['primary_label'])\n"
            f"        except Exception as load_err:\n"
            f"            print(f'Skip {{row[\"filename\"]}}: {{load_err}}')\n"
            f"            continue\n"
            f"    X = np.array(X)[..., np.newaxis]  # (N, TIME_STEPS, N_MELS, 1)\n"
            f"    le = LabelEncoder()\n"
            f"    y = le.fit_transform(y_labels)\n"
            f"    n_classes = len(le.classes_)\n"
            f"    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            f"\n"
            f"    # === YOUR MODEL HERE ===\n"
            f"    # Input shape: (TIME_STEPS={216}, N_MELS={64}, 1)\n"
            f"    # Output: Dense(n_classes, activation='softmax')\n"
            f"    # Loss: sparse_categorical_crossentropy\n"
            f"    # INSERT model = models.Sequential([...]) HERE\n"
            f"\n"
            f"    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
            f"    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,\n"
            f"                        validation_data=(X_val, y_val), verbose=1)\n"
            f"    _, val_acc = model.evaluate(X_val, y_val, verbose=0)\n"
            f"    metrics = {{\n"
            f'        "success": True, "model_type": "small_cnn",\n'
            f'        "train_loss": float(history.history["loss"][-1]),\n'
            f'        "val_loss": float(history.history["val_loss"][-1]),\n'
            f'        "val_auc": float(val_acc),\n'
            f'        "epochs_completed": EPOCHS,\n'
            f'        "runtime_seconds": float(time.time() - start_time)\n'
            f"    }}\n"
            f"except Exception as e:\n"
            f"    metrics = {{\n"
            f'        "success": False, "error_type": type(e).__name__, "error_message": str(e)\n'
            f"    }}\n"
            f"with open(METRICS_PATH, 'w') as f:\n"
            f"    json.dump(metrics, f, indent=2)\n"
        )

        user = (
            f"Experiment plan for context:\n{plan}\n\n"
            f"Complete the following Python training script by replacing the "
            f"'# INSERT model = models.Sequential([...]) HERE' comment with a working Keras model.\n\n"
            f"CRITICAL: Copy the script EXACTLY as given. Do NOT modify any line outside the model block. "
            f"Do NOT add or change any parameters to librosa.load(), melspectrogram(), or any other call.\n\n"
            f"Rules for the model block only:\n"
            f"- Use Conv2D layers (input shape is (TIME_STEPS, N_MELS, 1) = (216, 64, 1))\n"
            f"- 2-3 Conv2D layers, each followed by MaxPooling2D\n"
            f"- GlobalAveragePooling2D before the output\n"
            f"- Final layer: Dense(n_classes, activation='softmax')\n"
            f"- You choose the filter counts and kernel sizes\n\n"
            f"Output the COMPLETE script with your model inserted. Change NOTHING else.\n\n"
            f"{skeleton}"
        )
        return self._strip_markdown(self._call(system, user, temperature))
