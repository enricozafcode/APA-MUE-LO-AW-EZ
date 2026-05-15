"""Builds a complete training script deterministically from a structured plan dict."""

from __future__ import annotations

from pathlib import Path


def build_model_block(plan: dict) -> str:
    """Generates the Keras Sequential model definition from plan parameters."""
    conv_layers = plan.get("conv_layers", [{"filters": 32, "kernel_size": 3}])
    dropout = float(plan.get("dropout", 0.0))

    lines = ["    model = models.Sequential(["]
    for i, layer in enumerate(conv_layers):
        filters = layer["filters"]
        kernel = layer["kernel_size"]
        if i == 0:
            lines.append(
                f"        layers.Conv2D({filters}, ({kernel}, {kernel}), activation='relu', "
                f"input_shape=(TIME_STEPS, N_MELS, 1)),"
            )
        else:
            lines.append(
                f"        layers.Conv2D({filters}, ({kernel}, {kernel}), activation='relu'),"
            )
        lines.append("        layers.MaxPooling2D((2, 2)),")
        if dropout > 0:
            lines.append(f"        layers.Dropout({dropout}),")
    lines.append("        layers.GlobalAveragePooling2D(),")
    lines.append("        layers.Dense(n_classes, activation='softmax'),")
    lines.append("    ])")
    return "\n".join(lines)


def build_pretrained_script(plan: dict, data_dir: Path, exp_dir: Path) -> str:
    """Generates a MobileNetV2 transfer learning script from the plan."""
    metrics_path = exp_dir / "metrics.json"
    audio_dir = data_dir / "train_audio"
    train_csv = data_dir / "train.csv"

    lr = plan.get("learning_rate", 0.001)
    batch_size = plan.get("batch_size", 16)
    epochs = plan.get("epochs", 3)
    freeze = plan.get("freeze_backbone", True)
    trainable_str = "False" if freeze else "True"

    return (
        f"import time, json, os\n"
        f"import librosa\n"
        f"import numpy as np\n"
        f"import pandas as pd\n"
        f"from sklearn.model_selection import train_test_split\n"
        f"from sklearn.preprocessing import LabelEncoder\n"
        f"import tensorflow as tf\n"
        f"from tensorflow.keras import layers, models\n"
        f"\n"
        f"DATA_CSV     = r'{train_csv}'\n"
        f"AUDIO_DIR    = r'{audio_dir}'\n"
        f"METRICS_PATH = r'{metrics_path}'\n"
        f"N_MELS, HOP_LENGTH, IMG_SIZE = 64, 512, 128\n"
        f"DURATION = 5.0\n"
        f"BATCH_SIZE, EPOCHS = {batch_size}, {epochs}\n"
        f"LEARNING_RATE = {lr}\n"
        f"\n"
        f"start_time = time.time()\n"
        f"try:\n"
        f"    df_full = pd.read_csv(DATA_CSV)\n"
        f"    df = (df_full.groupby('primary_label', group_keys=False)\n"
        f"               .apply(lambda x: x.sample(min(len(x), 5), random_state=42))\n"
        f"               .reset_index(drop=True))\n"
        f"    X, y_labels = [], []\n"
        f"    for _, row in df.iterrows():\n"
        f"        try:\n"
        f"            audio, sr = librosa.load(os.path.join(AUDIO_DIR, row['filename']),\n"
        f"                                     sr=22050, duration=DURATION)\n"
        f"            mel = librosa.feature.melspectrogram(y=audio, sr=sr,\n"
        f"                                                 n_mels=N_MELS, hop_length=HOP_LENGTH)\n"
        f"            mel = librosa.power_to_db(mel, ref=np.max)\n"
        f"            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)\n"
        f"            mel_resized = tf.image.resize(mel[..., np.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()\n"
        f"            mel_3ch = np.concatenate([mel_resized, mel_resized, mel_resized], axis=-1)\n"
        f"            X.append(mel_3ch)\n"
        f"            y_labels.append(row['primary_label'])\n"
        f"        except Exception as load_err:\n"
        f"            print(f'Skip {{row[\"filename\"]}}: {{load_err}}')\n"
        f"            continue\n"
        f"    X = np.array(X)\n"
        f"    le = LabelEncoder()\n"
        f"    y = le.fit_transform(y_labels)\n"
        f"    n_classes = len(le.classes_)\n"
        f"    X_train, X_val, y_train, y_val = train_test_split(\n"
        f"        X, y, test_size=0.2, random_state=42)\n"
        f"\n"
        f"    base_model = tf.keras.applications.MobileNetV2(\n"
        f"        input_shape=(IMG_SIZE, IMG_SIZE, 3),\n"
        f"        include_top=False,\n"
        f"        weights='imagenet',\n"
        f"    )\n"
        f"    base_model.trainable = {trainable_str}\n"
        f"    x = base_model.output\n"
        f"    x = layers.GlobalAveragePooling2D()(x)\n"
        f"    output = layers.Dense(n_classes, activation='softmax')(x)\n"
        f"    model = tf.keras.Model(inputs=base_model.input, outputs=output)\n"
        f"\n"
        f"    model.compile(\n"
        f"        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),\n"
        f"        loss='sparse_categorical_crossentropy',\n"
        f"        metrics=['accuracy'],\n"
        f"    )\n"
        f"    history = model.fit(\n"
        f"        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,\n"
        f"        validation_data=(X_val, y_val), verbose=1,\n"
        f"    )\n"
        f"    _, val_acc = model.evaluate(X_val, y_val, verbose=0)\n"
        f"    metrics = {{\n"
        f'        "success": True,\n'
        f'        "model_type": "pretrained_mobilenetv2",\n'
        f'        "freeze_backbone": {str(freeze).lower()},\n'
        f'        "train_loss": float(history.history["loss"][-1]),\n'
        f'        "val_loss": float(history.history["val_loss"][-1]),\n'
        f'        "val_auc": float(val_acc),\n'
        f'        "epochs_completed": EPOCHS,\n'
        f'        "runtime_seconds": float(time.time() - start_time),\n'
        f"    }}\n"
        f"except ImportError as e:\n"
        f"    metrics = {{\n"
        f'        "success": False,\n'
        f'        "error_type": "missing_dependency",\n'
        f'        "error_message": str(e),\n'
        f"    }}\n"
        f"except Exception as e:\n"
        f"    metrics = {{\n"
        f'        "success": False,\n'
        f'        "error_type": type(e).__name__,\n'
        f'        "error_message": str(e),\n'
        f"    }}\n"
        f"with open(METRICS_PATH, 'w') as f:\n"
        f"    json.dump(metrics, f, indent=2)\n"
    )


def build_training_script(plan: dict, data_dir: Path, exp_dir: Path) -> str:
    """
    Routes to the correct script builder based on plan['model_type'].
    Returns a complete, self-contained Python training script.
    """
    if plan.get("model_type") == "pretrained":
        return build_pretrained_script(plan, data_dir, exp_dir)
    return _build_cnn_script(plan, data_dir, exp_dir)


def _build_cnn_script(plan: dict, data_dir: Path, exp_dir: Path) -> str:
    """
    Returns a complete, self-contained Python training script.

    All CNN hyperparameters are injected from plan — no LLM involvement.
    The script always writes metrics.json to exp_dir on success or failure.
    """
    metrics_path = exp_dir / "metrics.json"
    audio_dir = data_dir / "train_audio"
    train_csv = data_dir / "train.csv"

    lr = plan.get("learning_rate", 0.001)
    batch_size = plan.get("batch_size", 16)
    epochs = plan.get("epochs", 3)
    model_block = build_model_block(plan)

    return (
        f"import time, json, os\n"
        f"import librosa\n"
        f"import numpy as np\n"
        f"import pandas as pd\n"
        f"from sklearn.model_selection import train_test_split\n"
        f"from sklearn.preprocessing import LabelEncoder\n"
        f"import tensorflow as tf\n"
        f"from tensorflow.keras import layers, models\n"
        f"\n"
        f"DATA_CSV     = r'{train_csv}'\n"
        f"AUDIO_DIR    = r'{audio_dir}'\n"
        f"METRICS_PATH = r'{metrics_path}'\n"
        f"# Audio preprocessing constants — future: drive from plan['num_mels'], plan['spectrogram_resolution']\n"
        f"N_MELS, HOP_LENGTH, TIME_STEPS = 64, 512, 216\n"
        f"DURATION = 5.0\n"
        f"# THRESHOLDING = 'none'  # future: plan['thresholding_strategy']\n"
        f"BATCH_SIZE, EPOCHS = {batch_size}, {epochs}\n"
        f"LEARNING_RATE = {lr}\n"
        f"\n"
        f"start_time = time.time()\n"
        f"try:\n"
        f"    df_full = pd.read_csv(DATA_CSV)\n"
        f"    df = (df_full.groupby('primary_label', group_keys=False)\n"
        f"               .apply(lambda x: x.sample(min(len(x), 5), random_state=42))\n"
        f"               .reset_index(drop=True))\n"
        f"    X, y_labels = [], []\n"
        f"    for _, row in df.iterrows():\n"
        f"        try:\n"
        f"            audio, sr = librosa.load(os.path.join(AUDIO_DIR, row['filename']),\n"
        f"                                     sr=22050, duration=DURATION)\n"
        f"            mel = librosa.feature.melspectrogram(y=audio, sr=sr,\n"
        f"                                                 n_mels=N_MELS, hop_length=HOP_LENGTH)\n"
        f"            mel = librosa.power_to_db(mel, ref=np.max)  # (N_MELS, T)\n"
        f"            if mel.shape[1] < TIME_STEPS:\n"
        f"                mel = np.pad(mel, ((0, 0), (0, TIME_STEPS - mel.shape[1])))\n"
        f"            else:\n"
        f"                mel = mel[:, :TIME_STEPS]\n"
        f"            X.append(mel.T)  # (TIME_STEPS, N_MELS)\n"
        f"            y_labels.append(row['primary_label'])\n"
        f"        except Exception as load_err:\n"
        f"            print(f'Skip {{row[\"filename\"]}}: {{load_err}}')\n"
        f"            continue\n"
        f"    X = np.array(X)[..., np.newaxis]  # (N, TIME_STEPS, N_MELS, 1)\n"
        f"    le = LabelEncoder()\n"
        f"    y = le.fit_transform(y_labels)\n"
        f"    n_classes = len(le.classes_)\n"
        f"    X_train, X_val, y_train, y_val = train_test_split(\n"
        f"        X, y, test_size=0.2, random_state=42)\n"
        f"\n"
        f"{model_block}\n"
        f"\n"
        f"    model.compile(\n"
        f"        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),\n"
        f"        loss='sparse_categorical_crossentropy',\n"
        f"        metrics=['accuracy'],\n"
        f"    )\n"
        f"    history = model.fit(\n"
        f"        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,\n"
        f"        validation_data=(X_val, y_val), verbose=1,\n"
        f"    )\n"
        f"    _, val_acc = model.evaluate(X_val, y_val, verbose=0)\n"
        f"    metrics = {{\n"
        f'        "success": True,\n'
        f'        "model_type": "cnn_2d",\n'
        f'        "train_loss": float(history.history["loss"][-1]),\n'
        f'        "val_loss": float(history.history["val_loss"][-1]),\n'
        f'        "val_auc": float(val_acc),\n'
        f'        "epochs_completed": EPOCHS,\n'
        f'        "runtime_seconds": float(time.time() - start_time),\n'
        f"    }}\n"
        f"except Exception as e:\n"
        f"    metrics = {{\n"
        f'        "success": False,\n'
        f'        "error_type": type(e).__name__,\n'
        f'        "error_message": str(e),\n'
        f"    }}\n"
        f"with open(METRICS_PATH, 'w') as f:\n"
        f"    json.dump(metrics, f, indent=2)\n"
    )
