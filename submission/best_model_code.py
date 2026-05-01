EXPERIMENT_META = {
    "model_type": "cnn",
    "architecture": "4x Conv2D + GAP + Dropout",
    "change": "added more Conv2D layers",
    "key_params": {"lr": 1e-3, "batch_size": 32, "epochs": 5},
}


def get_training_config():
    return {
        "max_samples": 1500,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "val_split": 0.2,
    }


def build_model(input_shape, num_classes):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

# --- FINAL OVERRIDE: force final training config ---
def get_training_config():
    return {'max_samples': None, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 12, 'batch_size': 32, 'learning_rate': 0.001, 'optimizer': 'adam', 'val_split': 0.0, 'weight_decay': 0.0001, 'classifier_hidden_units': 256, 'pooling_type': 'global_avg', 'use_best_checkpoint': True}
