EXPERIMENT_META = {
    "model_type": "pretrained",
    "architecture": "MobileNetV2 frozen + GAP + BatchNorm",
    "change": "added_batch_normalization",
    "key_params": {"lr": 3e-4, "batch_size": 64, "epochs": 15, "image_size": 96},
}


def get_training_config():
    return {
        "max_samples": 2000,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 15,
        "batch_size": 64,
        "learning_rate": 3e-4,
    }


def build_model(input_shape, num_classes):
    import tensorflow as tf
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Resizing(96, 96)(inputs)
    x = tf.keras.layers.Concatenate()([x, x, x])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Added batch normalization
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="sigmoid",
        kernel_regularizer=reg,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model

# --- FINAL OVERRIDE: force final training config ---
def get_training_config():
    return {'max_samples': None, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 12, 'batch_size': 64, 'learning_rate': 0.0003, 'val_split': 0.0}
