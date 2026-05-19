def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x = tf.keras.layers.Dense(1024, activation="gelu")(inp)  # Dense stem

    for _ in range(4):  # Four residual blocks
        h = tf.keras.layers.Dense(1024)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("gelu")(h)
        h = tf.keras.layers.Dropout(0.3)(h)
        h = tf.keras.layers.Dense(1024)(h)
        x = tf.keras.layers.Add()([x, h])

    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

def _ORIG_GET_TRAINING_CONFIG():
    return {
        "learning_rate": 0.001,
        "batch_size": 256,
        "optimizer": "adamw",
        "perch_weight": 0.3,
        "epochs": 3,
        "patience": 3
    }

# --- PERCH TRAINING CAPS ---
_PERCH_TRAINING_CAPS = {'epochs': 3, 'patience': 3}

def get_training_config():
    cfg = _ORIG_GET_TRAINING_CONFIG()
    cfg.update(_PERCH_TRAINING_CAPS)
    return cfg
