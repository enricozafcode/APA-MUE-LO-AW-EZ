def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x = tf.keras.layers.Dense(1024, activation="gelu")(inp)  # Dense stem

    # Transformer block
    x_3d = tf.keras.layers.Reshape((1, 1024))(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=1024 // 4)(x_3d, x_3d)
    attn = tf.keras.layers.Reshape((1024,))(attn)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    ff = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation="gelu"),
        tf.keras.layers.Dense(1024)
    ])
    x = tf.keras.layers.Add()([x, ff(x)])
    x = tf.keras.layers.LayerNormalization()(x)

    # Output layer
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 0.0008,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 40,
        "patience": 7,
        "perch_weight": 0.2
    }