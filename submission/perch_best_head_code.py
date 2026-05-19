def build_head(emb_dim, num_classes):
    hidden_dim = 768
    proj_dim = 512
    dropout = 0.35
    
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(inp)
    
    for _ in range(3):
        h = tf.keras.layers.Dense(hidden_dim)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("relu")(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        h = tf.keras.layers.Dense(hidden_dim)(h)
        x = tf.keras.layers.Add()([x, h])
    
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 0.0012,
        "batch_size": 256,
        "optimizer": "adamw",
        "epochs": 40,
        "patience": 7,
        "perch_weight": 0.3
    }