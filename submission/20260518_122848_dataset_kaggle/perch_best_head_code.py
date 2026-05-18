def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x = tf.keras.layers.Dense(1024, activation="gelu")(inp)
    
    for _ in range(3):
        h = tf.keras.layers.Dense(1024)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("gelu")(h)
        h = tf.keras.layers.Dropout(0.3)(h)
        
        v = tf.keras.layers.Dense(1024, activation="linear")(h)
        g = tf.keras.layers.Dense(1024, activation="sigmoid")(h)
        gated = tf.keras.layers.Multiply()([v, g])
        x = tf.keras.layers.Add()([x, gated])
    
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 0.001,
        "batch_size": 256,
        "optimizer": "adamw",
        "epochs": 40,
        "patience": 7,
        "perch_weight": 0.3
    }