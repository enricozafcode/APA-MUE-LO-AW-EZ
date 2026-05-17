def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(inp)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 20,
        "patience": 5,
        "perch_weight": 0.2,
    }