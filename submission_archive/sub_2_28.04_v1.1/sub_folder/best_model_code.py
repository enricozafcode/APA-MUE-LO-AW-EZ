def get_training_config():
    return {
        "max_samples": None,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 128,
        "n_frames": 192,
        "epochs": 20,
        "batch_size": 8,
        "learning_rate": 0.0030336987857336314,
        "optimizer": "adam",
    }
 
 
def build_model(input_shape, num_classes):
    import tensorflow as tf
    filters_list = [48, 48, 48, 48, 48]
    max_pools = 5
    use_batch_norm = True
    use_residuals = False
    dropout_rate = 0.09
 
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate(filters_list):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if i < max_pools:
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        if use_residuals and i > 0:
            sc_filters = tf.keras.backend.int_shape(x)[-1]
            shortcut = tf.keras.layers.Conv2D(sc_filters, (1, 1), padding="same")(shortcut)
            if i < max_pools:
                shortcut = tf.keras.layers.MaxPooling2D((2, 2))(shortcut)
            if tf.keras.backend.int_shape(x) == tf.keras.backend.int_shape(shortcut):
                x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, x)
    return model
