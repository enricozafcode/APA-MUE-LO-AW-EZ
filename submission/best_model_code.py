def get_training_config():
    return {
        "max_samples": 2000,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "optimizer": "adam",
        "val_split": 0.2,
        "weight_decay": 0.0001,
        "classifier_hidden_units": 512,
        "pooling_type": "global_avg",
        "aug_prob": 0.0,
        "aug_noise_std": 0.0,
        "aug_time_mask": 0,
        "aug_freq_mask": 0,
    }


def build_model(input_shape, num_classes):
    import tensorflow as tf
    filters_list = [64]
    max_pools = 1
    use_batch_norm = True
    use_residuals = True
    dropout_rate = 0.0
    weight_decay = 0.0001
    classifier_hidden_units = 512
    pooling_type = "global_avg"
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate(filters_list):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=reg)(x)
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
    if pooling_type == "global_avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling_type == "global_max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.Flatten()(x)
    if classifier_hidden_units and classifier_hidden_units > 0:
        x = tf.keras.layers.Dense(classifier_hidden_units, activation="relu", kernel_regularizer=reg)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inputs, x)
    return model


# --- FINAL OVERRIDE: force final training config ---
def get_training_config():
    return {'max_samples': None, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0005, 'optimizer': 'adam', 'val_split': 0.0, 'weight_decay': 0.0001, 'classifier_hidden_units': 512, 'pooling_type': 'global_avg', 'use_best_checkpoint': True}
