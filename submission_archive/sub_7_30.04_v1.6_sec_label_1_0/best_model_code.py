def get_training_config():
    return {
        "max_samples": 2000,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.00017776545083782858,
        "optimizer": "adam",
        "val_split": 0.2,
        "weight_decay": 1.0645987236440721e-05,
        "classifier_hidden_units": 0,
        "pooling_type": "global_avg",
    }
 
 
def build_model(input_shape, num_classes):
    import tensorflow as tf
    filters_list = [32, 64, 128, 256, 512, 512, 512, 512]
    max_pools = 5
    use_batch_norm = True
    use_residuals = True
    dropout_rate = 0.0
    weight_decay = 1.0645987236440721e-05
    classifier_hidden_units = 0
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
    return {'max_samples': 10000, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.00017776545083782858, 'optimizer': 'adam', 'val_split': 0.2, 'weight_decay': 1.0645987236440721e-05, 'classifier_hidden_units': 0, 'pooling_type': 'global_avg', 'use_best_checkpoint': True}


# --- FINAL OVERRIDE: force final training config ---
def get_training_config():
    return {'max_samples': None, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 13, 'batch_size': 32, 'learning_rate': 0.00017776545083782858, 'optimizer': 'adam', 'val_split': 0.0, 'weight_decay': 1.0645987236440721e-05, 'classifier_hidden_units': 0, 'pooling_type': 'global_avg', 'use_best_checkpoint': True}

def build_features(audio_path, sample_rate, clip_seconds, n_mels, n_frames):
    import numpy as np
    import librosa
    import tensorflow as tf

    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy().astype(np.float32)



# --- FINAL OVERRIDE: force final training config ---
def get_training_config():
    return {'max_samples': None, 'sample_rate': 32000, 'clip_seconds': 5.0, 'n_mels': 64, 'n_frames': 128, 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.00017776545083782858, 'optimizer': 'adam', 'val_split': 0.0, 'weight_decay': 1.0645987236440721e-05, 'classifier_hidden_units': 0, 'pooling_type': 'global_avg', 'use_best_checkpoint': True, 'aug_probability': 0.6, 'aug_noise_std': 0.015, 'aug_time_shift_max_fraction': 0.1, 'aug_time_mask_max_fraction': 0.08, 'aug_freq_mask_max_fraction': 0.08, 'mixup_alpha': 0.0, 'secondary_label_weight': 1.0}
