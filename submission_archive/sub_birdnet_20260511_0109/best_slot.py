import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

EXPERIMENT_META = {
    'head_type': 'keras_mlp',
    'architecture': 'Dense(512) + Dropout(0.3) + weighted BCE',
    'change': 'baseline',
    'key_params': {'lr': 1e-3, 'batch_size': 256, 'epochs': 30},
}

def get_head_config():
    return {'epochs': 30, 'batch_size': 256, 'learning_rate': 1e-3}

def build_head(emb_dim, num_classes, y_train):
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
    pw = tf.constant(pos_weight[None, :], dtype=tf.float32)
    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        return tf.reduce_mean(pw*y_true*(-tf.math.log(y_pred)) + (1-y_true)*(-tf.math.log(1-y_pred)))
    inputs = tf.keras.Input(shape=(emb_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_bce, metrics=['accuracy'])
    model.callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')]
    return model