import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)

def get_training_config():
    """
    Returns the training configuration parameters for the BirdCLEF task.
    """
    return {
        "max_samples": 500,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }


def build_model(input_shape, num_classes):
    """
    Builds the CNN model for multi-label bird species classification.

    Args:
        input_shape (tuple): Shape of the input mel spectrogram (n_mels, n_frames, 1).
        num_classes (int): Number of target species (234).

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    model = Sequential([
        # Block 1
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classifier Head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        
        # Output Layer: Sigmoid for multi-label classification
        Dense(num_classes, activation='sigmoid')
    ])
    
    # Compile the model with required loss and optimizer
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model