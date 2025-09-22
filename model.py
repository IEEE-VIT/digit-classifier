import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
 
    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # Helps prevent overfitting
        layers.Dense(10, activation='softmax')
    ])
    return model