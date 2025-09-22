import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


history = model.fit(x_train, y_train, epochs=2, validation_split=0.1)
model.save('mnist_model.keras')
from tensorflow.keras.models import load_model
model = load_model('mnist_model.keras')