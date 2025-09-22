import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model

from setuptools import setup

setup(
    name='digit_classifier',  
    version='0.1.0',
    install_requires=[
        'tensorflow==2.15.0',
        'matplotlib',
        'numpy',
    ],
)
# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = build_model()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(x_train, y_train, epochs=2, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
