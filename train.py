import tensorflow as tf
from data.load_data import load_mnist
from model import build_model
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = load_mnist()
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
