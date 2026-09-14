import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Fit the generator on training data
datagen.fit(x_train.reshape(-1, 28, 28, 1))

# Build model
model = build_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train using augmented images
history = model.fit(
    datagen.flow(
        x_train.reshape(-1, 28, 28, 1),
        y_train,
        batch_size=32
    ),
    epochs=2,
    steps_per_epoch=len(x_train) // 32,
    validation_data=(x_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Create two subplots
plt.figure(figsize=(8, 10))

# Plot accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig("accuracy.png", dpi=300)

# Show graph
plt.show()