import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension (required for Conv2D + ImageDataGenerator)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build model
model = build_model()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Data augmentation generator
# Applies small, realistic transformations so the model doesn't memorize
# exact pixel positions and generalizes better to unseen digits.
datagen = ImageDataGenerator(
    rotation_range=10,        # slight rotations (degrees)
    width_shift_range=0.1,    # horizontal shift
    height_shift_range=0.1,   # vertical shift
    zoom_range=0.1,           # random zoom
    shear_range=0.1,          # slight shear
    validation_split=0.1      # reserve 10% of training data for validation
)

# Fit the generator on training data (needed for feature-wise stats, harmless otherwise)
datagen.fit(x_train)

batch_size = 64

train_generator = datagen.flow(
    x_train, y_train,
    batch_size=batch_size,
    subset="training"
)

validation_generator = datagen.flow(
    x_train, y_train,
    batch_size=batch_size,
    subset="validation"
)

# Train using streamed augmented batches instead of static arrays
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=2
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot accuracy
# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save both plots as a single image
plt.savefig("training_history.png", dpi=300)

# Show the graph
plt.show()