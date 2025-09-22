# train.py
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model

# -----------------------
# Config
# -----------------------
SAVED_MODEL_DIR = "saved_models"
BEST_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "best_model.h5")  # HDF5 checkpoint for best weights
FINAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "final_saved_model")  # SavedModel directory
EPOCHS = 10
BATCH_SIZE = 64

# ensure save directory exists
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# -----------------------
# Load dataset (MNIST)
# -----------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalize and add channel if necessary
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# If model expects a channel dimension, ensure shape (N, 28, 28)
# (the provided model uses layers.Input(shape=(28, 28)) so no channel needed)

# -----------------------
# Build & compile model
# -----------------------
model = build_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Callbacks: Save best model automatically
# -----------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",        # save best by validation accuracy
    verbose=1,
    save_best_only=True,           # only keep the best
    mode="max"
)

# optional: early stopping to avoid overfitting (uncomment to use)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------
# Train
# -----------------------
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[checkpoint_cb],   # add earlystop_cb here if you want it
    verbose=1
)

# -----------------------
# Evaluate
# -----------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# -----------------------
# Save the final model (SavedModel format)
# -----------------------
model.save(FINAL_MODEL_PATH, save_format="tf")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
print(f"Best checkpoint (by val_accuracy) saved to: {BEST_MODEL_PATH}")

# -----------------------
# Plot accuracy
# -----------------------
plt.plot(history.history.get('accuracy', []), label='train acc')
plt.plot(history.history.get('val_accuracy', []), label='val acc')
plt.legend()
plt.title("Training / Validation Accuracy")
plt.show()