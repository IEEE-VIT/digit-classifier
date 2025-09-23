import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def build_model():
    """Builds a simple feed-forward neural network for MNIST classification."""
    model = models.Sequential([
        layers.Input(shape=(28, 28)),     # Input shape for MNIST digits
        layers.Flatten(),                 # Flatten 28x28 images into vectors
        layers.Dense(256, activation='relu'),  # Increased units for better learning
        layers.Dropout(0.3),              # Added dropout for regularization
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    return model


def train_model():
    """Loads MNIST dataset, compiles and trains the model."""
    # Load dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    train_model()
