import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def build_model():
    """Builds a CNN for MNIST classification."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),                     # 28x28 grayscale images (1 channel)
        layers.Conv2D(32, (3, 3), activation='relu'),         # Learn 32 low-level features (edges, curves)
        layers.MaxPooling2D((2, 2)),                          # Downsample, keep strongest features
        layers.Conv2D(64, (3, 3), activation='relu'),         # Learn 64 higher-level features
        layers.MaxPooling2D((2, 2)),                          # Downsample again
        layers.Flatten(),                                     # NOW flatten, after spatial features extracted
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')                # Output layer for 10 classes
    ])
    return model


def train_model():
    """Loads MNIST dataset, compiles and trains the model."""
    # Load dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)   # add channel dimension
    x_test = x_test.reshape(-1, 28, 28, 1)

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
