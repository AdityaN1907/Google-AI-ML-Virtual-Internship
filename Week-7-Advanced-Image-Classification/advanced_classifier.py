import tensorflow as tf
import numpy as np
import os

print("=" * 60)
print("Week 7 - Advanced Image Classification")
print("=" * 60)

print("\nLoading Fashion-MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Lightweight dataset
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("\nBuilding advanced CNN model...")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    data_augmentation,

    tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Architecture:")
model.summary()

print("\nTraining advanced model...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    batch_size=64
)

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nPredictions:")

predictions = model.predict(x_test[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

for i in range(10):
    actual = class_names[int(y_test[i])]
    predicted = class_names[predicted_classes[i]]

    print(
        f"{i + 1}. Predicted: {predicted:<12} "
        f"Actual: {actual}"
    )

# Save model
os.makedirs("outputs", exist_ok=True)

model.save("outputs/advanced_fashion_classifier.keras")

print("\nAdvanced model saved successfully!")
print("Week 7 Advanced Image Classification completed!")