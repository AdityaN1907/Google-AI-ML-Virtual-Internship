import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Week 1 - MNIST Neural Network Classifier")
print("=" * 60)

# --------------------------------------------------
# 1. Load MNIST Dataset
# --------------------------------------------------

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# --------------------------------------------------
# 2. Normalize pixel values
# --------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nPixel values normalized to range 0-1")

# --------------------------------------------------
# 3. Build Neural Network
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

print("\nModel Architecture:")
model.summary()

# --------------------------------------------------
# 4. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------

print("\nTraining model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# --------------------------------------------------
# 6. Evaluate Model
# --------------------------------------------------

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

predictions = model.predict(x_test[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nPredictions:")
print("Predicted:", predicted_labels)
print("Actual:   ", y_test[:10])

# --------------------------------------------------
# 8. Save Model
# --------------------------------------------------

model.save("../Week-1-TensorFlow/models/mnist_model.keras")

print("\nModel saved successfully!")

# --------------------------------------------------
# 9. Display Sample Prediction
# --------------------------------------------------

plt.imshow(x_test[0], cmap="gray")
plt.title(
    f"Predicted: {predicted_labels[0]} | Actual: {y_test[0]}"
)
plt.axis("off")

plt.savefig("../Week-1-TensorFlow/outputs/mnist_prediction.png")

print("Prediction image saved to outputs/")
print("\nWeek 1 MNIST Classifier completed!")