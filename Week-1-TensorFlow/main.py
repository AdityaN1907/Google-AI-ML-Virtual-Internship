import tensorflow as tf

print("=" * 50)
print("Google AI-ML Virtual Internship")
print("Week 1 - TensorFlow Fundamentals")
print("=" * 50)

print("TensorFlow Version:", tf.__version__)

hello = tf.constant("Hello TensorFlow!")
print("Message:", hello.numpy().decode())

a = tf.constant(10)
b = tf.constant(20)

print("Addition:", tf.add(a, b).numpy())
print("Multiplication:", tf.multiply(a, b).numpy())

matrix = tf.constant([[1, 2], [3, 4]])
print("\nMatrix:")
print(matrix)

print("\nMatrix Shape:", matrix.shape)
print("Data Type:", matrix.dtype)