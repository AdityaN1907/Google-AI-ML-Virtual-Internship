import tensorflow as tf

print("=" * 50)
print("Week 1 - Tensor Basics")
print("=" * 50)

# Scalar
scalar = tf.constant(10)
print("\nScalar:")
print(scalar)
print("Shape:", scalar.shape)

# Vector
vector = tf.constant([10, 20, 30])
print("\nVector:")
print(vector)
print("Shape:", vector.shape)

# Matrix
matrix = tf.constant([[1, 2], [3, 4]])
print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)

# 3D Tensor
tensor = tf.constant([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("\n3D Tensor:")
print(tensor)
print("Shape:", tensor.shape)

# Reshape
reshaped = tf.reshape(vector, (3, 1))
print("\nReshaped Vector:")
print(reshaped)