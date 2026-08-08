import cv2
import numpy as np
import os

IMAGE_DIR = "images"
QUERY_IMAGE = "images/shoe1.jpg"
TOP_K = 3


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize for consistent comparison
    image = cv2.resize(image, (128, 128))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color histogram
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(histogram, histogram)

    return histogram.flatten()


def compare_images(query_features, image_features):
    # Histogram correlation
    return cv2.compareHist(
        query_features.astype(np.float32),
        image_features.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


print("=" * 60)
print("Week 4 - Product Image Search")
print("=" * 60)

print("\nExtracting product features...")

query_features = extract_features(QUERY_IMAGE)

if query_features is None:
    print("Error: Query image could not be loaded.")
    exit()

results = []

for filename in os.listdir(IMAGE_DIR):

    image_path = os.path.join(IMAGE_DIR, filename)

    # Skip query image
    if os.path.abspath(image_path) == os.path.abspath(QUERY_IMAGE):
        continue
    features = extract_features(image_path)

    if features is not None:
        similarity = compare_images(query_features, features)

        results.append((filename, similarity))


# Highest similarity first
results.sort(key=lambda x: x[1], reverse=True)

print("\nQuery Image:")
print(os.path.basename(QUERY_IMAGE))

print("\nTop Similar Products:")

for rank, (filename, similarity) in enumerate(results[:TOP_K], start=1):

    print(
        f"{rank}. {filename} "
        f"- Similarity: {similarity:.4f}"
    )

print("\nProduct image search completed!")