import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

IMAGE_DIR = "images"
QUERY_IMAGE = "images/shoe1.jpg"
TOP_K = 3


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    # Convert image to HSV
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

    # Flatten histogram into feature vector
    return histogram.flatten()


print("=" * 60)
print("Week 5 - Advanced Product Image Search")
print("=" * 60)

print("\nExtracting features from product images...")

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

        similarity = cosine_similarity(
            [query_features],
            [features]
        )[0][0]

        results.append((filename, similarity))


# Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

print("\nQuery Image:")
print(os.path.basename(QUERY_IMAGE))

print("\nTop Similar Products:")

for rank, (filename, similarity) in enumerate(
    results[:TOP_K], start=1
):
    print(
        f"{rank}. {filename} "
        f"- Similarity: {similarity:.4f}"
    )

print("\nAdvanced product search completed!")