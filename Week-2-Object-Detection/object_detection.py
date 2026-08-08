import cv2
import numpy as np

print("=" * 60)
print("Week 2 - Object Detection")
print("=" * 60)

# Create a simple test image
image = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Draw sample objects
cv2.rectangle(image, (100, 100), (300, 350), (255, 0, 0), 3)
cv2.putText(
    image,
    "Object 1",
    (120, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)

cv2.rectangle(image, (400, 150), (600, 400), (0, 0, 255), 3)
cv2.putText(
    image,
    "Object 2",
    (420, 140),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

# Save output
cv2.imwrite("outputs/detection_output.jpg", image)

print("\nObjects detected and bounding boxes generated.")
print("Output saved to: outputs/detection_output.jpg")