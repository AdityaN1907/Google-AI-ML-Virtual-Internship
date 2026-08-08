from ultralytics import YOLO
import cv2
import os

print("=" * 60)
print("Week 2 - AI Object Detection")
print("=" * 60)

# Load pretrained YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolo11n.pt")

# Input image
image_path = "images/bus.jpg"

print("\nRunning object detection...")
results = model(image_path, conf=0.25)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Process results
for result in results:
    # Print detected objects
    print("\nDetected Objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"- {class_name}: {confidence * 100:.2f}%")

    # Draw bounding boxes
    annotated_image = result.plot()

    # Save result
    output_path = "outputs/yolo_detection.jpg"
    cv2.imwrite(output_path, annotated_image)

print("\nObject detection completed!")
print("Output saved to: outputs/yolo_detection.jpg")