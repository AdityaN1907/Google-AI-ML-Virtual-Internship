from ultralytics import YOLO
import cv2
import os

print("=" * 60)
print("Week 3 - Advanced Object Detection")
print("=" * 60)

# Load pretrained YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolo11n.pt")

input_video = "videos/zidane.mp4"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

print("\nStarting object tracking...")

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = os.path.join(output_dir, "tracked_output.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Object tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        verbose=False
    )

    # Draw detections and tracking IDs
    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    frame_count += 1

cap.release()
writer.release()

print(f"\nProcessed frames: {frame_count}")
print("Object tracking completed!")
print(f"Output saved to: {output_path}")