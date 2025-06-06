from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # "n" = nano, lightweight and fast

# Load image
image_path = "test.jpg"  # Replace with your image file
if not os.path.exists(image_path):
    print(f"[ERROR] Image '{image_path}' not found.")
    exit()

# Run detection
results = model(image_path)

# Visualize results on the image
for result in results:
    result.show()

# Optional: Save the result image
results[0].save(filename="output.jpg")
print("[INFO] Detection complete. Saved as 'output.jpg'")
