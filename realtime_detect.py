from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (Nano = fast & light)
model = YOLO("yolov8n.pt")

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam")
    exit()

print("[INFO] Starting real-time detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, stream=True)

    # Draw results on the frame
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display result
    cv2.imshow("TargetSight - Live Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
