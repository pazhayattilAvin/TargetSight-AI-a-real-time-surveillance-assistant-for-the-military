from ultralytics import YOLO
import cv2
import pyttsx3

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam")
    exit()

# Define list of tracked hostile targets
hostile_targets = ['person', 'truck', 'airplane', 'boat']  # You can add: 'car', 'motorcycle', 'train', 'bus', 'drone'

print("[INFO] TargetSight AI running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    label_count = {}  # To count occurrences of each label
    alert_triggered = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Count each type of label
            if label.lower() in hostile_targets:
                label_count[label.lower()] = label_count.get(label.lower(), 0) + 1

    # Trigger custom alerts
    for target, count in label_count.items():
        if count == 1:
            speak(f"Unidentified {target} detected. Identify yourself.")
        elif count > 1:
            speak(f"Emergency! Multiple unidentified {target}s detected. Initiate security protocol.")
        alert_triggered = True

    # Show frame
    cv2.imshow("TargetSight - Multi Target Alert Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
