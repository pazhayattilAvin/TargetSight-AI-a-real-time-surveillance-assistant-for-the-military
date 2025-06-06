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

hostile_targets = ['person']

print("[INFO] TargetSight AI running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    person_count = 0
    alert_triggered = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            if label.lower() == 'person':
                person_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Trigger voice alerts
    if person_count == 1 and not alert_triggered:
        speak("Unidentified person detected. Identify yourself.")
        alert_triggered = True
    elif person_count > 1 and not alert_triggered:
        speak("Emergency! Multiple unidentified persons detected. Initiate security protocol.")
        alert_triggered = True

    # Show frame
    cv2.imshow("TargetSight - Emergency Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
