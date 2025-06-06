# TargetSight AI - Real-Time Threat Detection System

A real-time AI-powered surveillance tool for the military, capable of identifying and alerting on **unidentified threats** like persons, vehicles, tanks, or drones using computer vision and voice alerts.

##  Features
-  Real-time object detection using **YOLOv8**
-  Identifies multiple persons and triggers **voice warnings**
-  Emergency alerts for suspicious activity
-  Voice notifications (e.g. "Unidentified person detected. Identify yourself.")
-  Works with live **webcam** or camera feed


##  Tech Stack
- Python
- OpenCV
- YOLOv8 (`ultralytics`)
- pyttsx3 (for offline voice synthesis)

## Run the App
bash
Copy code
python scripts/realtime_with_all_alerts.py

## Alert Behavior
- 1 Person: "Unidentified person detected. Identify yourself."

- 2+ People: "Emergency! Multiple unidentified persons detected."

- Vehicle/Drones: "Unidentified truck/airplane/boat detected."

## Sample Output (Screenshots)


## License
-This project is 100% free for educational and personal use.

## Connect with Me
LinkedIn: www.linkedin.com/in/pazhayattilavin
