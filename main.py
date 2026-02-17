import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Load registered face encoding
registered_encoding = np.load("data/face_encoding.npy")
registered_name = np.load("data/face_name.npy")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

print("AI Proctoring System Started... Press Q to exit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w, c = frame.shape

    ####################################
    # FACE VERIFICATION
    ####################################

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    face_status = "No Face Detected"
    face_color = (0,0,255)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        landmarks = []
        x_list = []
        y_list = []

        for lm in face_landmarks.landmark:

            x = int(lm.x * w)
            y = int(lm.y * h)

            x_list.append(x)
            y_list.append(y)

            landmarks.append(lm.x)
            landmarks.append(lm.y)
            landmarks.append(lm.z)

        encoding = np.array(landmarks)

        distance = np.linalg.norm(registered_encoding - encoding)

        if distance < 1.0:
            face_status = f"Authorized: {registered_name}"
            face_color = (0,255,0)
        else:
            face_status = "ALERT: Unknown Person"
            face_color = (0,0,255)

        # Draw face box
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)

        cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), face_color, 2)

    ####################################
    # YOLO OBJECT DETECTION
    ####################################

    results = yolo_model(frame, stream=True, verbose=False)

    person_count = 0
    phone_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]

            if conf < 0.6:
                continue

            if label == "person":
                person_count += 1

            if label == "cell phone":
                phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (255,255,0)

            if label == "cell phone":
                color = (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ####################################
    # ALERT CONDITIONS
    ####################################

    alert_y = 50

    cv2.putText(frame, face_status, (20,alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)

    alert_y += 40

    if person_count > 1:
        cv2.putText(frame, "ALERT: Multiple Persons Detected!",
                    (20,alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        alert_y += 40

    if phone_detected:
        cv2.putText(frame, "ALERT: Phone Detected!",
                    (20,alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        alert_y += 40

    cv2.putText(frame, f"Person Count: {person_count}",
                (20,alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    ####################################

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
