import cv2
import mediapipe as mp
import numpy as np

registered_encoding = np.load("data/face_encoding.npy")
registered_name = np.load("data/face_name.npy")

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

print("Verification Started")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "No Face"
    color = (0,0,255)

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

        # Compare encoding
        distance = np.linalg.norm(registered_encoding - encoding)

        if distance < 1.0:
            status = f"Authorized: {registered_name}"
            color = (0,255,0)
        else:
            status = "ALERT: Unknown Person"
            color = (0,0,255)

        # Calculate bounding box
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)

        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    cv2.putText(frame, status, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
