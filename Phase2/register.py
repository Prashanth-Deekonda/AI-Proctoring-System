import cv2
import mediapipe as mp
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")

name = input("Enter student name: ")

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

print("Press 's' to capture")

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = []

        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
            landmarks.append(lm.z)

        encoding = np.array(landmarks)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            np.save("data/face_encoding.npy", encoding)
            np.save("data/face_name.npy", name)
            print("Face Registered Successfully")
            break

    cv2.imshow("Register Face", frame)

cap.release()
cv2.destroyAllWindows()
