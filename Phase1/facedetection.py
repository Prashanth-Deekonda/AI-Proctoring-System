import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# IMPORTANT CHANGE HERE
face_detection = mp_face.FaceDetection(
    model_selection=1,  # works better for multiple faces
    min_detection_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    face_count = 0

    if results.detections:
        face_count = len(results.detections)

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, c = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Show face count
    cv2.putText(frame, f'Faces detected: {face_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Cheating warning
    if face_count > 1:
        cv2.putText(frame, 'WARNING: Multiple Faces Detected!',
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
