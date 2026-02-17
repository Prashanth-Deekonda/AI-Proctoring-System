import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("YOLO Detection Started... Press Q to exit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, stream=True, verbose=False)

    person_count = 0
    phone_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # Ignore low confidence detections
            if conf < 0.6:
                continue

            if label == "person":
                person_count += 1

            if label == "cell phone":
                phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0,255,0)

            if label == "cell phone":
                color = (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if person_count > 1:
        cv2.putText(frame, "ALERT: Multiple Persons Detected!",
                    (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if phone_detected:
        cv2.putText(frame, "ALERT: Phone Detected!",
                    (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.putText(frame, f"Persons: {person_count}",
                (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("YOLO Proctoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
