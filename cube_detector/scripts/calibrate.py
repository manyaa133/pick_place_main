"""
calibrate.py
Standalone: just detects the object and prints its area_ratio every
frame. No arm movement at all. Use this to find the area_ratio value
that corresponds to "gripper is close enough to grab."
"""
import cv2
from ultralytics import YOLO

WEIGHTS = "runs/detect/train/weights/best.pt"
CAMERA_INDEX = 1
CONF = 0.5

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Showing live area_ratio. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    results = model(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Calibrate", annotated)

    boxes = results[0].boxes
    if len(boxes) > 0:
        box = boxes[0].xyxy[0].cpu().numpy()
        box_w = float(box[2] - box[0])
        box_h = float(box[3] - box[1])
        h, w = frame.shape[:2]
        area_ratio = (box_w * box_h) / (w * h)
        print(f"area_ratio = {area_ratio:.3f}")
    else:
        print("no detection")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()