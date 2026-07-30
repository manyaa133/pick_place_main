"""
list_cameras.py
Cycles through camera indices 0-4 and shows each one for 2 seconds
so you can identify which index is your Logitech cam.
"""
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        cap.release()
        continue

    print(f"Index {i}: opened. Showing preview for 2 seconds...")
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera index {i}", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print(f"Index {i}: opened but couldn't read a frame")

    cap.release()

print("Done. Note which index showed your Logitech cam.")