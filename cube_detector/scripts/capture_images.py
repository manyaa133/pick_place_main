"""
capture_images.py
Captures frames from the camera for YOLO training.
Press 'c' to save a frame, 'q' to quit.

Usage:
    python capture_images.py --save_dir dataset/positives_raw
    python capture_images.py --save_dir dataset/negatives_raw
"""
import cv2, os, argparse

def main(camera_index, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    count = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print(f"Starting count: {count}. Press 'c' to capture, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, f"Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            fname = f"img_{count:03d}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), frame)
            print(f"Saved {fname}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {count} images in {save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="dataset/positives_raw")
    args = p.parse_args()
    main(args.camera, args.save_dir)