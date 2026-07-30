"""
auto_label.py
Auto-generates YOLO bounding box labels using color-based detection.
Best for objects with a distinct color (like your orange charger)
against a fairly consistent background.

Usage:
    python auto_label.py --img_dir dataset/positives_raw
"""
import cv2, os, argparse

def main(img_dir):
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved, skipped = 0, 0

    for fname in files:
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Orange/red range in HSV -- adjust if detection misses your object
        lower1 = (0, 100, 100)
        upper1 = (15, 255, 255)
        lower2 = (165, 100, 100)
        upper2 = (180, 255, 255)

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (9, 9))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            skipped += 1
            continue

        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 500:  # too small, probably noise
            skipped += 1
            continue

        x, y, bw, bh = cv2.boundingRect(biggest)

        # Convert to YOLO format: class x_center y_center width height (normalized)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        label_path = os.path.join(img_dir, os.path.splitext(fname)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        saved += 1

    print(f"Auto-labeled: {saved}, Skipped (no detection): {skipped}")
    print("IMPORTANT: open these in LabelImg afterward and spot-check / fix boxes before training.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="dataset/positives_raw")
    args = p.parse_args()
    main(args.img_dir)