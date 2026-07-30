"""
detect_store_pick_place.py

Now corrects BOTH axes:
  - offset_x -> base   (was missing before -- this was why centering felt broken)
  - offset_y -> elbow
Both signs are auto-calibrated from real camera feedback, not guessed.
Center position is smoothed (EMA) to reduce jitter-driven false "on target" flicker.
"""
import cv2, time, json, os
from datetime import datetime
from ultralytics import YOLO
from arm_control import send_command, gripper_open, gripper_close

WEIGHTS = "../runs/detect/train/weights/best.pt"
CAMERA_INDEX = 1
CONF = 0.5

HOME_BASE, HOME_SHOULDER, HOME_ELBOW = 0.0, 0.0, 2.1
PLACE_BASE, PLACE_SHOULDER, PLACE_ELBOW = 1.2, 0.3, 1.8

CALIBRATION_STEP = 0.05
SHOULDER_STEP = 0.03
BASE_CORRECT_STEP = 0.015
ELBOW_CORRECT_STEP = 0.015

CENTER_TOLERANCE_PX = 30      # loosened slightly -- 25 was tight enough to fight jitter
EDGE_MARGIN_PX = 70
CLOSE_ENOUGH_AREA = 0.60
MAX_LOST_FRAMES = 5
EMA_ALPHA = 0.4               # smoothing factor for detected center (reduces jitter)

SAVE_DIR = "captures"
POSITIONS_FILE = "stored_positions.json"
STORE_POSITION_T = 110  # confirm against your firmware's docs


def go_to_pose(base, shoulder, elbow, hand=3.14):
    cmd = {"T": 102, "base": base, "shoulder": shoulder, "elbow": elbow,
           "hand": hand, "spd": 15, "acc": 10}
    try:
        resp = send_command(cmd)
    except Exception as e:
        print("  [send_command error]", e)
        return None
    return resp


class Smoother:
    def __init__(self):
        self.cx = None
        self.cy = None

    def update(self, cx, cy):
        if self.cx is None:
            self.cx, self.cy = cx, cy
        else:
            self.cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * self.cx
            self.cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * self.cy
        return self.cx, self.cy


def detect(model, cap, smoother):
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    results = model(frame, conf=CONF, verbose=False)
    cv2.imshow("Detect", results[0].plot())
    cv2.waitKey(1)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, None, frame
    box = boxes[0].xyxy[0].cpu().numpy()
    raw_cx = (box[0] + box[2]) / 2
    raw_cy = (box[1] + box[3]) / 2
    cx, cy = smoother.update(raw_cx, raw_cy)
    area = (box[2] - box[0]) * (box[3] - box[1])
    h, w = frame.shape[:2]
    area_ratio = area / (h * w)
    return (cx, cy), area_ratio, frame


def wait_for_detection(model, cap, smoother, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        center, area_ratio, frame = detect(model, cap, smoother)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, None, None
        if center is not None:
            return center, area_ratio, frame
        time.sleep(0.2)
    return None, None, None


def calibrate_axis_sign(model, cap, smoother, base, shoulder, elbow, axis):
    """axis: 'base' (checks offset_x) or 'elbow' (checks offset_y)."""
    print(f"Calibrating {axis} direction...")
    center0, area0, frame0 = wait_for_detection(model, cap, smoother)
    if center0 is None:
        return 1
    h0, w0 = frame0.shape[:2]
    offset0 = (center0[0] - w0 / 2) if axis == "base" else (center0[1] - h0 / 2)

    if axis == "base":
        go_to_pose(base + CALIBRATION_STEP, shoulder, elbow)
    else:
        go_to_pose(base, shoulder, elbow + CALIBRATION_STEP)
    time.sleep(0.6)

    center1, area1, frame1 = wait_for_detection(model, cap, smoother)
    go_to_pose(base, shoulder, elbow)
    time.sleep(0.6)

    if center1 is None:
        return -1
    h1, w1 = frame1.shape[:2]
    offset1 = (center1[0] - w1 / 2) if axis == "base" else (center1[1] - h1 / 2)

    return 1 if abs(offset1) < abs(offset0) else -1


def calibrate_shoulder_sign(model, cap, smoother, base, shoulder, elbow):
    print("Calibrating shoulder direction...")
    _, area0, _ = wait_for_detection(model, cap, smoother)
    if area0 is None:
        return 1
    go_to_pose(base, shoulder + CALIBRATION_STEP, elbow)
    time.sleep(0.6)
    _, area1, _ = wait_for_detection(model, cap, smoother)
    go_to_pose(base, shoulder, elbow)
    time.sleep(0.6)
    if area1 is None:
        return -1
    return 1 if area1 > area0 else -1


def save_capture_and_position(frame, base, shoulder, elbow, center, area_ratio):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(SAVE_DIR, f"object_{ts}.jpg")
    cv2.imwrite(img_path, frame)

    record = {
        "timestamp": ts,
        "image_path": img_path,
        "pixel_center": {"cx": center[0], "cy": center[1]},
        "area_ratio": area_ratio,
        "arm_pose": {"base": base, "shoulder": shoulder, "elbow": elbow}
    }

    data = []
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(record)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved image to {img_path} and logged pose to {POSITIONS_FILE}")

    store_cmd = {"T": STORE_POSITION_T, "base": base, "shoulder": shoulder, "elbow": elbow}
    try:
        resp = send_command(store_cmd)
        print(f"  -> sent store command to ESP32 | response: {resp}")
    except Exception as e:
        print(f"  [store command error] {e} -- position is still safe in {POSITIONS_FILE}")

    return record


def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    base, shoulder, elbow = HOME_BASE, HOME_SHOULDER, HOME_ELBOW
    smoother = Smoother()

    try:
        print("Moving to home position...")
        go_to_pose(base, shoulder, elbow)
        gripper_open()
        time.sleep(2)

        print("Waiting for object to be visible before calibrating...")
        center, area_ratio, frame = wait_for_detection(model, cap, smoother)
        if center is None:
            print("No object detected within timeout. Aborting.")
            return

        base_sign = calibrate_axis_sign(model, cap, smoother, base, shoulder, elbow, "base")
        elbow_sign = calibrate_axis_sign(model, cap, smoother, base, shoulder, elbow, "elbow")
        shoulder_sign = calibrate_shoulder_sign(model, cap, smoother, base, shoulder, elbow)
        print(f"Calibration done. base_sign={base_sign} elbow_sign={elbow_sign} shoulder_sign={shoulder_sign}")
        print("Approaching object. Press q to stop.")

        lost_frames = 0
        last_good_pose = (base, shoulder, elbow)
        final_record = None

        while True:
            center, area_ratio, frame = detect(model, cap, smoother)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                cap.release(); cv2.destroyAllWindows(); return

            if center is None:
                lost_frames += 1
                print(f"No object detected ({lost_frames}/{MAX_LOST_FRAMES})...")
                if lost_frames >= MAX_LOST_FRAMES:
                    base, shoulder, elbow = last_good_pose
                    go_to_pose(base, shoulder, elbow)
                    time.sleep(0.6)
                    lost_frames = 0
                time.sleep(0.3)
                continue

            lost_frames = 0
            last_good_pose = (base, shoulder, elbow)

            h, w = frame.shape[:2]
            offset_x = center[0] - w / 2
            offset_y = center[1] - h / 2
            near_top = center[1] < EDGE_MARGIN_PX
            near_bottom = center[1] > h - EDGE_MARGIN_PX
            on_target_x = abs(offset_x) < CENTER_TOLERANCE_PX
            on_target_y = abs(offset_y) < CENTER_TOLERANCE_PX
            on_target = on_target_x and on_target_y

            print(f"area_ratio={area_ratio:.3f} offset_x={offset_x:.1f} offset_y={offset_y:.1f} on_target={on_target}")

            if area_ratio >= CLOSE_ENOUGH_AREA and on_target:
                print(f"Close enough and centered (area_ratio={area_ratio:.3f}). Capturing + storing position...")
                final_record = save_capture_and_position(frame, base, shoulder, elbow, center, area_ratio)
                break

            # Correct BOTH axes before allowing a forward step
            if not on_target_x:
                if offset_x > CENTER_TOLERANCE_PX:
                    base += base_sign * BASE_CORRECT_STEP
                else:
                    base -= base_sign * BASE_CORRECT_STEP

            if not on_target_y or near_top or near_bottom:
                if offset_y > CENTER_TOLERANCE_PX:
                    elbow += elbow_sign * ELBOW_CORRECT_STEP
                else:
                    elbow -= elbow_sign * ELBOW_CORRECT_STEP

            if not on_target_x or not on_target_y or near_top or near_bottom:
                go_to_pose(base, shoulder, elbow)
                time.sleep(0.3)
                continue

            shoulder += shoulder_sign * SHOULDER_STEP
            go_to_pose(base, shoulder, elbow)
            time.sleep(0.3)

        if final_record is None:
            print("Never reached a stored position -- stopping before pick.")
            return

        stored = final_record["arm_pose"]
        print(f"Moving to stored position: {stored}")
        go_to_pose(stored["base"], stored["shoulder"], stored["elbow"])
        time.sleep(1.5)

        print("Closing gripper to grab object...")
        gripper_close()
        time.sleep(1)

        print("Moving to place position...")
        go_to_pose(PLACE_BASE, PLACE_SHOULDER, PLACE_ELBOW)
        time.sleep(2)

        print("Opening gripper to release object...")
        gripper_open()
        time.sleep(1)

        print("Returning to home position...")
        go_to_pose(HOME_BASE, HOME_SHOULDER, HOME_ELBOW)
        time.sleep(1.5)
        print("Done.")

    except Exception as e:
        print("UNEXPECTED ERROR:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()