"""
pick_and_place.py

Detect -> APPROACH -> ANGLE ALIGN -> FINAL DESCENT -> BLIND CREEP -> GRAB.

WHY THIS CHANGED (v2):
The camera is mounted ABOVE the gripper, not co-located with it. That means
the pixel offset the camera sees does NOT correspond to where the gripper
actually needs to be -- there's a parallax/offset error between "where the
camera thinks the object is centered" and "where the gripper needs to
approach from." Previously the arm drove straight at the camera-detected
center and grabbed, which put the gripper in the wrong position/orientation
because it approached from the camera's viewpoint, not its own.

Fix: this version adds a camera->gripper offset compensation step, and
splits the final approach into two distinct new phases:

  1. ANGLE ALIGN  - back off slightly (up/back), rotate the wrist so the
                     gripper's approach vector points at the object instead
                     of the camera's, while continuing to track/hold the
                     object's position with small corrective moves.
  2. FINAL DESCENT - once the gripper orientation is corrected, move in
                      slowly along the corrected approach vector using the
                      camera-to-gripper compensated target, with small
                      live corrections, instead of the old single blind
                      creep straight from the raw APPROACH phase.

BLIND CREEP is kept, but now only runs for a short final push AFTER angle
correction + final descent, once the gripper is already lined up correctly
-- not as a substitute for correcting the approach angle.

State machine:
  DETECT -> LOCKING -> APPROACH -> ANGLE_ALIGN -> FINAL_DESCENT -> BLIND_CREEP -> GRAB
"""
import cv2, time, os, json
from collections import deque
from ultralytics import YOLO
from arm_control import send_command, gripper_open, gripper_close

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "..", "runs", "detect", "train", "weights", "best.pt")
CALIB_PATH = os.path.join(SCRIPT_DIR, "grab_calibration.json")

CAMERA_INDEX = 1
CONF = 0.5

HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_WRIST = 0.0, 0.0, 2.1, 0.0

SMOOTH_WINDOW = 7
LOCK_DEADBAND_PX = 15
LOCK_TOLERANCE_PX = 35
LOCK_STEP_RAD = 0.015
LOCK_CONSECUTIVE_NEEDED = 6

SHOULDER_REACH_STEP_RAD = 0.018
SHOULDER_FINE_STEP_RAD = 0.008
ELBOW_TRIM_GAIN = 0.0004
ELBOW_TRIM_MAX_RAD = 0.02

AREA_SMOOTH_WINDOW = 5
AREA_CONSECUTIVE_NEEDED = 5

APPROACH_DRIFT_TOLERANCE_PX = 60
APPROACH_DRIFT_STEP_RAD = 0.010

DEFAULT_NEAR_AREA_RATIO = 0.35
DEFAULT_FINE_APPROACH_AREA_RATIO = 0.80
DEFAULT_GRAB_AREA_RATIO = 0.95   # once confirmed here, switch to ANGLE_ALIGN

# --- CAMERA -> GRIPPER OFFSET COMPENSATION (the core fix) ---
# The camera sits above/behind the gripper's own centerline, so the raw
# pixel offset it reports is systematically biased relative to where the
# gripper itself needs to be. These are constant compensation terms added
# to the raw pixel offsets before they're used for any positioning
# decision, so every phase after detection works off a "gripper's-eye-
# view" estimate instead of the raw camera view.
#
# Tune by jogging the arm so the gripper is physically touching/centered
# on a test object, then reading what offset_x/offset_y the camera reports
# at that moment -- that reported value (negated) is your offset.
CAMERA_TO_GRIPPER_OFFSET_X_PX = 0     # camera-right positive; set from calibration
CAMERA_TO_GRIPPER_OFFSET_Y_PX = -40   # camera is above gripper -> object appears
                                       # higher in-frame than gripper's true target
CAMERA_TO_GRIPPER_OFFSET_AREA = 0.0   # optional area_ratio bias correction, usually 0

# --- ANGLE ALIGNMENT PHASE (new) ---
# Once APPROACH confirms we're close (grab_streak reached), don't grab yet.
# Instead back the arm up/away slightly and rotate the wrist so the
# gripper's own approach vector is pointed at the object, while small
# corrective moves keep the object centered under the new viewpoint.
ANGLE_ALIGN_LIFT_SHOULDER_RAD = 0.08   # configurable "move up/back" distance
ANGLE_ALIGN_LIFT_ELBOW_RAD = 0.05
ANGLE_ALIGN_WRIST_ROT_RAD = 0.35       # configurable wrist/TCP rotation target
ANGLE_ALIGN_STEPS = 6                  # frames to settle + correct during align
ANGLE_ALIGN_CORRECTION_GAIN_RAD_PER_PX = 0.0006
ANGLE_ALIGN_CORRECTION_MAX_RAD = 0.03
ANGLE_ALIGN_SETTLE_DELAY = 0.5

# --- FINAL DESCENT PHASE (new) ---
# Slow, corrected move back in toward the object along the now-corrected
# approach vector, using the compensated (camera->gripper offset applied)
# target instead of the raw camera center.
FINAL_DESCENT_STEPS = 6
FINAL_DESCENT_STEP_RAD = 0.012
FINAL_DESCENT_CORRECTION_GAIN_RAD_PER_PX = 0.0005
FINAL_DESCENT_CORRECTION_MAX_RAD = 0.02
FINAL_DESCENT_SETTLE_DELAY = 0.4

# --- FINAL BLIND CREEP ---
# Short push with no vision feedback at all, used only once the gripper
# is already angle-corrected and lined up from FINAL_DESCENT -- not as a
# substitute for alignment like before.
BLIND_CREEP_STEPS = 3
BLIND_CREEP_STEP_RAD = 0.006

JOG_STEP = 0.02

BASE_MIN, BASE_MAX = HOME_BASE - 0.6, HOME_BASE + 0.6
SHOULDER_MIN, SHOULDER_MAX = HOME_SHOULDER - 0.5, HOME_SHOULDER + 1.5
ELBOW_MIN, ELBOW_MAX = HOME_ELBOW - 0.5, HOME_ELBOW + 1.2
WRIST_MIN, WRIST_MAX = -1.57, 1.57

MAX_LOST_FRAMES = 8
LOOP_DELAY = 0.15
SETTLE_DELAY = 0.6

SPD = 3
ACC = 4


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def go_to_pose(base, shoulder, elbow, wrist=HOME_WRIST, hand=3.14):
    base = clamp(base, BASE_MIN, BASE_MAX)
    shoulder = clamp(shoulder, SHOULDER_MIN, SHOULDER_MAX)
    elbow = clamp(elbow, ELBOW_MIN, ELBOW_MAX)
    wrist = clamp(wrist, WRIST_MIN, WRIST_MAX)
    try:
        send_command({"T": 102, "base": base, "shoulder": shoulder, "elbow": elbow,
                       "wrist": wrist, "hand": hand, "spd": SPD, "acc": ACC})
    except Exception as e:
        print(f"  [send_command error] {e}")
    return base, shoulder, elbow, wrist


def detect(model, cap):
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    results = model(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()
    h, w = frame.shape[:2]
    cv2.drawMarker(annotated, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.imshow("Pick", annotated)
    cv2.waitKey(1)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, None, frame

    box = boxes[0].xyxy[0].cpu().numpy()
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    area_ratio = area / (h * w)
    return (cx, cy), area_ratio, frame


def compensate_for_gripper(raw_offset_x, raw_offset_y, raw_area_ratio):
    """
    Convert camera-frame detection into the gripper's target frame.
    This is step 2 of the required behavior: camera coords -> gripper
    target coords, using a fixed camera-to-gripper calibration offset.
    """
    gripper_offset_x = raw_offset_x + CAMERA_TO_GRIPPER_OFFSET_X_PX
    gripper_offset_y = raw_offset_y + CAMERA_TO_GRIPPER_OFFSET_Y_PX
    gripper_area_ratio = raw_area_ratio + CAMERA_TO_GRIPPER_OFFSET_AREA
    return gripper_offset_x, gripper_offset_y, gripper_area_ratio


def load_calibration():
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH) as f:
                data = json.load(f)
            print(f"Loaded existing calibration: {data}")
            return data
        except Exception as e:
            print(f"Could not read calibration file: {e}")
    return None


def save_calibration(data):
    try:
        with open(CALIB_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration: {data}")
    except Exception as e:
        print(f"Could not save calibration file: {e}")


def run_calibration(model, cap):
    print("=" * 70)
    print("CALIBRATION MODE -- jog to the pick point, press c to capture.")
    print("=" * 70)
    base, shoulder, elbow, wrist = go_to_pose(HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_WRIST)
    gripper_open()
    step = JOG_STEP
    area_hist = deque(maxlen=AREA_SMOOTH_WINDOW)

    while True:
        center, area_ratio, frame = detect(model, cap)
        if area_ratio is not None:
            area_hist.append(area_ratio)
        smoothed_area = sum(area_hist) / len(area_hist) if area_hist else None
        disp = f"{smoothed_area:.3f}" if smoothed_area is not None else "no detection"
        print(f"\rsmoothed_area={disp}  base={base:.3f} shoulder={shoulder:.3f} "
              f"elbow={elbow:.3f} wrist={wrist:.3f} step={step:.3f}   ", end="")

        key = cv2.waitKey(50) & 0xFF
        moved = False
        if key == ord('w'): elbow += step; moved = True
        elif key == ord('x'): elbow -= step; moved = True
        elif key == ord('a'): shoulder -= step; moved = True
        elif key == ord('d'): shoulder += step; moved = True
        elif key == ord('j'): base -= step; moved = True
        elif key == ord('l'): base += step; moved = True
        elif key == ord('u'): wrist -= step; moved = True
        elif key == ord('o'): wrist += step; moved = True
        elif key == ord('['): step = max(0.005, step - 0.005); print(f"\nstep={step:.3f}")
        elif key == ord(']'): step += 0.005; print(f"\nstep={step:.3f}")
        elif key == ord('c'):
            if smoothed_area is None:
                print("\n  No detection -- keep jogging."); continue
            print(f"\n*** CAPTURED: {smoothed_area:.3f} ***")
            calib = {"grab_area_ratio": round(smoothed_area, 4),
                      "fine_approach_area_ratio": round(smoothed_area * 0.85, 4),
                      "near_area_ratio": round(smoothed_area * 0.45, 4)}
            save_calibration(calib)
            return calib
        elif key == ord('s'):
            print("\nSkipping calibration."); return None

        if moved:
            base, shoulder, elbow, wrist = go_to_pose(base, shoulder, elbow, wrist)
            area_hist.clear()
            time.sleep(0.15)


def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    calib = load_calibration()
    if calib is None:
        if input("No calibration found. Run calibration now? (y/n): ").strip().lower() == "y":
            calib = run_calibration(model, cap)
    else:
        if input("Calibration found. Use it (u) or recalibrate (r)? ").strip().lower() == "r":
            calib = run_calibration(model, cap)

    if calib:
        NEAR_AREA_RATIO = calib["near_area_ratio"]
        FINE_APPROACH_AREA_RATIO = calib["fine_approach_area_ratio"]
        GRAB_AREA_RATIO = calib["grab_area_ratio"]
    else:
        NEAR_AREA_RATIO = DEFAULT_NEAR_AREA_RATIO
        FINE_APPROACH_AREA_RATIO = DEFAULT_FINE_APPROACH_AREA_RATIO
        GRAB_AREA_RATIO = DEFAULT_GRAB_AREA_RATIO

    print(f"Thresholds: near={NEAR_AREA_RATIO:.3f} fine={FINE_APPROACH_AREA_RATIO:.3f} "
          f"grab={GRAB_AREA_RATIO:.3f}  blind_creep_steps={BLIND_CREEP_STEPS}")
    print(f"Camera->gripper offset: x={CAMERA_TO_GRIPPER_OFFSET_X_PX}px "
          f"y={CAMERA_TO_GRIPPER_OFFSET_Y_PX}px area={CAMERA_TO_GRIPPER_OFFSET_AREA}")

    base, shoulder, elbow, wrist = HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_WRIST

    try:
        print("Moving home, gripper closed...")
        base, shoulder, elbow, wrist = go_to_pose(base, shoulder, elbow, wrist)
        gripper_close()
        time.sleep(2)

        offset_x_hist = deque(maxlen=SMOOTH_WINDOW)
        offset_y_hist = deque(maxlen=SMOOTH_WINDOW)
        area_hist = deque(maxlen=AREA_SMOOTH_WINDOW)

        lost_frames = 0
        consecutive_centered = 0
        gripper_is_open = False
        near_streak = 0
        fine_streak = 0
        grab_streak = 0

        # --- explicit state machine ---
        # LOCKING -> APPROACH -> ANGLE_ALIGN -> FINAL_DESCENT -> BLIND_CREEP -> DONE
        phase = "LOCKING"
        align_steps_done = 0
        descent_steps_done = 0
        blind_steps_done = 0

        print("PHASE 1/6: LOCKING. Press 'q' to quit.")
        while True:
            center, raw_area, frame = detect(model, cap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit by user.")
                cap.release(); cv2.destroyAllWindows(); return

            if center is None and phase not in ("ANGLE_ALIGN", "FINAL_DESCENT", "BLIND_CREEP"):
                lost_frames += 1
                print(f"No object ({lost_frames}/{MAX_LOST_FRAMES})")
                if lost_frames >= MAX_LOST_FRAMES:
                    base, shoulder, elbow, wrist = go_to_pose(HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_WRIST)
                    offset_x_hist.clear(); offset_y_hist.clear(); area_hist.clear()
                    consecutive_centered = near_streak = fine_streak = grab_streak = 0
                    phase = "LOCKING"
                    lost_frames = 0
                    time.sleep(0.6)
                time.sleep(LOOP_DELAY)
                continue

            if center is not None:
                lost_frames = 0
                h, w = frame.shape[:2]
                raw_offset_x = center[0] - w / 2
                raw_offset_y = center[1] - h / 2
                offset_x_hist.append(raw_offset_x)
                offset_y_hist.append(raw_offset_y)
                smoothed_raw_x = sum(offset_x_hist) / len(offset_x_hist)
                smoothed_raw_y = sum(offset_y_hist) / len(offset_y_hist)
                area_hist.append(raw_area)
                raw_area_ratio = sum(area_hist) / len(area_hist)
            else:
                smoothed_raw_x = smoothed_raw_y = 0
                raw_area_ratio = raw_area = 0

            # step 2 of required behavior: camera coords -> gripper target coords
            offset_x, offset_y, area_ratio = compensate_for_gripper(
                smoothed_raw_x, smoothed_raw_y, raw_area_ratio)

            # ---------------- LOCKING ----------------
            if phase == "LOCKING":
                centered = abs(offset_x) <= LOCK_TOLERANCE_PX and abs(offset_y) <= LOCK_TOLERANCE_PX
                print(f"[LOCKING] gripper-frame=({offset_x:.0f},{offset_y:.0f}) "
                      f"streak={consecutive_centered}/{LOCK_CONSECUTIVE_NEEDED}")
                if centered:
                    consecutive_centered += 1
                    if consecutive_centered >= LOCK_CONSECUTIVE_NEEDED:
                        print("*** LOCKED. PHASE 2/6: APPROACH. ***")
                        phase = "APPROACH"
                        offset_x_hist.clear(); offset_y_hist.clear(); area_hist.clear()
                        time.sleep(0.3)
                    else:
                        time.sleep(LOOP_DELAY)
                    continue
                else:
                    consecutive_centered = 0

                moved = False
                if abs(offset_x) > LOCK_DEADBAND_PX:
                    base += -LOCK_STEP_RAD if offset_x > 0 else LOCK_STEP_RAD
                    moved = True
                if abs(offset_y) > LOCK_DEADBAND_PX:
                    step = LOCK_STEP_RAD if offset_y > 0 else -LOCK_STEP_RAD
                    shoulder += step; elbow += step
                    moved = True
                if moved:
                    base, shoulder, elbow, wrist = go_to_pose(base, shoulder, elbow, wrist)
                    offset_x_hist.clear(); offset_y_hist.clear()
                    time.sleep(SETTLE_DELAY)
                else:
                    time.sleep(LOOP_DELAY)

            # ---------------- APPROACH ----------------
            elif phase == "APPROACH":
                near_streak = near_streak + 1 if area_ratio >= NEAR_AREA_RATIO else 0
                fine_streak = fine_streak + 1 if area_ratio >= FINE_APPROACH_AREA_RATIO else 0
                grab_streak = grab_streak + 1 if area_ratio >= GRAB_AREA_RATIO else 0

                print(f"[APPROACH] gripper-frame=({offset_x:.0f},{offset_y:.0f}) "
                      f"raw_area={raw_area:.3f} comp_area={area_ratio:.3f} "
                      f"near={near_streak} fine={fine_streak} grab={grab_streak} "
                      f"(shoulder={shoulder:.3f} elbow={elbow:.3f})")

                if not gripper_is_open and near_streak >= AREA_CONSECUTIVE_NEEDED:
                    print("  Opening gripper.")
                    gripper_open()
                    gripper_is_open = True

                if grab_streak >= AREA_CONSECUTIVE_NEEDED:
                    print(f"Confirmed close (comp_area={area_ratio:.3f}). "
                          f"Not grabbing yet -- PHASE 3/6: ANGLE ALIGN.")
                    phase = "ANGLE_ALIGN"
                    align_steps_done = 0
                    continue

                if abs(offset_x) > APPROACH_DRIFT_TOLERANCE_PX:
                    base += -APPROACH_DRIFT_STEP_RAD if offset_x > 0 else APPROACH_DRIFT_STEP_RAD

                if fine_streak >= AREA_CONSECUTIVE_NEEDED:
                    shoulder += SHOULDER_FINE_STEP_RAD
                else:
                    shoulder += SHOULDER_REACH_STEP_RAD

                elbow_trim = clamp(ELBOW_TRIM_GAIN * offset_y, -ELBOW_TRIM_MAX_RAD, ELBOW_TRIM_MAX_RAD)
                elbow += elbow_trim

                base, shoulder, elbow, wrist = go_to_pose(base, shoulder, elbow, wrist)
                time.sleep(SETTLE_DELAY)

            # ---------------- ANGLE ALIGN (new) ----------------
            elif phase == "ANGLE_ALIGN":
                # Step 5: move slightly up/back, rotate wrist so the gripper
                # points at the object, while small corrections keep the
                # object roughly centered under the new viewpoint.
                align_steps_done += 1
                target_shoulder = shoulder - (ANGLE_ALIGN_LIFT_SHOULDER_RAD / ANGLE_ALIGN_STEPS)
                target_elbow = elbow - (ANGLE_ALIGN_LIFT_ELBOW_RAD / ANGLE_ALIGN_STEPS)
                target_wrist = wrist + (ANGLE_ALIGN_WRIST_ROT_RAD / ANGLE_ALIGN_STEPS)

                x_corr = clamp(ANGLE_ALIGN_CORRECTION_GAIN_RAD_PER_PX * offset_x,
                                -ANGLE_ALIGN_CORRECTION_MAX_RAD, ANGLE_ALIGN_CORRECTION_MAX_RAD)
                y_corr = clamp(ANGLE_ALIGN_CORRECTION_GAIN_RAD_PER_PX * offset_y,
                                -ANGLE_ALIGN_CORRECTION_MAX_RAD, ANGLE_ALIGN_CORRECTION_MAX_RAD)
                target_base = base + (-x_corr if offset_x else 0)
                target_elbow += (y_corr if center else 0)

                print(f"[ANGLE ALIGN] step {align_steps_done}/{ANGLE_ALIGN_STEPS} "
                      f"target(base={target_base:.3f} shoulder={target_shoulder:.3f} "
                      f"elbow={target_elbow:.3f} wrist={target_wrist:.3f}) "
                      f"gripper-frame=({offset_x:.0f},{offset_y:.0f})")

                base, shoulder, elbow, wrist = go_to_pose(target_base, target_shoulder,
                                                            target_elbow, target_wrist)
                time.sleep(ANGLE_ALIGN_SETTLE_DELAY)

                if align_steps_done >= ANGLE_ALIGN_STEPS:
                    print(f"Angle alignment complete (wrist={wrist:.3f}). "
                          f"PHASE 4/6: FINAL DESCENT.")
                    phase = "FINAL_DESCENT"
                    descent_steps_done = 0

            # ---------------- FINAL DESCENT (new) ----------------
            elif phase == "FINAL_DESCENT":
                # Step 6: slow, corrected move back in along the now-
                # corrected approach vector, using the compensated target.
                descent_steps_done += 1
                x_corr = clamp(FINAL_DESCENT_CORRECTION_GAIN_RAD_PER_PX * offset_x,
                                -FINAL_DESCENT_CORRECTION_MAX_RAD, FINAL_DESCENT_CORRECTION_MAX_RAD)
                y_corr = clamp(FINAL_DESCENT_CORRECTION_GAIN_RAD_PER_PX * offset_y,
                                -FINAL_DESCENT_CORRECTION_MAX_RAD, FINAL_DESCENT_CORRECTION_MAX_RAD)

                target_base = base + (-x_corr if offset_x else 0)
                target_shoulder = shoulder + FINAL_DESCENT_STEP_RAD
                target_elbow = elbow + FINAL_DESCENT_STEP_RAD + (y_corr if center else 0)

                print(f"[FINAL DESCENT] step {descent_steps_done}/{FINAL_DESCENT_STEPS} "
                      f"target(base={target_base:.3f} shoulder={target_shoulder:.3f} "
                      f"elbow={target_elbow:.3f} wrist={wrist:.3f}) "
                      f"gripper-frame=({offset_x:.0f},{offset_y:.0f}) comp_area={area_ratio:.3f}")

                base, shoulder, elbow, wrist = go_to_pose(target_base, target_shoulder,
                                                            target_elbow, wrist)
                time.sleep(FINAL_DESCENT_SETTLE_DELAY)

                if descent_steps_done >= FINAL_DESCENT_STEPS:
                    print("Final descent complete. PHASE 5/6: BLIND CREEP.")
                    phase = "BLIND_CREEP"
                    blind_steps_done = 0

            # ---------------- BLIND CREEP (short, angle-corrected) ----------------
            elif phase == "BLIND_CREEP":
                # Step 7: only runs now that the gripper angle is already
                # corrected -- this is a short final push, not a substitute
                # for alignment like in the old logic.
                blind_steps_done += 1
                print(f"[BLIND CREEP] step {blind_steps_done}/{BLIND_CREEP_STEPS} "
                      f"(vision ignored) shoulder={shoulder:.3f} wrist={wrist:.3f}")
                if blind_steps_done >= BLIND_CREEP_STEPS:
                    print("Blind creep complete. PHASE 6/6: GRAB.")
                    break

                shoulder += BLIND_CREEP_STEP_RAD
                base, shoulder, elbow, wrist = go_to_pose(base, shoulder, elbow, wrist)
                time.sleep(SETTLE_DELAY)

        if not gripper_is_open:
            print("Opening gripper before grab (safety net)...")
            gripper_open()
            time.sleep(0.8)

        print("GRABBING: closing gripper...")
        gripper_close()
        time.sleep(1)
        # Step 8: confirm successful grasp using the object's area reading
        # one more time -- if it's gone (dropped/missed), area collapses.
        confirm_center, confirm_area, confirm_frame = detect(model, cap)
        if confirm_area is not None and confirm_area > 0.02:
            print(f"Grasp check: object still visible near gripper (area={confirm_area:.3f}) "
                  f"-- likely grabbed OK, but this is only a coarse visual check.")
        else:
            print("Grasp check: object no longer clearly visible -- verify grasp manually.")

        print("Lifting back to home pose...")
        go_to_pose(HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_WRIST)
        time.sleep(1.5)
        print("Pick sequence complete.")

    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()