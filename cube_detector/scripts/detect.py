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
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    # CLAMP to actual frame bounds -- a box that YOLO reports as extending
    # past the frame edge (common when the object/claw is very close and
    # partially out of view) would otherwise inflate area beyond what's
    # really visible. This is what was likely causing area_ratio to hit
    # a false 1.000 instead of reflecting real distance.
    x1c = max(0, min(x1, w))
    y1c = max(0, min(y1, h))
    x2c = max(0, min(x2, w))
    y2c = max(0, min(y2, h))

    cx = (x1 + x2) / 2   # center uses the RAW (unclamped) box -- still accurate
    cy = (y1 + y2) / 2   # for tracking position even near the edge

    area = (x2c - x1c) * (y2c - y1c)
    area_ratio = area / (h * w)

    was_clipped = (x1 < 0 or y1 < 0 or x2 > w or y2 > h)
    print(f"  [DEBUG] raw_box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) frame={w}x{h} "
          f"clipped={was_clipped} area_ratio={area_ratio:.3f}")

    return (cx, cy), area_ratio, frame