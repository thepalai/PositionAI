import cv2
import numpy as np
import time

# ── Cameras ────────────────────────────────────────────────────────────────
cap1 = cv2.VideoCapture(1)   # iPhone
cap2 = cv2.VideoCapture(2)   # MacBook webcam

for cap, name in [(cap1, "iPhone (1)"), (cap2, "MacBook (2)")]:
    if not cap.isOpened():
        print(f"ERROR: Cannot open {name}")
        exit()

# ── ArUco setup ────────────────────────────────────────────────────────────
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

TARGET_ID = 0   # Plate A marker

# ── Colors ─────────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,   255, 0)
YELLOW = (0,   255, 255)
RED    = (0,   0,   255)
GREY   = (160, 160, 160)
CYAN   = (255, 255, 0)

# ── Windows ────────────────────────────────────────────────────────────────
cv2.namedWindow("FEEDS - move to iPad", cv2.WINDOW_NORMAL)
cv2.namedWindow("STATUS - keep on MacBook", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FEEDS - move to iPad", 1280, 900)
cv2.resizeWindow("STATUS - keep on MacBook", 900, 400)

# ── Stable detection counter ───────────────────────────────────────────────
stable_count = 0
STABLE_NEEDED = 30   # ~1 second of both cameras detecting

print("Running. Hold marker ID 0 in front of both cameras. Q = quit.")

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        time.sleep(0.05)
        continue

    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]

    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # ── Detect in both feeds ───────────────────────────────────────────────
    corners1, ids1, _ = detector.detectMarkers(gray1)
    corners2, ids2, _ = detector.detectMarkers(gray2)

    def find_marker(corners, ids, target_id):
        """Return corner array for target_id or None."""
        if ids is None:
            return None
        for i, mid in enumerate(ids.flatten()):
            if mid == target_id:
                return corners[i][0]   # shape (4, 2)
        return None

    m1 = find_marker(corners1, ids1, TARGET_ID)
    m2 = find_marker(corners2, ids2, TARGET_ID)

    # ── Draw all detected markers on feeds ─────────────────────────────────
    def draw_markers(frame, corners, ids, target_id):
        if ids is None:
            return
        for i, mid in enumerate(ids.flatten()):
            pts = corners[i][0].astype(int)
            color = GREEN if mid == target_id else GREY
            cv2.polylines(frame, [pts], True, color, 3)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, f"ID:{mid}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 5)
            cv2.putText(frame, f"ID:{mid}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            # Draw center dot
            cv2.circle(frame, (cx, cy), 10, color, -1)

    draw_markers(f1, corners1, ids1, TARGET_ID)
    draw_markers(f2, corners2, ids2, TARGET_ID)

    # ── Stability logic ────────────────────────────────────────────────────
    both_detected = (m1 is not None and m2 is not None)
    if both_detected:
        stable_count = min(stable_count + 1, STABLE_NEEDED)
    else:
        stable_count = max(stable_count - 2, 0)   # decay faster than grow

    # ── Print stable detection info to terminal ────────────────────────────
    if both_detected:
        cx1 = int(m1[:, 0].mean())
        cy1 = int(m1[:, 1].mean())
        cx2 = int(m2[:, 0].mean())
        cy2 = int(m2[:, 1].mean())
        print(f"CAM1 center=({cx1},{cy1})  CAM2 center=({cx2},{cy2})  "
              f"stable={stable_count}/{STABLE_NEEDED}", end="\r")

    # ── Build feeds window (top/bottom) ────────────────────────────────────
    TARGET_W = 1280
    scale1 = TARGET_W / w1
    scale2 = TARGET_W / w2
    disp1 = cv2.resize(f1, (TARGET_W, int(h1 * scale1)))
    disp2 = cv2.resize(f2, (TARGET_W, int(h2 * scale2)))

    for disp, label in [(disp1, "CAM 1 - iPhone"), (disp2, "CAM 2 - MacBook")]:
        cv2.putText(disp, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK,  5)
        cv2.putText(disp, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 2)

    feeds_window = np.vstack([disp1, disp2])
    cv2.imshow("FEEDS - move to iPad", feeds_window)

    # ── Build status window ────────────────────────────────────────────────
    SW, SH = 900, 400
    status = np.zeros((SH, SW, 3), dtype=np.uint8)
    status[:] = (15, 15, 15)

    def huge(text, y, color, scale=2.0, thick=4):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = (SW - tw) // 2
        cv2.putText(status, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 6)
        cv2.putText(status, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def medium(text, y, color=WHITE, scale=1.0, thick=2):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = (SW - tw) // 2
        cv2.putText(status, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 3)
        cv2.putText(status, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    cam1_txt = "CAM1: DETECTED" if m1 is not None else "CAM1: not detected"
    cam2_txt = "CAM2: DETECTED" if m2 is not None else "CAM2: not detected"
    cam1_col = GREEN if m1 is not None else RED
    cam2_col = GREEN if m2 is not None else RED

    medium(cam1_txt, 80,  cam1_col, scale=1.2)
    medium(cam2_txt, 130, cam2_col, scale=1.2)

    # Progress bar
    bar_x0, bar_x1 = 60, SW - 60
    bar_y0, bar_y1 = 160, 195
    bar_fill = int((stable_count / STABLE_NEEDED) * (bar_x1 - bar_x0))
    bar_col  = GREEN if stable_count >= STABLE_NEEDED else YELLOW
    cv2.rectangle(status, (bar_x0, bar_y0), (bar_x1, bar_y1), (50, 50, 50), -1)
    if bar_fill > 0:
        cv2.rectangle(status, (bar_x0, bar_y0),
                      (bar_x0 + bar_fill, bar_y1), bar_col, -1)
    medium(f"Stable: {stable_count}/{STABLE_NEEDED}", 240, bar_col, scale=1.1)

    if stable_count >= STABLE_NEEDED:
        huge("MARKER DETECTED OK", 330, GREEN, scale=1.8)
    elif both_detected:
        huge("Hold still...", 330, YELLOW, scale=1.8)
    elif m1 is None and m2 is None:
        huge("Show marker to both cameras", 330, GREY, scale=1.2)
    elif m1 is None:
        huge("CAM1 cannot see marker", 330, RED, scale=1.4)
    else:
        huge("CAM2 cannot see marker", 330, RED, scale=1.4)

    cv2.imshow("STATUS - keep on MacBook", status)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("\nDone.")