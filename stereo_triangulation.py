import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request
from collections import deque

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading pose_landmarker_heavy.task …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ── Calibration ────────────────────────────────────────────────────────────
CALIB_FILE = "stereo_calib_ipad.npz"
if not os.path.exists(CALIB_FILE):
    print(f"ERROR: {CALIB_FILE} not found. Copy it to the working directory.")
    exit()

data = np.load(CALIB_FILE)
K1, D1 = data["K1"], data["D1"]   # Camera 1 — iPhone (index 1)
K2, D2 = data["K2"], data["D2"]   # Camera 2 — MacBook (index 2)
R,  T  = data["R"],  data["T"]

# Projection matrices
P1 = K1 @ np.hstack([np.eye(3),    np.zeros((3, 1))])   # Camera 1 = world origin
P2 = K2 @ np.hstack([R,             T              ])    # Camera 2 relative to Camera 1

# ── MediaPipe ──────────────────────────────────────────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def make_detector():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(opts)

det1 = make_detector()
det2 = make_detector()

# ── Cameras ────────────────────────────────────────────────────────────────
cap1 = cv2.VideoCapture(1)   # iPhone
cap2 = cv2.VideoCapture(2)   # MacBook webcam

for cap, name in [(cap1, "iPhone (1)"), (cap2, "MacBook (2)")]:
    if not cap.isOpened():
        print(f"ERROR: Cannot open {name}")
        exit()

# ── Constants ──────────────────────────────────────────────────────────────
VIS_HIP   = 0.5
VIS_KNEE  = 0.3
VIS_THRESH = 0.3   # used for drawing dots/lines
SMOOTH_N     = 30          # rolling average window

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)
RED    = (0, 0, 255)
GREY   = (160, 160, 160)
BLACK  = (0, 0, 0)
CYAN   = (255, 255, 0)

# Landmark indices
L_HIP, R_HIP     = 23, 24
L_KNEE, R_KNEE   = 25, 26

# ── Helpers ────────────────────────────────────────────────────────────────
def undistort_pt(px, py, K, D):
    """Undistort a single pixel point. Returns undistorted pixel coords."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.undistortPoints(pt, K, D, P=K)   # P=K -> stays in pixel space
    return out[0, 0]   # (x, y) pixels, distortion removed

def triangulate(u1, v1, K1, D1, u2, v2, K2, D2, P1, P2):
    """
    Undistort two pixel points, triangulate, return (X, Y, Z) in mm.
    """
    n1 = undistort_pt(u1, v1, K1, D1)
    n2 = undistort_pt(u2, v2, K2, D2)

    pt4d = cv2.triangulatePoints(P1, P2,
                                  n1.reshape(2, 1),
                                  n2.reshape(2, 1))
    pt3d = pt4d[:3] / pt4d[3]
    return pt3d.flatten()   # (X, Y, Z) mm

def dist3d(a, b):
    return float(np.linalg.norm(a - b))

def get_lm_pixel(lm, w, h):
    """Return (px, py) from normalised landmark."""
    return lm.x * w, lm.y * h

def lm_visible_hip(lm):
    return lm.visibility >= VIS_HIP

def lm_visible_knee(lm):
    return lm.visibility >= VIS_KNEE

def draw_dot(frame, lm, w, h, color, r=22):
    if lm.visibility < VIS_THRESH:
        return
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(frame, (cx, cy), r, color, -1)
    cv2.circle(frame, (cx, cy), r + 3, WHITE, 2)

def draw_line(frame, lm_a, lm_b, w, h, color, thick=6):
    if lm_a.visibility < VIS_THRESH or lm_b.visibility < VIS_THRESH:
        return
    pa = (int(lm_a.x * w), int(lm_a.y * h))
    pb = (int(lm_b.x * w), int(lm_b.y * h))
    cv2.line(frame, pa, pb, color, thick)

def big_text(frame, text, y, color, scale=1.6, thick=3):
    """Centred text at height y."""
    h_img, w_img = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (w_img - tw) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 4)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def small_text(frame, text, x, y, color=WHITE, scale=0.55, thick=1):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ── Rolling buffers ────────────────────────────────────────────────────────
buf_left  = deque(maxlen=SMOOTH_N)
buf_right = deque(maxlen=SMOOTH_N)

frozen_left  = None   # mm — frozen when landmarks lost
frozen_right = None
frozen_count = 0

start_time = time.time()
ts1 = ts2 = 0

print("Running. Stand facing the midpoint between cameras.")
print("Wait for counter = 30, then measure. Walk back to compare. Q = quit.")

cv2.namedWindow("FEEDS — move this window to iPad", cv2.WINDOW_NORMAL)
cv2.namedWindow("RESULT — keep on MacBook screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FEEDS — move this window to iPad", 1280, 900)
cv2.resizeWindow("RESULT — keep on MacBook screen", 900, 500)

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        print("Stream dropped — retrying …")
        time.sleep(0.05)
        continue

    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]

    # ── Run MediaPipe ──────────────────────────────────────────────────────
    now_ms = int((time.time() - start_time) * 1000)

    mp1 = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
    mp2 = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))

    ts1 += 33
    ts2 += 33
    r1 = det1.detect_for_video(mp1, ts1)
    r2 = det2.detect_for_video(mp2, ts2)

    lm1 = r1.pose_landmarks[0] if r1.pose_landmarks else None
    lm2 = r2.pose_landmarks[0] if r2.pose_landmarks else None

    # ── Draw skeleton on both feeds ────────────────────────────────────────
    for frame, lm, w, h in [(f1, lm1, w1, h1), (f2, lm2, w2, h2)]:
        if lm is None:
            continue
        for hip_i, knee_i, col in [(L_HIP, L_KNEE, GREEN), (R_HIP, R_KNEE, CYAN)]:
            draw_line(frame, lm[hip_i], lm[knee_i], w, h, col)
            draw_dot(frame,  lm[hip_i],              w, h, col)
            draw_dot(frame,  lm[knee_i],             w, h, col)

    # ── Triangulate if both feeds have all 4 landmarks ─────────────────────
    both_left  = (lm1 and lm2 and
                  lm_visible_hip(lm1[L_HIP])   and lm_visible_knee(lm1[L_KNEE]) and
                  lm_visible_hip(lm2[L_HIP])   and lm_visible_knee(lm2[L_KNEE]))

    both_right = (lm1 and lm2 and
                  lm_visible_hip(lm1[R_HIP])   and lm_visible_knee(lm1[R_KNEE]) and
                  lm_visible_hip(lm2[R_HIP])   and lm_visible_knee(lm2[R_KNEE]))

    if both_left:
        u1h, v1h = get_lm_pixel(lm1[L_HIP],  w1, h1)
        u1k, v1k = get_lm_pixel(lm1[L_KNEE], w1, h1)
        u2h, v2h = get_lm_pixel(lm2[L_HIP],  w2, h2)
        u2k, v2k = get_lm_pixel(lm2[L_KNEE], w2, h2)

        p3_lhip  = triangulate(u1h, v1h, K1, D1, u2h, v2h, K2, D2, P1, P2)
        p3_lknee = triangulate(u1k, v1k, K1, D1, u2k, v2k, K2, D2, P1, P2)
        buf_left.append(dist3d(p3_lhip, p3_lknee))

    if both_right:
        u1h, v1h = get_lm_pixel(lm1[R_HIP],  w1, h1)
        u1k, v1k = get_lm_pixel(lm1[R_KNEE], w1, h1)
        u2h, v2h = get_lm_pixel(lm2[R_HIP],  w2, h2)
        u2k, v2k = get_lm_pixel(lm2[R_KNEE], w2, h2)

        p3_rhip  = triangulate(u1h, v1h, K1, D1, u2h, v2h, K2, D2, P1, P2)
        p3_rknee = triangulate(u1k, v1k, K1, D1, u2k, v2k, K2, D2, P1, P2)
        buf_right.append(dist3d(p3_rhip, p3_rknee))

    # ── Freeze when person walks away ──────────────────────────────────────
    person_visible = bool(lm1 or lm2)

    if not person_visible and (buf_left or buf_right):
        # Freeze the last stable average
        if buf_left:
            frozen_left  = np.mean(buf_left)
            frozen_count = len(buf_left)
        if buf_right:
            frozen_right = np.mean(buf_right)
        buf_left.clear()
        buf_right.clear()

    if person_visible:
        frozen_left = frozen_right = None   # reset freeze when back in frame

    # ══════════════════════════════════════════════════════════════════════
    # WINDOW 1 — FEEDS  (drag this to iPad extended display)
    # Full-res feeds top and bottom — dots already drawn full-size above
    # ══════════════════════════════════════════════════════════════════════
    TARGET_W = 1280
    scale1 = TARGET_W / w1
    scale2 = TARGET_W / w2
    disp1 = cv2.resize(f1, (TARGET_W, int(h1 * scale1)))
    disp2 = cv2.resize(f2, (TARGET_W, int(h2 * scale2)))

    # Camera labels — large enough to read on iPad
    cv2.putText(disp1, "CAM 1 - iPhone",  (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK,  5)
    cv2.putText(disp1, "CAM 1 - iPhone",  (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 2)
    cv2.putText(disp2, "CAM 2 - MacBook", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK,  5)
    cv2.putText(disp2, "CAM 2 - MacBook", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 2)

    feeds_window = np.vstack([disp1, disp2])
    cv2.imshow("FEEDS — move this window to iPad", feeds_window)

    # ══════════════════════════════════════════════════════════════════════
    # WINDOW 2 — RESULT  (stays on MacBook — readable from across the room)
    # ══════════════════════════════════════════════════════════════════════
    RW, RH = 900, 500
    result = np.zeros((RH, RW, 3), dtype=np.uint8)
    result[:] = (15, 15, 15)

    def huge(text, y, color, scale=2.2, thick=4):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = (RW - tw) // 2
        cv2.putText(result, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 6)
        cv2.putText(result, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def medium(text, y, color, scale=1.1, thick=2):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = (RW - tw) // 2
        cv2.putText(result, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 4)
        cv2.putText(result, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    if frozen_left is not None or frozen_right is not None:
        huge("WALK BACK — READ RESULT", 80, YELLOW, scale=1.6, thick=3)
        cv2.line(result, (40, 100), (RW - 40, 100), (60, 60, 60), 1)
        if frozen_left is not None:
            huge(f"LEFT:  {frozen_left:.0f} mm", 210, GREEN)
        if frozen_right is not None:
            huge(f"RIGHT: {frozen_right:.0f} mm", 340, CYAN)
        medium(f"averaged over {frozen_count} frames", 440, GREY, scale=0.9)

    elif buf_left or buf_right:
        count = max(len(buf_left), len(buf_right))

        # Progress bar
        bar_x0, bar_x1 = 60, RW - 60
        bar_y0, bar_y1 = 55, 95
        bar_fill = int((count / SMOOTH_N) * (bar_x1 - bar_x0))
        bar_col = GREEN if count >= SMOOTH_N else YELLOW
        cv2.rectangle(result, (bar_x0, bar_y0), (bar_x1, bar_y1), (50, 50, 50), -1)
        cv2.rectangle(result, (bar_x0, bar_y0), (bar_x0 + bar_fill, bar_y1), bar_col, -1)
        medium(f"Stabilising  {count} / {SMOOTH_N}", 130, bar_col, scale=1.3)

        if buf_left:
            huge(f"LEFT:  {np.mean(buf_left):.0f} mm", 240, GREEN)
        if buf_right:
            huge(f"RIGHT: {np.mean(buf_right):.0f} mm", 360, CYAN)

        if count >= SMOOTH_N:
            medium("STABLE - measure your leg now", 450, GREEN, scale=1.2)
        else:
            medium("Stand still — wait for bar to fill", 450, GREY)

    else:
        huge("Stand facing midpoint", 200, GREY, scale=1.8, thick=3)
        huge("between both cameras", 300, GREY, scale=1.8, thick=3)

    cv2.imshow("RESULT — keep on MacBook screen", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
det1.close()
det2.close()
print("Done.")