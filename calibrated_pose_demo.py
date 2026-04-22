import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import urllib.request

# --- Download model if missing ---
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pose_landmarker_heavy.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- MediaPipe setup ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

# --- Constants ---
CALIBRATION_FILE = "ghost_calibration.json"

VIS_HIP_KNEE = 0.5
VIS_ANKLE    = 0.3

TARGET_KNEE_ANGLE = 175.0
TOLERANCE_GREEN   = 5.0
TOLERANCE_YELLOW  = 15.0

RED    = (0,   0,   255)
YELLOW = (0,   255, 255)
GREEN  = (0,   255, 0)
AMBER  = (0,   165, 255)
WHITE  = (255, 255, 255)
GREY   = (180, 180, 180)

# --- Load or initialize ghost ---
def load_ghost():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        print(f"Ghost loaded from {CALIBRATION_FILE}")
        return data
    print("No calibration file found. Stand at X-ray position and press C.")
    return None

def save_ghost(landmarks, w, h):
    # Save normalized coords — NOT pixel coords
    # x is saved in flipped space so ghost matches display
    data = {
        "l_hip":   [1.0 - landmarks[23].x, landmarks[23].y],
        "r_hip":   [1.0 - landmarks[24].x, landmarks[24].y],
        "l_knee":  [1.0 - landmarks[25].x, landmarks[25].y],
        "r_knee":  [1.0 - landmarks[26].x, landmarks[26].y],
        "l_ankle": [1.0 - landmarks[27].x, landmarks[27].y],
        "r_ankle": [1.0 - landmarks[28].x, landmarks[28].y],
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Ghost saved to {CALIBRATION_FILE}")
    return data

GHOST_CONNECTIONS = [
    ("l_hip",   "r_hip"),
    ("l_hip",   "l_knee"),
    ("l_knee",  "l_ankle"),
    ("r_hip",   "r_knee"),
    ("r_knee",  "r_ankle"),
]

# --- Drawing helpers ---
def draw_dotted_line(frame, pt_a, pt_b, color, thickness=2, gap=10):
    dist = np.linalg.norm(np.array(pt_b) - np.array(pt_a))
    if dist == 0:
        return
    steps = int(dist / gap)
    for i in range(steps):
        if i % 2 == 0:
            t0 = i / steps
            t1 = min((i + 1) / steps, 1.0)
            p0 = (int(pt_a[0] + t0 * (pt_b[0] - pt_a[0])),
                  int(pt_a[1] + t0 * (pt_b[1] - pt_a[1])))
            p1 = (int(pt_a[0] + t1 * (pt_b[0] - pt_a[0])),
                  int(pt_a[1] + t1 * (pt_b[1] - pt_a[1])))
            cv2.line(frame, p0, p1, color, thickness)

def draw_ghost(frame, ghost, w, h):
    pts = {k: (int(v[0] * w), int(v[1] * h)) for k, v in ghost.items()}
    for (a, b) in GHOST_CONNECTIONS:
        draw_dotted_line(frame, pts[a], pts[b], WHITE, thickness=2, gap=12)
    for pt in pts.values():
        cv2.circle(frame, pt, 5, WHITE, 1)

def draw_connection(frame, lm_a, lm_b, color, w, h,
                    thickness=3, vis_threshold=0.3):
    if lm_a.visibility < vis_threshold or lm_b.visibility < vis_threshold:
        return
    pt_a = (int(lm_a.x * w), int(lm_a.y * h))
    pt_b = (int(lm_b.x * w), int(lm_b.y * h))
    cv2.line(frame, pt_a, pt_b, color, thickness)

def draw_dot(frame, lm, color, w, h, radius=7, vis_threshold=0.3):
    if lm.visibility < vis_threshold:
        return
    pt = (int(lm.x * w), int(lm.y * h))
    cv2.circle(frame, pt, radius, color, -1)

def draw_status_box(frame, left_color, right_color, pelvis_tilt, w, h):
    def label(c):
        if c == GREEN:  return "GREEN",  GREEN
        if c == YELLOW: return "YELLOW", YELLOW
        if c == RED:    return "RED",    RED
        return "AMBER", AMBER

    l_text, l_col = label(left_color)
    r_text, r_col = label(right_color)

    cv2.rectangle(frame, (10, 10), (300, 100), (30, 30, 30), -1)
    cv2.putText(frame, f"LEFT:  {l_text}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_col, 2)
    cv2.putText(frame, f"RIGHT: {r_text}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_col, 2)
    cv2.putText(frame, f"PELVIS TILT: {pelvis_tilt:.1f}", (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

def draw_calibration_prompt(frame, landmarks_ready, w, h):
    # Show instruction at bottom of screen
    if landmarks_ready:
        msg = "C = save ghost  |  Q = quit"
        color = GREEN
    else:
        msg = "Stand at X-ray position — landmarks not ready"
        color = AMBER
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, msg, (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def score_to_color(angle, target, tol_green, tol_yellow):
    diff = abs(angle - target)
    if diff <= tol_green:
        return GREEN
    elif diff <= tol_yellow:
        return YELLOW
    else:
        return RED

def check_leg_visibility(landmarks, hip_idx, knee_idx, ankle_idx):
    hip_ok   = landmarks[hip_idx].visibility  >= VIS_HIP_KNEE
    knee_ok  = landmarks[knee_idx].visibility >= VIS_HIP_KNEE
    ankle_ok = landmarks[ankle_idx].visibility >= VIS_ANKLE
    return hip_ok and knee_ok and ankle_ok

# --- Main ---
ghost = load_ghost()
calibration_mode = ghost is None  # if no file, start in calibration mode

print("Running. Press C to calibrate ghost. Press Q to quit.")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # Draw ghost if we have one
    if ghost:
        draw_ghost(frame, ghost, w, h)

    landmarks_ready = False

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        left_visible  = check_leg_visibility(landmarks, 23, 25, 27)
        right_visible = check_leg_visibility(landmarks, 24, 26, 28)
        landmarks_ready = left_visible and right_visible

        # Score
        if left_visible:
            left_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            left_color = score_to_color(left_angle, TARGET_KNEE_ANGLE,
                                        TOLERANCE_GREEN, TOLERANCE_YELLOW)
        else:
            left_color = AMBER

        if right_visible:
            right_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            right_color = score_to_color(right_angle, TARGET_KNEE_ANGLE,
                                         TOLERANCE_GREEN, TOLERANCE_YELLOW)
        else:
            right_color = AMBER

        pelvis_tilt = abs(landmarks[23].y - landmarks[24].y) * 100

        # Draw left leg
        draw_connection(frame, landmarks[23], landmarks[25], left_color, w, h)
        draw_connection(frame, landmarks[25], landmarks[27], left_color, w, h)
        draw_dot(frame, landmarks[23], left_color, w, h)
        draw_dot(frame, landmarks[25], left_color, w, h)
        draw_dot(frame, landmarks[27], left_color, w, h)

        # Draw right leg
        draw_connection(frame, landmarks[24], landmarks[26], right_color, w, h)
        draw_connection(frame, landmarks[26], landmarks[28], right_color, w, h)
        draw_dot(frame, landmarks[24], right_color, w, h)
        draw_dot(frame, landmarks[26], right_color, w, h)
        draw_dot(frame, landmarks[28], right_color, w, h)

        # Hip connection
        draw_connection(frame, landmarks[23], landmarks[24], WHITE, w, h)

        draw_status_box(frame, left_color, right_color, pelvis_tilt, w, h)

    # Always show calibration prompt at bottom
    draw_calibration_prompt(frame, landmarks_ready, w, h)

    frame = cv2.flip(frame, 1)
    cv2.imshow("PositionAI - Step 6", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and landmarks_ready:
        ghost = save_ghost(landmarks, w, h)
        print("Ghost calibrated. Standing position saved.")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done.")