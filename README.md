# PositionAI

Real-time positioning guidance for X-ray radiographers, built for resource-constrained hospitals.

## What it is

PositionAI uses two smartphones and a laptop to give radiographers live visual feedback while positioning patients for X-ray exposure. A "ghost skeleton" overlay shows the target position; the live skeleton turns red → yellow → green as the patient is moved into place. When both camera angles are green, the laptop displays a **READY** signal and the radiographer can expose.

The problem it addresses: incorrect patient positioning is a leading cause of repeat X-ray exposures in rural and lower-resource hospitals where dedicated radiographers may not be available and assistants are trained on the job. Each retake means additional radiation dose to the patient, missed diagnostic detail, and machine wear. Camera-based positioning aids exist, mostly built into new premium X-ray systems or sold as add-ons. PositionAI explores a low-cost, phone-based setup for hospitals that cannot replace their machines.

## Status

**Paused.** Each component was built and tested on an iPhone + MacBook-webcam test rig. Testing showed the final setup needs two phones using rear cameras in portrait orientation to capture breathing in detail, so calibration will be redone before integration. Meanwhile I am studying the positioning standards in Bontrager, view by view.

| Component | Status | Notes |
|---|---|---|
| MediaPipe pose + RED/YELLOW/GREEN scoring | ✅ Tested | `calibrated_pose_demo.py` — single webcam; knee angle scored against a fixed 175° target; ghost saved by keypress |
| Stereo calibration (two cameras) | ✅ Tested | RMSE 2.46 px (target < 0.5 px; Iriun Wi-Fi H.264 compression suspected) |
| DLT 3D triangulation | ✅ Tested | Hip-to-knee distance within ~1–2 cm of a tape measure (n = 1 informal trial; tape reference imprecise; repeat trials planned on final rig) |
| ArUco marker detection | ✅ Tested | Stable detection in both feeds (`DICT_4X4_50`, 18.1 cm marker) |
| `positioning_standards.json` (Hip AP) | ✅ Drafted | Positioning values from Bontrager 8th ed. p.283–284; numeric tolerances are placeholders pending radiographer review |
| Integrated feedback loop | ⏳ Planned | Triangulation + ArUco + standards + ghost, after recalibration on the final rig |
| Streamlit three-route UI | ⏳ Planned | `/front`, `/side`, `/` |
| Optical-flow breathing detection (Chest PA) | ⏳ Planned | Farneback method |

## Architecture

```
┌────────────────────────────┐         ┌──────────────────────────┐
│   PATIENT SIDE             │         │   RADIOGRAPHER SIDE      │
│                            │         │                          │
│   📱 Camera 1 (front)      │  WiFi   │   💻 MacBook (master)    │
│   📱 Camera 2 (side, ~90°) │ ──────▶ │   • MediaPipe Pose       │
│                            │         │   • Stereo calibration   │
│   Ghost skeleton overlay   │ ◀────── │   • DLT triangulation    │
│   on each phone screen     │         │   • ArUco anchoring      │
│                            │         │   • Tolerance scoring    │
└────────────────────────────┘         └──────────────────────────┘
```

The laptop is the single processor. Phones stream video via EpocCam or Camo and act purely as displays. Three Streamlit routes will serve the three different views: `/front` and `/side` to the phones (full-screen ghost overlay only), `/` to the laptop (patient intake + technique chart + READY signal).

## Ghost skeleton

The core feedback mechanism is a **partial** ghost skeleton — a dotted target outline showing where the patient's body should be. Each clinical view declares which landmarks to validate (e.g. Hip AP validates hips, knees, ankles only) and which to ignore. Landmarks outside the validated set are never drawn, never scored, and never cause false readings.

Four states:
- 🔴 **RED** — Live skeleton far from target. Move toward ghost.
- 🟡 **YELLOW** — Within ~50% of tolerance. Fine adjustment.
- 🟢 **GREEN** — All parameters within tolerance. READY.
- 🟠 **AMBER** — Validated landmark visibility below 0.5. "Cannot assess." Never READY, never WAIT.

ArUco markers (printed, laminated, taped to the room) anchor the ghost skeleton to physical objects: marker A on the X-ray plate, marker B on the floor at the standing point. Ghost coordinates are defined relative to marker positions, so the system tolerates small phone displacement without full recalibration.

## Clinical scope

Six priority views, all referenced to *Bontrager's Textbook of Radiographic Positioning and Related Anatomy*, 8th edition:

- Hip AP (p.283)
- Neck of Femur Lateral / Danelius-Miller (p.284)
- Knee AP (p.244)
- Skull AP Towne (p.411) — uses MediaPipe Face Mesh
- Skull Lateral (p.412) — uses MediaPipe Face Mesh
- Chest PA (p.91) — uses optical-flow breathing detection

Patient conditions handled in MVP: hip replacement (contraindication warning), amputation / missing landmark (visibility-based AMBER), limited ROM (relaxed tolerance with radiographer flag), bedridden / portable.

## Hardware

**Current test rig** (used for all testing so far):

- **MacBook Air M2** — runs all processing
- **iPhone 15 Pro Max** — Camera 1, streamed to the MacBook over Wi-Fi via Iriun
- **MacBook webcam** — Camera 2, a stand-in for the second phone
- **Tripod with taped feet** for the iPhone — cameras must not move after calibration
- **One printed ArUco marker** (`DICT_4X4_50`, ID 0, 18.1 cm)
- **iPad Air Gen 5** — extended display for the camera-feed window

**Planned setup:**

- **Two phones on tripods with taped feet**, using rear cameras in portrait orientation to capture breathing in detail — stereo calibration will be redone on this rig
- **MacBook Air M2** — single processor, unchanged
- **Three printed ArUco markers** (`DICT_4X4_50`, sizes 10–20 cm) — generated at [chev.me/arucogen](https://chev.me/arucogen)

## Setup

```bash
# Clone
git clone https://github.com/thepalai/PositionAI.git
cd PositionAI

# Create venv (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The MediaPipe pose model (`pose_landmarker_heavy.task`, ~30 MB) is downloaded automatically on first run.

Stereo calibration files (`stereo_calib_*.npz`) and per-session ghost calibration (`ghost_calibration.json`) are not committed — these are specific to the cameras and room they were calibrated in. You'll need to run calibration in your own setup.

## Repository contents

| File | Purpose |
|---|---|
| `calibrated_pose_demo.py` | Single-camera ghost skeleton + RED/YELLOW/GREEN scoring |
| `stereo_triangulation.py` | Two-camera DLT triangulation; reports hip-to-knee distance for checking against a tape measure |
| `aruco_detection.py` | ArUco marker detection across both feeds |
| `positioning_standards.json` | Clinical tolerances per view (Hip AP drafted) |
| `requirements.txt` | Python dependencies |

## Built by

**Mo (Benjamaporn)** — single developer.

Background: BEng Mechanical Engineering (Thammasat); Postgraduate Diploma in Medical Radiation Physics (Swansea University — CT imaging, ICRP 89/93 dose modelling); 4 months as a radiographer assistant during COVID-19, performing bedside chest X-rays in a temporary hospital.

Clinical input:
- A radiographer I worked with during COVID-19, whose answers to my questions shaped this idea.

Built on the open-source work of:
- [MediaPipe](https://github.com/google/mediapipe) — pose and face landmark detection (Google)
- [OpenCV](https://opencv.org/) — stereo calibration, ArUco, optical flow
- [TemugeB/python_stereo_camera_calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) — stereo calibration reference
- [TemugeB/bodypose3d](https://github.com/TemugeB/bodypose3d) — DLT triangulation reference

## How it was built

Code written with AI coding assistants. I designed the system, ran and measured the tests, and am now learning the codebase module by module.

## License

To be added — the core will be released under Apache 2.0 once the MVP is integrated. The validated clinical database for Thai hospital deployment will remain proprietary.
