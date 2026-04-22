# PositionAI

Real-time positioning guidance for X-ray radiographers, built for resource-constrained hospitals.

## What it is

PositionAI uses two smartphones and a laptop to give radiographers live visual feedback while positioning patients for X-ray exposure. A "ghost skeleton" overlay shows the target position; the live skeleton turns red → yellow → green as the patient is moved into place. When both camera angles are green, the laptop displays a **READY** signal and the radiologist can expose.

The problem it addresses: incorrect patient positioning is a leading cause of repeat X-ray exposures in rural and lower-resource hospitals where dedicated radiographers may not be available and assistants are trained on the job. Each retake means additional radiation dose to the patient, missed diagnostic detail, and machine wear. No affordable real-time positioning guidance currently exists for any X-ray machine brand.

## Status

**Active development.** Core components (Chats 1–6) are individually validated and ready for integration. Chat 7 brings them together into a single Hip AP feedback loop.

| Component | Status | Notes |
|---|---|---|
| MediaPipe pose + RED/YELLOW/GREEN scoring | ✅ Validated | `step6_calibrated_ghost.py` — ghost calibration via keypress |
| Stereo calibration (two cameras) | ✅ Validated | RMSE 2.43 with H.264-compressed WiFi feeds |
| DLT 3D triangulation | ✅ Validated | 1–2 cm error vs. tape measurement |
| ArUco marker detection | ✅ Validated | `DICT_4X4_50`, both feeds simultaneously |
| `positioning_standards.json` (Hip AP) | ✅ Drafted | Pending radiologist review |
| Integrated feedback loop (Chat 7) | 🚧 In progress | Triangulation + ArUco + standards + ghost |
| Streamlit three-route UI | ⏳ Planned | `/front`, `/side`, `/` |
| Optical-flow breathing detection (Chest PA) | ⏳ Planned | Farneback method |

## Architecture

```
┌────────────────────────────┐         ┌──────────────────────────┐
│   PATIENT SIDE             │         │   RADIOLOGIST SIDE       │
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

Five priority views, all referenced to *Bontrager's Handbook of Radiographic Positioning*, 8th edition:

- Hip AP (p.283)
- Neck of Femur Lateral / Danelius-Miller (p.284)
- Knee AP (p.244)
- Skull AP Towne (p.411) — uses MediaPipe Face Mesh
- Skull Lateral (p.412) — uses MediaPipe Face Mesh
- Chest PA (p.91) — uses optical-flow breathing detection

Patient conditions handled in MVP: hip replacement (contraindication warning), amputation / missing landmark (visibility-based AMBER), limited ROM (relaxed tolerance with radiologist flag), bedridden / portable.

## Hardware

- **MacBook Air M2** — single processor, runs everything
- **iPhone 15 Pro Max** (front view) — Camera 1, streamed via EpocCam
- **iPad Air Gen 5 / second iPhone** (side view, ~90° from front) — Camera 2
- **Tripods with taped feet** — phones must not move after calibration
- **Three printed ArUco markers** (`DICT_4X4_50`, sizes 10–20 cm) — generated at [chev.me/arucogen](https://chev.me/arucogen)

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/positionai.git
cd positionai

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
| `step6_calibrated_ghost.py` | Single-camera ghost skeleton + RED/YELLOW/GREEN scoring |
| `step7_dlt_validate.py` | Two-camera DLT triangulation, validated against tape measure |
| `Step8_aruco_detect.py` | ArUco marker detection across both feeds |
| `positioning_standards.json` | Clinical tolerances per view (Hip AP drafted) |
| `requirements.txt` | Python dependencies |
| `project_reference_map_v3.html` | Architecture and module reference |

## Built by

**Mo (Benjamaporn)** — single developer.

Background: BEng Mechanical Engineering (Thammasat); Postgraduate Diploma in Medical Radiation Physics (Swansea University — CT imaging, ICRP 89/93 dose modelling); 4 months as a radiographer assistant during COVID-19, performing bedside chest X-rays in a temporary hospital.

Clinical advisors:
- **Cousin** — medical device specialist, training X-ray users in rural Thai hospitals (primary deployment channel)
- **Radiologist ex-coworker** — Siriraj-connected, reviewing `positioning_standards.json`

Built on the open-source work of:
- [MediaPipe](https://github.com/google/mediapipe) — pose and face landmark detection (Google)
- [OpenCV](https://opencv.org/) — stereo calibration, ArUco, optical flow
- [TemugeB/python_stereo_camera_calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) — stereo calibration reference
- [TemugeB/bodypose3d](https://github.com/TemugeB/bodypose3d) — DLT triangulation reference

## License

To be added — the core will be released under Apache 2.0 once the MVP is integrated. The validated clinical database for Thai hospital deployment will remain proprietary.
