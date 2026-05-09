# Malaria-Diagnosis-via-microscopic-examination-using-yolov11
AI-Powered Web framework for Enhancing Malaria Diagnosis using YOLOv11 for Rapid Image-Based Detection.

# 🔬 BoboMal — Malaria Diagnosis Web App

> **BSc Final Year Project · Miva Open University**  
> Department of Data Science, School of Computing · 2026

| | |
|---|---|
| **Author** | Yomi Aledare |
| **Matriculation No.** | 2023/C/DSC/161 |
| **Supervisor** | Chinonso Alaebo |
| **Stack** | Python · Streamlit · YOLOv11 · OpenCV · PyTorch |

---

## Overview

**BoboMal** is a web-based malaria diagnostic tool that analyses Giemsa-stained thin blood smear images using a custom-trained **YOLOv11** object detection model. It is designed to assist medical personnel — especially in resource-limited settings — by producing fast, cell-level detection of *Plasmodium* parasites without requiring specialist laboratory infrastructure at the point of care.

The system identifies two clinically significant *Plasmodium* species directly from a single smear image:

- ***P. falciparum*** — detected via classes `Seg-F`, `F-R`, `F-S`, `F-T`
- ***P. vivax*** — detected via classes `V-G`, `V-R`, `V-S`, `V-T`

---

## The YOLOv11 Model

[YOLO (You Only Look Once)](https://docs.ultralytics.com/) is a family of real-time object detection architectures. **YOLOv11** is the latest generation, offering improved accuracy and speed over its predecessors through architectural refinements including enhanced feature extraction and a more efficient detection head.

### Why YOLO for Malaria Detection?

Traditional malaria diagnosis requires a trained microscopist to manually scan hundreds of fields per slide — a slow, subjective, and error-prone process. YOLO's single-pass inference makes it ideal for this task because it:

- Scans the full image in one forward pass, detecting all cells simultaneously
- Produces bounding-box overlays that give visual, interpretable output for clinical review
- Runs in approximately **~2 seconds** per image, even on modest hardware
- Operates at the **cell level**, distinguishing infected from healthy cells with high precision

### Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv11 (Ultralytics) |
| Input resolution | 640 × 640 px |
| Confidence threshold | 0.25 |
| Weight file | `weights/best.pt` |
| Training data | Annotated Giemsa-stained microscopy images |
| Positive classes | `Seg-F`, `F-R`, `F-S`, `F-T`, `V-G`, `V-R`, `V-S`, `V-T` |
| Output | Bounding boxes · class labels · confidence scores |

Images are automatically resized to **640 × 640 px** (LANCZOS resampling) before being passed to the model, ensuring consistent inference regardless of the original upload resolution.

---

## Web Application

The app is built with **Streamlit** and structured as a three-page single-file application (`app.py`). It is gated behind a login screen and only accessible to authorised users.

### Pages

#### 🔐 Login
Users must authenticate before accessing any part of the app. Invalid credentials display an error; successful login redirects to the Home dashboard.

#### 🏠 Home (Dashboard)
A landing page that displays key system stats (average inference time, required image format) and a brief overview of how the system works. Serves as an orientation screen for new users.

#### 🔬 Diagnosis
The core page of the application. It is split into two panels:

**Left panel — Image Feed**

- Drag-and-drop or click-to-browse file uploader (supports JPG, PNG, BMP, TIFF)
- Uploaded image previewed immediately before analysis
- "Run Analysis" button triggers inference

**Right panel — Analysis Result**

After inference, the right panel displays:

1. **Result banner** — a green *Negative* or red *Positive* banner with a plain-language summary
2. **Detection output** — the original image annotated with YOLOv11 bounding boxes and class labels
3. **Analysis Summary card** — a structured breakdown containing:

| Field | Description |
|---|---|
| Class | Positive ⚠️ or Negative ✅ |
| Parasite Name *(positive only)* | *P. falciparum*, *P. vivax*, or both |
| Confidence Score | Highest detection confidence among positive-class boxes |
| Parasites Detected | Count of positive-class boxes (positive) or all detected boxes (negative) |
| Volume of Infection | Parasitemia category derived from estimated parasite density |
| Risk Level | Submicroscopic / Low / Moderate / High / Severe |
| Inference Time | Wall-clock seconds for model prediction |
| Model | YOLOv11 |

#### ℹ️ About
Project description, author information, key technologies, and a visual "How It Works" grid explaining the four-step pipeline from sample input to clinical review.

---

## Diagnosis Logic

```
Upload image
    └─▶ Convert to RGB
    └─▶ Resize to 640×640 (LANCZOS)
    └─▶ Run YOLOv11 inference (conf ≥ 0.25)
            └─▶ For each detected box:
                    Is class in {Seg-F, F-R, F-S, F-T}?  →  Falciparum hit
                    Is class in {V-G, V-R, V-S, V-T}?    →  Vivax hit
            └─▶ Any hit?  →  POSITIVE  (parasite count = positive-class boxes)
            └─▶ No hit?   →  NEGATIVE  (parasite count = all detected boxes)
    └─▶ Parasite name: Falciparum | Vivax | Falciparum & Vivax | None
    └─▶ Parasitemia classified by estimated parasite density (n × 500 /µL)
```

### Parasitemia Classification

| Category | Estimated density | Risk |
|---|---|---|
| Submicroscopic / low | < 50 /µL | Very low |
| Low parasitemia | 50 – 5,000 /µL | Low |
| Moderate | 5,000 – 100,000 /µL | Moderate |
| High | 100,000 – 250,000 /µL | High |
| Severe malaria risk | > 250,000 /µL | Severe |

---

## Project Structure

```
bobomal/
├── app.py              # Full Streamlit application (single-file)
├── weights/
│   └── best.pt         # Trained YOLOv11 weights (required)
└── README.md
```

---

## Installation & Running

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bobomal.git
cd bobomal

# 2. Install dependencies
pip install streamlit ultralytics opencv-python pillow

# 3. Place model weights
mkdir -p weights
# Copy your trained best.pt into the weights/ folder

# 4. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Default Login Credentials

| Username | Password |
|---|---|
| `admin` | `admin123` |

> **Note:** Update credentials in `app.py` before any deployment. The current credentials are for development/demonstration only.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `ultralytics` | YOLOv11 model loading and inference |
| `opencv-python` | BGR→RGB conversion for annotated output |
| `Pillow` | Image opening, conversion, and resizing |
| `torch` / `torchvision` | PyTorch backend for YOLO (auto-installed with ultralytics) |

---

## ⚠️ Disclaimer

BoboMal is a **research and educational tool** developed as a BSc final year project. It is **not a certified medical device** and must not be used as a substitute for professional laboratory diagnosis. All results should be reviewed and confirmed by a qualified healthcare professional before any clinical decision is made.

---

*Miva Open University · School of Computing · Data Science · 2026*