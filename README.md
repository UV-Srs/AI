# PPE Detection & Facial Recognition Compliance Monitoring System

An enterprise-grade, edge-AI computer vision system that inspects industrial Personal Protective Equipment (PPE) compliance in real-time while automatically verifying worker identities using on-device facial recognition.

---

## Table of Contents

- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Approved PPE Classes](#approved-ppe-classes)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Docker Deployment (CPU & GPU)](#docker-deployment-cpu--gpu)
- [Commands to Run the Project](#commands-to-run-the-project)
- [Running Automated Verification Tests](#running-automated-verification-tests)
- [Project Directory Structure](#project-directory-structure)
- [API & WebSocket Documentation](#api--websocket-documentation)
- [Privacy & Data Governance](#privacy--data-governance)

---

## About the Project

Workplace safety compliance in industrial, construction, and manufacturing sites requires continuous monitoring of Personal Protective Equipment (PPE). Traditional supervision is manual, error-prone, and lacks accountability.

This project delivers a **100% local, on-device compliance monitoring and facial recognition system** combining:
1. **YOLO26n Combined SH17 Object Detector**: Trained on industrial PPE datasets (SH17 + custom) across **8 approved classes**: `person`, `helmet`, `safety_vest`, `gloves`, `safety_shoes`, `goggles`, `mask`, and `ear_protection`.
2. **OpenCV YuNet Face Detector**: High-speed, multi-scale, CPU-optimized face detection (Apache-2.0).
3. **Google MediaPipe Face Landmarker**: 478 3D facial landmarks, anatomical contours (oval, eyes, lips, eyebrows), and 3D head pose estimation (yaw, pitch, roll).
4. **dlib ResNet-34 128-D Biometric Recognition**: 128-dimensional Euclidean feature space with calibrated ambiguity rejection ($\Delta \ge 0.08$ separation margin against second-best match) and 3-photo enrollment (Front, Left, Right).
5. **Spatial Geometry Association Engine**: Geometrically pairs detected PPE items to the corresponding person and associates the person with their verified employee identity.
6. **Dual Camera / Side-by-Side View**: Side-by-side display of the raw high-definition live webcam feed alongside real-time AI bounding boxes, 3D mesh overlays, and compliance diagnostics.
7. **Modern Human-Crafted User Interface**: Clean dashboard with white background, corporate blue buttons, 3D mesh toggles, dual camera view, recognition testing suite, and persistent audit logs.

---

## System Architecture

```mermaid
flowchart LR
    A[HD Webcam Stream] --> B[FastAPI WebSocket Endpoint\n/ws/detection]
    B --> C[Integrated Inference Pipeline]
    C --> D[YOLO26n Combined SH17\n8 PPE Classes]
    C --> E[OpenCV YuNet\nFace Detection & Keypoints]
    E --> F[Google MediaPipe\n478 3D Mesh & Head Pose]
    E --> G[dlib ResNet-34\n128-D Descriptors]
    G --> H[Ambiguity Rejection Matcher\nMargin Delta >= 0.08]
    D & H --> I[Spatial Geometry Association\nPerson <-> Face <-> PPE]
    I --> J[Role-Based Compliance Evaluator\nCompliant / Non-Compliant / Review]
    J --> K[SQLite / PostgreSQL DB\nAudit Event Cooldown Log]
    J --> L[Real-Time JSON Stream to UI]
    L --> M[React + TypeScript Frontend\nDual View & 3D Mesh Overlays]
```

---

## Key Features

- **Real-Time Dual-Engine Inference**: Simultaneously runs YOLO26n object detection and OpenCV SFace facial recognition locally on standard CPU hardware.
- **Spatial PPE-to-Person Alignment**:
  - Face bounding box centers are aligned to the upper 45% of the YOLO person detection box.
  - Helmets are mapped to the head region (top 30%).
  - Safety vests are mapped to the torso region (15%–75%).
  - Eye goggles are mapped to the facial crop.
  - Automatic headshot fallback generates an anchor box when only a webcam headshot is present.
- **Anti-Spoofing & Quality Gate**:
  - Sharpness check via Laplacian variance ($\text{threshold} \ge 45.0$).
  - Lighting check via mean pixel luminance ($35.0 \le \text{brightness} \le 225.0$).
  - Minimum resolution threshold ($50 \times 50$ px).
  - High-frequency edge texture gradient and HSV color gamut analysis.
- **Multi-Frame Identity Stabilization**: Employs a rolling FIFO track buffer to prevent identity flickering across adjacent video frames.
- **Local SQLite Database (`data/ppe_compliance.db`)**: Stores employee profiles, 128-d floating-point biometric templates, persistent audit logs, and settings.
- **Privacy & Right to Erasure**:
  - Operates 100% on-device with zero cloud dependencies.
  - Raw face photos are never saved; only irreversible 128-d numerical embeddings are retained.
  - Unconditional withdrawal of biometric consent immediately purges templates from disk.
- **Clean Enterprise Dashboard**:
  - **Dashboard**: High-level KPIs (compliance rate, total inspections, safety violations, active personnel) and recent event tables.
  - **Live Monitoring**: Real-time webcam viewport with color-coded compliance boxes, personnel tags, PPE counters, and model selection.
  - **Employees**: Personnel directory with search, employee registration, and instant webcam face enrollment.
  - **Audit Logs**: Filterable event log table with one-click **CSV Export**.
  - **Settings**: Adjustable sliders for face match threshold (0.200–0.650), event deduplication cooldown (5s–120s), and retention policies.

---

## Approved PPE Classes

The YOLO26n model detects 6 approved classes:

| Class ID | Class Name | Display Label | Description |
| :---: | :--- | :--- | :--- |
| `0` | `person` | Worker / Person | Human subject detection anchor |
| `1` | `helmet` | Hard Hat / Helmet | Head protection equipment |
| `2` | `safety_vest` | Safety Vest | High-visibility reflective vest |
| `3` | `gloves` | Safety Gloves | Hand protection gear |
| `4` | `safety_shoes` | Safety Shoes | Protective work boots |
| `5` | `goggles` | Eye Goggles | Protective eyewear |

---

## Prerequisites

Before running the application, ensure you have:

1. **Python 3.10 or 3.11** installed.
2. **Node.js (v18 or higher)** and **npm** installed.
3. A connected **webcam** or integrated camera.

---

## Installation & Setup

### 1. Clone or Open the Repository

Navigate to the project root directory:

```bash
cd "AI ML"
```

### 2. Set Up the Python Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install Python requirements
pip install -r requirements.txt
```

### 2.1 GPU Acceleration Setup (NVIDIA CUDA) - Branch: `GPU`

This branch (`GPU`) is pre-configured for NVIDIA CUDA hardware acceleration:

```bash
# 1. Verify NVIDIA GPU Driver & CUDA support:
nvidia-smi

# 2. Install PyTorch with CUDA 12.1 support (or CUDA 11.8 if using older drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Verify CUDA is detected:
python -c "import torch; print('CUDA Active:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Hardware Auto-Detection & Fallback:** The backend dynamically checks `torch.cuda.is_available()`. If an NVIDIA GPU is present, YOLO26n PPE inference, continuous retraining, and face recognition run directly on `cuda:0` with up to **4x-10x throughput speedup**. If no CUDA device is present, it automatically and seamlessly falls back to CPU mode without modifying code or settings.

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure PostgreSQL Database

The application connects to **PostgreSQL** by default, storing all employee records, 128-dimensional biometric face templates (`BYTEA`), and audit inspection events in PostgreSQL tables.

1. Open `.env` (or copy `.env.example` to `.env`) in the project root:

```env
DB_TYPE=postgres

# PostgreSQL Connection Credentials
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=PPE_Detect
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
```

2. Test your PostgreSQL connection and auto-initialize tables:

```powershell
python backend/app/db/test_postgres_connection.py
```

*(Optional) If you have existing records in SQLite and want to copy them to PostgreSQL:*

```powershell
python backend/app/db/migrate_to_postgres.py
```

> **Note**: If PostgreSQL is temporarily unreachable or credentials are not yet supplied, the system automatically and safely falls back to local SQLite (`data/ppe_compliance.db`) so operations are never interrupted.

---

## Docker Deployment (CPU & GPU)

The project includes an enterprise-grade Docker containerization setup orchestrating:
- **FastAPI Backend**: Port `8001` (CPU and NVIDIA CUDA 12.1 GPU builds)
- **React Frontend**: Port `5173` & Port `80` (production Nginx reverse-proxy with WebSocket support)
- **PostgreSQL Database**: Port `5432` (PostgreSQL 16 with automatic schema initialization)

### 1. Run in CPU Mode (Universal - No GPU required)

To run the complete system (DB, Backend, Frontend) with standard CPU execution:

```bash
# Using Docker Compose:
docker-compose up --build

# Or on Windows using 1-click script:
.\start_docker_cpu.ps1

# Or on Linux / macOS:
./start_docker_cpu.sh
```

### 2. Run in NVIDIA GPU Mode (CUDA 12.1 Hardware Acceleration)

To run with NVIDIA CUDA hardware acceleration:

```bash
# Using Docker Compose with the GPU overlay:
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Or on Windows using 1-click script:
.\start_docker_gpu.ps1

# Or on Linux:
./start_docker_gpu.sh
```

> **GPU Requirements**: NVIDIA host drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`docker-nvidia` / `nvidia-ctk`).

### Accessing the Containerized System

- **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173) or [http://localhost](http://localhost)
- **Backend API & Swagger Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **Backend Health Check**: [http://localhost:8001/api/health](http://localhost:8001/api/health)
- **PostgreSQL Database**: `localhost:5432` (`db: PPE_Detect`, `user: postgres`)

---

## Commands to Run the Project (Host / Local Setup)

You do not need batch files. Run the backend and frontend directly in separate terminal windows:

### Terminal 1: Start the FastAPI Backend

From the project root:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run the backend server
python backend/run.py
```

*Alternatively, using uvicorn directly:*

```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
```

- **Backend API**: `http://127.0.0.1:8001`
- **Swagger Documentation**: `http://127.0.0.1:8001/docs`
- **Live WebSocket**: `ws://127.0.0.1:8001/ws/detection`

---

### Terminal 2: Start the React Frontend

Open a new terminal window:

```bash
cd frontend
npm run dev
```

- **Frontend Application**: `http://localhost:5173`

Open `http://localhost:5173` in your browser. Click **"Turn On Camera"** and **"Start Detection"** on the Live Monitoring page to begin real-time verification.

---

## Running Automated Verification Tests

The repository includes a comprehensive end-to-end verification test suite that validates the database, OpenCV face engine, YOLO26n pipeline, consent withdrawal, and REST endpoints.

Run the test suite with:

```powershell
$env:PYTHONUNBUFFERED="1"
.\.venv\Scripts\python.exe backend/tests/verify_enterprise_system.py
```

Expected result:

```text
======================================================================
STARTING ENTERPRISE E2E VERIFICATION TEST SUITE
======================================================================
--> Step 1: Testing SQLite Biometric & Compliance Schema... [OK]
--> Step 2: Testing Employee Registration & Role-Based PPE... [OK]
--> Step 3: Testing OpenCV YuNet + SFace Face Engine... [OK]
--> Step 4: Testing Quality Assessment & Liveness Heuristics... [OK]
--> Step 5: Testing Integrated Multimodal Pipeline... [OK]
--> Step 6: Testing Event Logging & Stats... [OK]
--> Step 7: Testing Biometric Consent Withdrawal & Erasure... [OK]
--> Step 8: Testing FastAPI REST Route Handlers... [OK]
======================================================================
ALL ENTERPRISE VERIFICATION TESTS PASSED (100% SUCCESS)!
======================================================================
```

---

## Project Directory Structure

```text
AI ML/
├── README.md                              # Complete Project Guide & Documentation
├── requirements.txt                       # Python dependencies
├── .venv/                                 # Python virtual environment
│
├── backend/                               # FastAPI Application
│   ├── run.py                             # Backend entrypoint (uvicorn runner)
│   ├── app/
│   │   ├── main.py                        # FastAPI application & route registration
│   │   ├── db/
│   │   │   └── database.py                # SQLite schema, CRUD, consent & audit logic
│   │   ├── face/
│   │   │   └── face_engine.py             # OpenCV YuNet + SFace engine & liveness
│   │   ├── inference/
│   │   │   └── pipeline.py                # Integrated multimodal detection pipeline
│   │   ├── models/                        # Pretrained ONNX face models (Apache 2.0)
│   │   │   ├── face_detection_yunet_2023mar.onnx
│   │   │   └── face_recognition_sface_2021dec.onnx
│   │   ├── routes/
│   │   │   ├── api.py                     # REST endpoints (employees, events, stats, settings)
│   │   │   └── ws.py                      # Real-time WebSocket detection endpoint
│   │   ├── schemas/
│   │   │   └── detection.py               # Pydantic schemas
│   │   └── services/
│   │       └── model_manager.py           # YOLO26n checkpoint loader & switcher
│   └── tests/
│       └── verify_enterprise_system.py    # Automated E2E test suite
│
├── frontend/                              # React + TypeScript + Tailwind Frontend
│   ├── index.html                         # Light-themed HTML shell
│   ├── package.json                       # npm dependencies & build scripts
│   ├── tailwind.config.js                 # Tailwind CSS configuration
│   ├── vite.config.ts                     # Vite build configuration
│   └── src/
│       ├── main.tsx                       # React application root
│       ├── App.tsx                        # Main application container
│       ├── index.css                      # Global light styles (#f8fafc background)
│       ├── components/
│       │   ├── Header.tsx                 # Clean white header with blue active tabs
│       │   ├── WebcamView.tsx             # Camera stream, dual-canvas overlay, blue controls
│       │   ├── RightPanel.tsx             # Stats HUD, personnel list, model selector
│       │   ├── BottomPanel.tsx            # Live event stream & system specifications
│       │   └── DetectionCounters.tsx      # Clean PPE class counters
│       ├── views/
│       │   ├── DashboardView.tsx          # Executive compliance overview & KPI cards
│       │   ├── EmployeesView.tsx          # Directory, modal form, webcam face enrollment
│       │   ├── EventsView.tsx             # Filterable audit log & CSV export
│       │   └── SettingsView.tsx           # Configurable thresholds & privacy terms
│       ├── services/
│       │   ├── api.ts                     # REST API client & WebSocket helper
│       │   └── classes.ts                 # 6 approved PPE class definitions & colors
│       └── types/
│           └── index.ts                   # TypeScript interfaces
│
├── Model/                                 # Trained Model Weights
│   ├── PPE_YOLO26n_Best.pt                # YOLO26n baseline checkpoint
│   └── PPE_YOLO26n_Resumed_Best.pt        # YOLO26n resumed training checkpoint
│
├── Datasets/                              # Dataset Directory
│   └── Processed/
│       └── PPE_Final/
│           ├── data.yaml                  # Dataset configuration
│           ├── train/                     # 2,276 training images
│           ├── val/                       # 650 validation images
│           └── test/                      # 326 held-out test images
│
└── data/                                  # Local Persistent Data
    └── ppe_compliance.db                  # Local SQLite database
```

---

## API & WebSocket Documentation

### REST Endpoints (`/api`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health, active model, and engine readiness status. |
| `GET` | `/api/models` | Lists available trained YOLO26n checkpoints with file sizes. |
| `POST`| `/api/models/select` | Switches the active model checkpoint in real-time. |
| `GET` | `/api/classes` | Returns the 6 approved PPE classes and visual configurations. |
| `GET` | `/api/employees` | Lists all registered personnel and biometric enrollment states. |
| `POST`| `/api/employees` | Creates or updates an employee profile with required PPE rules. |
| `DELETE` | `/api/employees/{id}` | Deletes an employee and purges all associated biometric data. |
| `POST`| `/api/employees/enroll-face` | Evaluates quality and registers a 128-d face embedding template. |
| `POST`| `/api/employees/{id}/consent`| Withdraws biometric consent and immediately purges templates. |
| `GET` | `/api/events` | Retrieves inspection audit events with optional result/ID filters. |
| `GET` | `/api/events/export` | Generates a downloadable CSV audit log file. |
| `GET` | `/api/stats` | Aggregated metrics: compliance rate %, total scans, violations. |
| `GET` | `/api/settings` | Current face matching thresholds and event cooldown windows. |
| `POST`| `/api/settings` | Updates system thresholds and default required PPE lists. |

### WebSocket Endpoint (`/ws/detection`)

- **URL**: `ws://127.0.0.1:8001/ws/detection`
- **Request Payload** (JSON):
  ```json
  {
    "image": "data:image/jpeg;base64,...",
    "conf_threshold": 0.10,
    "camera_id": "CAM-01"
  }
  ```
- **Response Payload** (JSON):
  ```json
  {
    "success": true,
    "persons": [
      {
        "person_box": [120, 80, 480, 560],
        "face": {
          "bbox": [240, 100, 360, 240],
          "employee_id": "EMP-101",
          "employee_name": "Marcus Vance",
          "match_score": 0.892,
          "identity_state": "REGISTERED"
        },
        "associated_ppe": [
          {"class_name": "helmet", "confidence": 0.84, "box": [220, 60, 380, 160]},
          {"class_name": "safety_vest", "confidence": 0.78, "box": [160, 220, 440, 500]}
        ],
        "checklist": {
          "helmet": "DETECTED",
          "safety_vest": "DETECTED"
        },
        "missing_ppe": [],
        "overall_result": "COMPLIANT"
      }
    ],
    "total_persons": 1,
    "compliant_count": 1,
    "violation_count": 0,
    "inference_time_ms": 112.5,
    "fps": 8.9
  }
  ```

---

## Privacy & Data Governance

1. **Local Execution**: All frame processing, face detection, feature extraction, and database persistence occur on your local machine. No images, video streams, or biometric data are transmitted over the internet or sent to external cloud APIs.
2. **Non-Reversible Embeddings**: Raw face photographs are not stored permanently. Only irreversible 128-dimensional mathematical feature vectors are saved.
3. **Unconditional Right to Erasure**: Employees have the legal right to revoke biometric consent at any time. Revocation triggers an immediate purge of the biometric template from SQLite storage.
4. **Transparent Liveness Analysis**: Anti-spoofing relies on local texture variance heuristics and color gamut analysis; the system clearly reports when a subject requires visual review (`LIVENESS_REVIEW`).

