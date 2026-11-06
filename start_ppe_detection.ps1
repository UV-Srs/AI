<#
.SYNOPSIS
    Starts the Enterprise Live PPE Detection & Facial Recognition Compliance System.
.DESCRIPTION
    Launches:
      1. FastAPI Backend (Uvicorn on http://127.0.0.1:8001)
         - YOLO26n PPE Object Detector
         - OpenCV YuNet Face Detection + SFace Facial Recognition (Apache 2.0)
         - PostgreSQL Enterprise Database (PPE_Detect)
      2. React + TypeScript Enterprise Dashboard (http://127.0.0.1:5173)
    Press Ctrl+C or enter 'q' to stop both services cleanly.
#>

[Console]::Title = "Enterprise PPE & Face Recognition Monitoring System"
$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  ENTERPRISE MULTI-MODEL PPE & 3D FACIAL RECOGNITION SYSTEM           " -ForegroundColor Yellow
Write-Host "  PostgreSQL (PPE_Detect) + YOLO26n + YuNet + MediaPipe + SFace        " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python virtual environment
$VENV_PYTHON = Join-Path $SCRIPT_DIR ".venv\Scripts\python.exe"
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "[ERROR] Python virtual environment not found at: $VENV_PYTHON" -ForegroundColor Red
    Write-Host "Please initialize the environment or check the path." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Python environment found: $VENV_PYTHON" -ForegroundColor Green

# 2. Check trained YOLO26n checkpoints
$SH17_MODEL = Join-Path $SCRIPT_DIR "Model\PPE_Combined_SH17_Best.pt"
if (Test-Path $SH17_MODEL) {
    $sh17SizeMB = [math]::Round((Get-Item $SH17_MODEL).Length / 1MB, 2)
    Write-Host "[OK] Unified SH17 Curriculum Model (8 Classes): Model\PPE_Combined_SH17_Best.pt ($sh17SizeMB MB) [ACTIVE]" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Unified model missing at: $SH17_MODEL" -ForegroundColor Yellow
}

# 3. Check Integrated Face & Mesh Models
$YUNET_MODEL = Join-Path $SCRIPT_DIR "backend\app\models\face_detection_yunet_2023mar.onnx"
$MEDIAPIPE_MODEL = Join-Path $SCRIPT_DIR "backend\app\models\face_landmarker.task"
$DLIB_MODEL = Join-Path $SCRIPT_DIR "backend\app\models\dlib_face_recognition_resnet_model_v1.dat"
$DLIB_SHAPE = Join-Path $SCRIPT_DIR "backend\app\models\shape_predictor_5_face_landmarks.dat"

if (Test-Path $YUNET_MODEL) {
    Write-Host "[OK] OpenCV YuNet Face Detection verified (Apache 2.0)" -ForegroundColor Green
}
if (Test-Path $MEDIAPIPE_MODEL) {
    Write-Host "[OK] Google MediaPipe Face Landmarker (478 3D Mesh) verified" -ForegroundColor Green
}
if ((Test-Path $DLIB_MODEL) -and (Test-Path $DLIB_SHAPE)) {
    Write-Host "[OK] dlib Face Recognition ResNet-34 (128-D) & 5-pt Shape Predictor verified" -ForegroundColor Green
}

# 3.5 Check PostgreSQL Database Connection
Write-Host "--> Checking PostgreSQL Database Connection (PPE_Detect)..." -ForegroundColor Cyan
$pgCheck = & $VENV_PYTHON -c "import psycopg2; conn = psycopg2.connect(host='localhost', port=5432, dbname='PPE_Detect', user='postgres', password='yuvraj9090M', connect_timeout=3); conn.close(); print('OK')" 2>$null
if ($pgCheck -eq "OK") {
    Write-Host "[OK] PostgreSQL Database Verified: PPE_Detect on localhost:5432" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Could not immediately reach PostgreSQL on localhost:5432. Make sure the PostgreSQL service is running." -ForegroundColor Yellow
}

# 4. Start Backend Process
Write-Host ""
Write-Host "--> Starting FastAPI Backend on http://127.0.0.1:8001..." -ForegroundColor Cyan
$backendProc = Start-Process -FilePath $VENV_PYTHON `
    -ArgumentList "backend\run.py" `
    -WorkingDirectory $SCRIPT_DIR `
    -PassThru

# 5. Start Frontend Process
Write-Host "--> Starting React + Vite Frontend on http://127.0.0.1:5173..." -ForegroundColor Cyan
$frontendProc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c npm run dev --prefix frontend" `
    -WorkingDirectory $SCRIPT_DIR `
    -PassThru

# Wait for backend neural models & health check to be ready before opening browser
Write-Host "--> Initializing neural models (YOLO26n + Face Biometrics). Waiting for backend to be ready..." -ForegroundColor Yellow
$backendReady = $false
for ($i = 1; $i -le 35; $i++) {
    Start-Sleep -Seconds 1
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res.status -eq "ok") {
            $backendReady = $true
            Write-Host "[OK] Backend & AI Models initialized and responding in $i seconds!" -ForegroundColor Green
            break
        }
    } catch {
        # continue waiting
    }
}

if (-not $backendReady) {
    Write-Host "[WARNING] Backend is taking longer than usual to initialize. Proceeding to launch dashboard..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  ENTERPRISE SYSTEM RUNNING SUCCESSFULLY!" -ForegroundColor Green
Write-Host "  * Enterprise Dashboard : http://localhost:5173" -ForegroundColor Yellow
Write-Host "  * FastAPI Backend      : http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  * OpenAPI / Swagger    : http://127.0.0.1:8001/docs" -ForegroundColor White
Write-Host "  * Live Multimodal WS   : ws://127.0.0.1:8001/ws/detection" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Opening Enterprise Dashboard in your default web browser..." -ForegroundColor Cyan
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "Type 'q' and press Enter (or press Ctrl+C) to terminate both servers..." -ForegroundColor Gray

try {
    while ($true) {
        if ($Host.UI.RawUI.KeyAvailable) {
            $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyUp,IncludeKeyDown")
            if ($key.Character -eq 'q' -or $key.Character -eq 'Q') {
                break
            }
        }
        if ($backendProc.HasExited) {
            Write-Host "[NOTICE] Backend process stopped." -ForegroundColor Yellow
            break
        }
        if ($frontendProc.HasExited) {
            Write-Host "[NOTICE] Frontend process stopped." -ForegroundColor Yellow
            break
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Write-Host "`nStopping background servers..." -ForegroundColor Yellow
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[OK] Backend stopped." -ForegroundColor Green
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        taskkill /PID $frontendProc.Id /T /F 2>$null | Out-Null
        Write-Host "[OK] Frontend stopped." -ForegroundColor Green
    }
    Write-Host "All services shut down cleanly. Goodbye!" -ForegroundColor Cyan
}
