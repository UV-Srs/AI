# =============================================================================
# Run PPE Detection & Face Recognition System in Docker (NVIDIA GPU Mode)
# =============================================================================
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  STARTING ENTERPRISE PPE DETECTION (DOCKER - NVIDIA GPU)" -ForegroundColor Green
Write-Host "  NVIDIA CUDA 12.1 Hardware Acceleration Enabled" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5173  (or http://localhost)" -ForegroundColor Yellow
Write-Host "  Backend:  http://localhost:8001/api/health" -ForegroundColor Yellow
Write-Host "  Database: localhost:5432 (PostgreSQL)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
