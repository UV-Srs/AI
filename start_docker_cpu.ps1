# =============================================================================
# Run PPE Detection & Face Recognition System in Docker (CPU Mode)
# =============================================================================
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  STARTING ENTERPRISE PPE DETECTION (DOCKER - CPU MODE)" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5173  (or http://localhost)" -ForegroundColor Yellow
Write-Host "  Backend:  http://localhost:8001/api/health" -ForegroundColor Yellow
Write-Host "  Database: localhost:5432 (PostgreSQL)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

docker-compose up --build
