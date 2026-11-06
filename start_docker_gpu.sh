#!/usr/bin/env bash
set -e
echo "============================================================"
echo "  STARTING ENTERPRISE PPE DETECTION (DOCKER - NVIDIA GPU)"
echo "  NVIDIA CUDA 12.1 Hardware Acceleration Enabled"
echo "  Frontend: http://localhost:5173  (or http://localhost)"
echo "  Backend:  http://localhost:8001/api/health"
echo "  Database: localhost:5432 (PostgreSQL)"
echo "============================================================"
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
