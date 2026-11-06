#!/usr/bin/env bash
set -e
echo "============================================================"
echo "  STARTING ENTERPRISE PPE DETECTION (DOCKER - CPU MODE)"
echo "  Frontend: http://localhost:5173  (or http://localhost)"
echo "  Backend:  http://localhost:8001/api/health"
echo "  Database: localhost:5432 (PostgreSQL)"
echo "============================================================"
docker-compose up --build
