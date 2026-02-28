#!/usr/bin/env bash
# Run the Multimodal Image-Text Retrieval System
set -e
cd "$(dirname "$0")"

# Free port 8000 if a stale process holds it
if lsof -i :8000 -t >/dev/null 2>&1; then
    echo "Port 8000 in use. Killing existing process..."
    kill $(lsof -t -i :8000) 2>/dev/null || true
    sleep 2
fi

echo "Starting API on http://localhost:8000 ..."
echo "Wait for 'Application startup complete' before using the frontend."
echo ""
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
