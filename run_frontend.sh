#!/usr/bin/env bash
# Run Streamlit frontend (start API first with ./run.sh)
# Usage: ./run_frontend.sh [API_PORT]  e.g. ./run_frontend.sh 8001
set -e
cd "$(dirname "$0")"

if [[ -n "$1" && "$1" =~ ^[0-9]+$ ]]; then
    export RETRIEVAL_API_PORT="$1"
    echo "Using API at http://localhost:$1"
else
    echo "Using API at http://localhost:8000 (override: ./run_frontend.sh 8001)"
fi
echo "Starting Streamlit on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501
