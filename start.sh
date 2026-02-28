#!/bin/bash
set -e

# Generate data and index
python scripts/create_sample_data.py --method hf --count 2000
python scripts/index_dataset.py --dataset sample --max-items 2000

# Start FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
