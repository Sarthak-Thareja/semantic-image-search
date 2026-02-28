# Multimodal Semantic Image–Text Retrieval System

A semantic retrieval system that lets you search images by natural language or by image similarity using CLIP embeddings and vector search.

## Features

- **Text-to-image search** — Find images using natural language (e.g., "dog playing in snow")
- **Image-to-image search** — Upload an image to find similar ones
- **CLIP + ChromaDB** — OpenCLIP ViT-B/32 embeddings, HNSW-based cosine similarity
- **REST API** — FastAPI endpoints for integration
- **Streamlit UI** — Web demo for exploration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Sample Data & Index

```bash
# Download a diverse sample of real images from MS-COCO
python scripts/create_sample_data.py --method hf --count 2000

# Index the downloaded images and captions into the vector DB
python scripts/index_dataset.py --dataset sample --max-items 2000
```

### 3. Start API (Terminal 1)

```bash
./run.sh
# or: uvicorn app.main:app --reload
```

**Wait for "Application startup complete"** (CLIP model load ~30–60s on first start).

API: http://localhost:8000 | Docs: http://localhost:8000/docs

### 4. Start Frontend (Terminal 2)

```bash
./run_frontend.sh
# or: streamlit run frontend/app.py
```

UI: http://localhost:8501

### If Port 8000 Is In Use

```bash
# Run API on port 8001
uvicorn app.main:app --reload --port 8001

# Point frontend to it
RETRIEVAL_API_PORT=8001 streamlit run frontend/app.py
```

## Docker

```bash
# Build and run
docker-compose up -d

# Create sample data and index (run once, with container or locally)
python scripts/create_sample_data.py
python scripts/index_dataset.py --dataset sample --max-items 100
```

## Dataset Setup

### MS-COCO

1. Download [COCO val2017](https://coco-dataset.org/) images and [annotations](https://github.com/cocodataset/cocoapi).
2. Place under `data/`:
   - `data/annotations/captions_val2017.json`
   - `data/images/val2017/` (or `data/val2017/`)

```bash
python scripts/index_dataset.py --data-dir ./data --dataset coco
```

### Flickr30k

1. Download [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) (or use HuggingFace).
2. Place under `data/`:
   - `data/flickr30k_images/` (or `data/images/`)
   - `data/results.csv` (image_name | caption_number | caption)

```bash
python scripts/index_dataset.py --data-dir ./data --dataset flickr30k
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search-text` | Search by text query |
| POST | `/api/v1/search-image` | Search by image upload |
| GET | `/api/v1/health` | Health + index stats |

### Example: Text Search

```bash
curl -X POST http://localhost:8000/api/v1/search-text \
  -H "Content-Type: application/json" \
  -d '{"query": "red square", "top_k": 5}'
```

### Example: Image Search

```bash
curl -X POST http://localhost:8000/api/v1/search-image \
  -F "file=@myimage.jpg" \
  -F "top_k=5"
```

## Run Scripts

| Script | Purpose |
|--------|---------|
| `./run.sh` | Starts API on 8000; frees port if in use |
| `./run_frontend.sh` | Starts Streamlit UI; uses `RETRIEVAL_API_URL` if set |

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI app
│   ├── config.py         # Settings
│   ├── api/
│   │   ├── routes.py     # Endpoints
│   │   └── schemas.py    # Request/response models
│   └── services/
│       ├── encoder.py    # CLIP encoding
│       └── retriever.py  # ChromaDB vector search
├── frontend/
│   └── app.py            # Streamlit UI
├── scripts/
│   ├── index_dataset.py  # Batch indexing
│   ├── load_dataset.py   # COCO/Flickr30k loaders
│   ├── create_sample_data.py
│   └── evaluate.py       # Recall@K, mAP
├── data/                 # ChromaDB + images
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Configuration

Copy `.env.example` to `.env` and set:

- `RETRIEVAL_CHROMA_PERSIST_DIR` — Vector DB path
- `RETRIEVAL_CLIP_MODEL_NAME` — e.g. ViT-B-32
- `RETRIEVAL_CLIP_PRETRAINED` — e.g. openai

## Evaluation

Run retrieval over a test set, then:

```bash
python scripts/evaluate.py --results-file results.json --ground-truth gt.json
```

## License

MIT. See PRD.md for full product requirements.
