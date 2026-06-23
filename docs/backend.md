# Backend

FastAPI + Uvicorn scaffold that matches the frontend contract.

## Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs on `http://localhost:8000`.

## Endpoints

- `GET /health`
  - Response: `{"status":"ok"}`
- `GET /models`
  - Response: list of models (`id`, `name`, `description`)
- `POST /predict`
  - Request:
    - `model`: string
    - `features`: key/value map with string, number, boolean, or null values
  - Response:
    - `prediction`: string or number
    - `confidence`: optional number
    - `modelVersion`: optional string

## Docker

Build image:

```bash
cd backend
docker build -t ml-backend .
```

Run container:

```bash
docker run --rm -p 8000:8000 ml-backend
```

## Docker Compose

From repository root:

```bash
docker compose up --build
```

Backend is exposed on `http://localhost:8000`.
