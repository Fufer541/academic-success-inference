# Academic Success Inference

A full-stack ML web service that predicts whether a student will **graduate** or **drop out**, built with a CatBoost + LightGBM ensemble and a React frontend.

**Model metrics:** 79% accuracy · 86% ROC AUC · 77% Macro F1

---

## Stack

| Layer | Technology |
|---|---|
| ML | CatBoost, LightGBM, scikit-learn, pandas, numpy |
| Backend | Python 3.12, FastAPI, Uvicorn |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| Infra | Docker, Docker Compose, Nginx |

---

## How it works

1. **Research** — `research/academic_success_modeling.ipynb` trains a binary ensemble on the [UCI Student Dropout dataset](https://zenodo.org/record/5777340). Enrolled students with no known outcome are pseudo-labeled at 95% confidence threshold and added back into training (semi-supervised step).
2. **Backend** — FastAPI loads the trained `.pkl` bundle and exposes three endpoints: `/health`, `/models`, `/predict`.
3. **Frontend** — React fetches model metadata from `/models` and dynamically renders the input form. No field names are hardcoded in the UI.

---

## Dataset

**Predict Students' Dropout and Academic Success**
Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021).
Zenodo. [https://doi.org/10.5281/zenodo.5777340](https://doi.org/10.5281/zenodo.5777340)
License: CC BY 4.0

---

## Prerequisites

- Python `3.12+`
- Node.js `v24+` (recommended via `nvm`)
- Docker + Docker Compose (optional)

---

## Quickstart

### 1. Train the model

```bash
cd research
pip install jupyter ipykernel seaborn optuna
pip install -r ../backend/requirements.txt
jupyter nbconvert --to notebook --execute academic_success_modeling.ipynb \
  --output academic_success_modeling_executed.ipynb
```

This saves `research/pickles/academic_success_enrollment_catboost_lightgbm_bundle.pkl`.

### 2. Run the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`

Check health:
```bash
curl http://localhost:8000/health
```

### 3. Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 4. Alternative: Docker Compose

```bash
# Add volumes to docker-compose.yml backend service first:
# volumes:
#   - ./research/pickles:/models

docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`

---

## Project structure

```
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, endpoints
│   │   └── model_service.py # Bundle loading, inference
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/             # Backend API calls
│   │   ├── components/      # AppShell — main UI
│   │   ├── config/          # Env config, category labels
│   │   └── types/           # Shared TypeScript types
│   └── Dockerfile
├── research/
│   ├── academic_success_modeling.ipynb
│   ├── data/
│   │   └── academic_success.csv
│   └── pickles/             # .pkl files (git-ignored)
├── docs/
└── docker-compose.yml
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health + model status |
| GET | `/models` | Model metadata + feature list |
| POST | `/predict` | Run inference on one student record |

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "academic-success-terminal-binary-ensemble-v1",
    "features": {
      "Application mode": 17,
      "Course": 171,
      "Admission grade": 127.3,
      "Age at enrollment": 20,
      "Tuition fees up to date": 1,
      "Scholarship holder": 0,
      "Gender": 1,
      "Unemployment rate": 10.8,
      "Inflation rate": 1.4,
      "GDP": 1.74
    }
  }'
```

### Example response

```json
{
  "model": "academic-success-terminal-binary-ensemble-v1",
  "prediction": "Graduate",
  "confidence": 0.84,
  "probabilities": {
    "Dropout": 0.16,
    "Graduate": 0.84
  },
  "modelVersion": "academic-success-terminal-binary-ensemble-v1-enrollment"
}
```

---

## Author

**Musheg Tovmasian**
Built as a portfolio project for university transfer applications.
LAVC, 2025