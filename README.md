# ml_web_service_template

Template repository for ML web services with:

- Frontend: React + Vite + TypeScript + Tailwind CSS
- Backend: FastAPI + Uvicorn

Documentation:

- Frontend guide: `docs/frontend.md`
- Backend guide: `docs/backend.md`
- Research workspace: `docs/research.md`

## Prerequisites

- Node.js `v24.12.0` (recommended with `nvm`)
- Python `3.12+`
- Docker + Docker Compose (optional, for containerized flow)

## Development Pipeline

### 1) Clone and enter project

```bash
git clone https://github.com/thesecondpioneer/ml_web_service_template.git
cd ml_web_service_template
```

### 2) Start in local dev mode (recommended for coding)

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend (in a second terminal):

```bash
cd frontend
source ~/.nvm/nvm.sh
nvm use v24.12.0
npm install
cp .env.example .env
npm run dev
```

Research:

- Use `research/starter_notebook.ipynb` as your notebook starting point.
- Store generated `.pkl` files in `research/pickles/` (ignored by git).

### 3) Alternative: run full stack with Docker

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`

### 4) Typical project workflow

1. Create a feature branch from `main`.
2. Build your model logic in `backend/` and UI in `frontend/`.
3. Keep API contract aligned (`/models`, `/predict`, `/health`).
4. Run checks before pushing:
   - Frontend: `npm run lint && npm run build`
   - Backend: `python -m compileall backend/app`
5. Commit, push branch, and open a PR.
