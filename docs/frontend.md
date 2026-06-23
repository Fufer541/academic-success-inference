# Frontend

Minimal React + TypeScript + Vite scaffold for building a Kaggle-style ML prediction web UI.

## Quick Start

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## Environment

- `VITE_API_BASE_URL`: Base URL for your backend API (default in code: `http://localhost:8000`)

## Project Structure

```text
src/
  api/         # Backend-facing API modules
  components/  # Reusable UI building blocks
  config/      # Environment and runtime config
  lib/         # Shared utilities (HTTP, helpers)
  index.css    # Tailwind entrypoint + base layer
  types/       # Shared TypeScript contracts
```

## Styling

- Tailwind CSS v3 (`tailwindcss@^3.4`) with standard PostCSS setup
- Config files:
  - `tailwind.config.js`
  - `postcss.config.js`

## Docker

Build image:

```bash
cd frontend
docker build -t ml-frontend --build-arg VITE_API_BASE_URL=http://localhost:8000 .
```

Run container:

```bash
docker run --rm -p 8080:80 ml-frontend
```

The app is served by Nginx and available at `http://localhost:8080`.

## Docker Compose

From repository root:

```bash
docker compose up --build
```

This brings up:

- Frontend on `http://localhost:8080`
- Backend API on `http://localhost:8000`

## Scaffolded API Contracts

- `src/api/predictions.ts`
  - `getModels()` -> `GET /models`
  - `runPrediction()` -> `POST /predict`
- `src/types/prediction.ts`
  - Request/response and model metadata types

Adjust endpoint paths and response types to match your backend implementation.
