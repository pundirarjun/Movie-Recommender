# ---- Stage 1: build React frontend ----
FROM node:20-alpine AS frontend

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---- Stage 2: Python runtime + API ----
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY backend/ ./backend/
COPY src/ ./src/
COPY data/movies.csv ./data/movies.csv
COPY model_weights.npz movie_popularity.npy item_item_sim.npz ./

COPY --from=frontend /build/dist ./frontend/dist

EXPOSE 8080

WORKDIR /app/backend

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "180", "--graceful-timeout", "30", "app:app"]