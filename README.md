# CineMatch — Movie Recommendation Engine

**Live demo:** https://movierecommender-arjun.fly.dev/

A full-stack movie recommendation system combining **Matrix Factorization** (64 latent factors) with **item-item collaborative filtering**, served through a Flask API and a React frontend with TMDB posters and YouTube trailers.

---

## Demo

| | |
|---|---|
| **Frontend** | React 19 + Vite, mood/genre/movie-similarity inputs |
| **Backend** | Flask + Gunicorn, hybrid scoring pipeline |
| **Data** | MovieLens (87,585 movies, 84,432 in model) |
| **Deployed on** | Fly.io (Singapore region, Docker) |

Try it: pick a mood (`Something popular`, `Mind-bending`, `Something scary`…), select genres, or enter movies you like (e.g. *The Matrix*, *Inception*) — get 10 ranked recommendations with posters, ratings, and trailers.

---

## Model

**Hybrid scoring:**

```
final_score = α · MF(user, item) + (1 - α) · normalized_item_item_CF(item)
```

- **Matrix Factorization** — SGD-trained, 64 latent factors, biased MF
- **Item-item CF** — cosine similarity over the rating matrix (sparse CSR)
- **Popularity prior** — blends in for cold-start users

### Honest evaluation

Trained/validated/tested on an **80/10/10 split** (no train-on-test leakage):

| Metric | Value |
|---|---|
| Val RMSE | **0.7618** |
| Precision@10 | 0.0425 |
| Recall@10 | 0.0239 |
| NDCG@10 | 0.0508 |

> Precision/Recall/NDCG are computed against each user's held-out test ratings — standard for MovieLens-scale data where users rate thousands of items but interact with few in any top-10 list.

Training curve: [`Training vs Validation RMSE plot.png`](./Training%20vs%20Validation%20RMSE%20plot.png)

---

## Tech stack

**ML / Backend**
- Python 3.12, NumPy, SciPy (sparse matrices), Numba, pandas
- Flask + Flask-CORS, Gunicorn
- TMDB API (posters, metadata, YouTube trailers)

**Frontend**
- React 19, Vite
- No UI framework — custom CSS

**Infra**
- Docker (multi-stage: Node build → Python runtime)
- Fly.io, GitHub Actions-ready (`flyctl deploy`)

---

## Project structure

```
├── backend/
│   ├── app.py              # Flask API + serves built frontend
│   └── recommender.py      # model loading, hybrid scoring, TMDB fetch
├── frontend/
│   └── src/App.jsx         # React UI
├── src/
│   ├── model.py            # MF model (biased SGD)
│   ├── train_split.py      # 80/10/10 split + training loop
│   ├── item_item_cf.py     # item-item CF + hybrid_scores()
│   ├── evaluate_honest.py  # Precision@k, Recall@k, NDCG@k, RMSE
│   └── tune_hyperparams.py # factor/regularization grid search
├── data/                   # movies.csv (ratings.csv gitignored, 836MB)
├── model_weights.npz       # trained MF (U, V, biases)
├── item_item_sim.npz       # item-item cosine similarity
├── movie_popularity.npy    # popularity prior
├── Dockerfile
└── fly.toml
```

---

## Local setup

```bash
# 1. Backend
pip install -r requirements.txt
cp .env.example .env          # add your TMDB_API_KEY
python backend/app.py         # http://127.0.0.1:5000

# 2. Frontend (separate terminal)
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

### Retrain the model

```bash
python src/train_split.py       # trains MF, saves weights + plots
python src/item_item_cf.py      # builds item-item similarity matrix
python src/evaluate_honest.py   # honest test-set metrics
```

---

## API endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | status, model size |
| POST | `/recommend` | `{movies, genres, mood}` | 10 recommendations + posters/trailers |
| GET | `/search?q=` | — | movie title search |
| GET | `/` | — | built React frontend |

---

## Deployment

```bash
flyctl secrets set TMDB_API_KEY=<your-key> --app movierecommender-arjun
flyctl deploy --remote-only
```

Multi-stage Docker build: Node compiles the React app, Python slim image runs Gunicorn serving both the API and static frontend on port 8080.

---

## License

MIT
