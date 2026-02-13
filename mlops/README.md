# MLOps for ResearchAssistant

This project has lightweight MLOps features to track data ingestion and metrics.

What was added:

- MLflow tracking server (local) via `docker-compose` service `mlflow`.
- MLflow instrumentation in `backend/upload` endpoint to record runs, params and metrics.
- `prometheus-client` metric `upload_chunks_total` incremented when uploads succeed.
- CI workflow `/.github/workflows/ci.yml` to run a quick import test and build Docker images.

How to run locally:

1. Ensure `.env` contains your keys (OpenAI, Pinecone, etc.).
2. Start services:

```bash
docker-compose up --build
```

3. MLflow UI will be available at: http://localhost:5000
4. Backend API: http://localhost:8000
5. Streamlit UI: http://localhost:8501

Notes & next steps:

- For production, use a persistent artifact store (S3 / GCS) for MLflow artifacts.
- Add Prometheus scraping for metrics and a Grafana dashboard for visualization.
- Optionally add model evaluation runs and register models to MLflow Model Registry.
