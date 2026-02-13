# Multi-stage Dockerfile for Research Assistant
# For production, use docker-compose.yml with backend/ and frontend/ Dockerfiles instead

FROM python:3.13.7-slim as base

WORKDIR /app
COPY . /app/

# Install dependencies for both backend and frontend
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

# Default to running the backend (override with docker-compose for dual service setup)
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]