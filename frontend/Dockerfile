# Frontend Streamlit Dockerfile
FROM python:3.13.7-slim

WORKDIR /app/frontend

# Copy requirements and install dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend/ .

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit (disable telemetry for cleaner output)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=info", "--client.showErrorDetails=true"]
