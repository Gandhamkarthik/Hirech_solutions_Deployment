# Multi-stage build for the RAG application
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY streamlit_app_tfidf.py .
COPY main.py .

EXPOSE 8000 8501
COPY data/ ./data/

# This Dockerfile is simplified for a single-container setup or can be used as a base
CMD ["streamlit" ,"run","streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
