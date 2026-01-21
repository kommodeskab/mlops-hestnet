# docker builds -f api.dockerfile . -t gcp_test_app:latest
# docker run -p 8080:8080 -e PORT=8080 gcp_test_app:latest
# gcloud buils submit . --project dtumlops-484208
# Lightweight Python base image
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies + certificates
RUN apt-get update && apt-get install -y \
    build-essential libffi-dev libssl-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /api

# Copy only requirements first (for Docker cache efficiency)
COPY requirements_api.txt ./requirements_api.txt

RUN python -m pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements_api.txt

# Download model
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('distilgpt2'); AutoTokenizer.from_pretrained('distilgpt2')"

# Copy the entire api folder
COPY . .

# Expose port if running a web server (Cloud Run default)
EXPOSE $PORT

# Run the application
CMD exec uvicorn api:app --port $PORT --host 0.0.0.0 --workers 1
