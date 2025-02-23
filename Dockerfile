# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create directories for data persistence
RUN mkdir -p /app/data/uploads /app/data/chromadb

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in virtual environment
RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# RUN pip install llama-index --upgrade --no-cache-dir --force-reinstall
RUN pip install smart-llm-loader==0.1.0

# Copy the application code
COPY . .

# Set environment variables for timeouts
ENV TIMEOUT=900
ENV KEEP_ALIVE_TIMEOUT=120

# Expose the port the app runs on
EXPOSE 8003

# Command to run the application with standard asyncio loop
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-keep-alive", "120", "--loop", "asyncio"]
