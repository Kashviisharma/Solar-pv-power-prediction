FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Create necessary directories
RUN mkdir -p data/processed models

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Default to running the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
