FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for psycopg2 and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create models directory
RUN mkdir -p /app/ml_models

EXPOSE 8000

RUN mkdir -p /app/ml_models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
