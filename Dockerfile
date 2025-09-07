FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code. NOTE: copy your model files into the container build context before building.
COPY app.py /app/app.py
# These two files must exist in the build context (same folder) when you run `docker build`:
COPY fraud_model_rf.pkl /app/fraud_model_rf.pkl
COPY scaler_params.json /app/scaler_params.json

# If you prefer to mount model files at runtime instead, comment the COPY lines above,
# and mount a volume with -v "$(pwd)":/app when running.

EXPOSE 8080

# Use env PORT if provided by the platform (Render/Railway), default 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]