# Dockerfile
# ----------
# Builds the PRODUCTION API (api/ package) only.
# The debug UI (app.py + static/) is local-only and is NOT included here.
#
# Model weights are NOT baked into this image — they are large (4+ GB) and
# change independently from the code. Provide them at runtime via one of:
#   A) AWS EFS volume mounted at /app/models/
#   B) S3 download on container startup (see ENTRYPOINT comment below)
#   C) Set SUMMARIZER_MODEL_PATH / CLASSIFIER_MODEL_PATH to any valid path.
#
# Build:
#   docker build -t context-service:latest .
#
# Run (with models already on host at ./models/):
#   docker run -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -e SUMMARIZER_MODEL_PATH=/app/models/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf \
#     -e CLASSIFIER_MODEL_PATH=/app/models/google_gemma-3n-E2B-it-Q4_K_M.gguf \
#     context-service:latest

FROM python:3.11-slim

# Install system build tools needed by llama-cpp-python's C++ compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer-cached unless requirements change).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the packages and modules needed for the production API.
# The debug UI (app.py, static/) is intentionally excluded.
COPY summarizer/ summarizer/
COPY classifier/ classifier/
COPY api/        api/

# PORT is read by api/config.py; default matches the EXPOSE below.
ENV PORT=8000
EXPOSE 8000

# Health check — poll /v1/health until models_loaded is true.
# Start period of 120s gives the models time to load before Docker marks
# the container unhealthy.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c \
        "import urllib.request, json, sys; \
         r = urllib.request.urlopen('http://localhost:${PORT:-8000}/v1/health', timeout=4); \
         d = json.loads(r.read()); \
         sys.exit(0 if d.get('models_loaded') else 1)"

# Use sh -c so ${PORT} is expanded at runtime from the container's env.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
