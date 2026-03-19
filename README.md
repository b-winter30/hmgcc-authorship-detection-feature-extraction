# Stylometric Feature Extraction & Authorship Detection API

A multilingual stylometric feature extraction and authorship detection service. Analyses email content using pure stylometry to predict authorship, detect phishing attempts, identify tonal and semantic outliers, and visualise author clusters.

## Prerequisites

- **Docker** (v20.10+) and optionally **Docker Compose** (v2.0+)
- **NVIDIA GPU** with CUDA support (recommended for model inference)
- **NVIDIA Container Toolkit** installed if using GPU — see the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

### Option 1: Docker Build & Run

Build the image:

```bash
docker build -t authorship-detection-fe .
```

Run **with GPU** (recommended):

```bash
docker run -d \
  --name authorship-detection-fe \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  authorship-detection-fe
```

Run **without GPU** (CPU-only, slower inference):

```bash
docker run -d \
  --name authorship-detection-fe \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  authorship-detection-fe
```

Verify the service is running:

```bash
curl http://localhost:8000/health
```

### Option 2: Docker Compose

From the **parent directory** containing this project as a subdirectory, create a `docker-compose.yml`:

```yaml
services:
  feature-extraction-gpu0:
    build:
      context: ./hmgcc-authorship-detection-feature-extraction
      dockerfile: Dockerfile
    container_name: authorship-detection-fe-gpu0
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./hmgcc-authorship-detection-feature-extraction:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    networks:
      - authorship-network

networks:
  authorship-network:
    driver: bridge
```

To run on **multiple GPUs**, duplicate the service block with a different GPU ID and port:

```yaml
services:
  feature-extraction-gpu0:
    build:
      context: ./hmgcc-authorship-detection-feature-extraction
      dockerfile: Dockerfile
    container_name: authorship-detection-fe-gpu0
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./hmgcc-authorship-detection-feature-extraction:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    networks:
      - authorship-network

  feature-extraction-gpu1:
    build:
      context: ./hmgcc-authorship-detection-feature-extraction
      dockerfile: Dockerfile
    container_name: authorship-detection-fe-gpu1
    ports:
      - "8001:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - ./hmgcc-authorship-detection-feature-extraction:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    networks:
      - authorship-network

networks:
  authorship-network:
    driver: bridge
```

Start the service(s):

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f feature-extraction-gpu0
```

Stop everything:

```bash
docker compose down
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) the container can see (e.g. `0`, `0,1`) | All available |
| `PYTHONUNBUFFERED` | Disable Python output buffering for real-time logs | — |

## API Usage

Once the service is running, the API is available at `http://localhost:8000`. A few example requests to get started:

**Health check:**

```bash
curl http://localhost:8000/health
```

**Extract features from an email:**

```bash
curl -X POST http://localhost:8000/extract-features \
  -H "Content-Type: application/json" \
  -d '{"content": "Hi team, please find the report attached. Let me know your thoughts.", "language": "auto"}'
```

**Predict authorship:**

```bash
curl -X POST http://localhost:8000/predict-author \
  -H "Content-Type: application/json" \
  -d '{"content": "Hi team, please find the report attached. Let me know your thoughts."}'
```

**Detect phishing:**

```bash
curl -X POST http://localhost:8000/phishing-detection \
  -H "Content-Type: application/json" \
  -d '{"content": "URGENT: Your account has been compromised. Click here immediately to verify your identity or your account will be permanently suspended within 24 hours."}'
```

For full endpoint documentation, see [api-documentation.md](api-documentation.md).

## Project Structure

```
.
├── main.py                 # Application entrypoint
├── endpoints.py            # Route definitions
├── feature_extractor.py    # Stylometric feature extraction logic
├── models.py               # Pydantic request/response schemas
├── helpers.py              # Utility functions
├── dependencies.py         # Dependency injection
├── models/                 # Pre-trained ML model files
├── pyproject.toml          # Python project metadata & dependencies
├── uv.lock                 # Locked dependency versions
└── Dockerfile
```

## Troubleshooting

**Container exits immediately** — check the logs with `docker logs authorship-detection-fe-gpu0`. A common cause is missing model files in the `models/` directory.

**GPU not detected** — ensure the NVIDIA Container Toolkit is installed and the Docker daemon has been restarted. Verify with `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.

**Port already in use** — change the host port mapping (e.g. `-p 8001:8000` or update the `ports` value in `docker-compose.yml`).