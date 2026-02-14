# AIVerifySnap ML Service

Production-ready FastAPI microservice for AI-powered deepfake detection using a **Dual-Stream Hybrid Neural Network** (AIVerifyNet).

## 🏗️ Architecture

AIVerifySnap uses a sophisticated dual-stream architecture combining spatial and frequency analysis:

```
┌─────────────────┐     ┌─────────────────┐
│  RGB Image      │     │   ELA Image     │
│  (224x224x3)    │     │   (224x224x3)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   ResNet-50     │     │   ResNet-18     │
│   (Spatial)     │     │   (Frequency)   │
│   2048 features │     │   512 features  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Fusion Layer   │
          │  FC → Real/Fake │
          └─────────────────┘
```

### Key Components

1. **Spatial Stream (RGB)**: Analyzes raw images using pre-trained ResNet-50 for semantic features
2. **Frequency Stream (ELA)**: Detects compression artifacts using Error Level Analysis + ResNet-18
3. **Fusion Layer**: Combines both streams for robust classification

## 🚀 Features

- **Dual-Stream Detection**: Combines RGB spatial analysis with ELA frequency analysis
- **Mock Mode**: Run without trained weights for development/testing
- **Multiple Backends**: AIVerifyNet, TorchScript, or heuristic fallback
- **Docker Ready**: Production Dockerfiles for CPU and GPU deployment
- **Spring Boot Integration**: CORS configured for backend communication
- **Comprehensive API**: REST endpoints with Swagger documentation

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload image file for detection |
| `POST` | `/predict-base64` | Submit base64-encoded image |
| `GET` | `/health` | Service health check |
| `GET` | `/model-info` | Model architecture information |
| `GET` | `/docs` | Swagger API documentation |

## 🏃 Quick Start

### Local Development

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

### Docker Deployment

```bash
# Build and run (CPU)
docker-compose up -d ml-service

# Build and run (GPU with NVIDIA)
docker-compose --profile gpu up -d ml-service-gpu
```

## 📦 Project Structure

```
ML_Service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # Detection model wrapper
│   ├── aiverifynet.py       # Dual-stream neural network
│   ├── ela_utils.py         # ELA computation utilities
│   ├── preprocess.py        # Image preprocessing
│   ├── schemas.py           # Pydantic models
│   └── config.py            # Configuration settings
├── models/
│   └── README.md            # Model weights directory
├── tests/
│   └── test_api.py          # API tests
├── Dockerfile               # CPU Docker image
├── Dockerfile.gpu           # GPU Docker image
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Production dependencies
└── requirements-dev.txt     # Development dependencies
```

## ⚙️ Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `aiverifynet` | Model backend: `aiverifynet`, `torchscript`, `stub` |
| `MODEL_PATH` | `models/aiverifynet.pth` | Path to trained weights |
| `ALLOW_UNTRAINED` | `1` | Allow mock mode without weights |
| `DEVICE` | `auto` | PyTorch device: `cuda`, `cpu`, `auto` |
| `MAX_IMAGE_SIDE` | `1024` | Max image dimension |
| `CORS_ORIGINS` | `localhost:*` | Allowed CORS origins |
| `ELA_QUALITY` | `90` | JPEG quality for ELA |
| `CONFIDENCE_THRESHOLD` | `0.5` | Classification threshold |

## 📝 Example Requests

### Upload Image File

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test_image.jpg"
```

### Base64 Encoded Image

```python
import base64
import requests

with open('test_image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('ascii')

response = requests.post(
    'http://127.0.0.1:8000/predict-base64',
    json={'image_base64': image_b64}
)
print(response.json())
```

### Response Format

```json
{
  "filename": "test_image.jpg",
  "prediction": "Fake",
  "is_deepfake": true,
  "confidence": 0.87,
  "raw_score": 0.87,
  "processing_time_ms": 142.5,
  "elapsed_ms": 156,
  "model_status": "aiverifynet_mock (untrained)",
  "details": {
    "backend": "aiverifynet",
    "device": "cuda:0",
    "ela_mean": 0.023456,
    "ela_std": 0.015234,
    "ela_max_ratio": 12.5432
  }
}
```

## 🧪 Running Tests

```powershell
pip install -r requirements-dev.txt
pytest -v
```

## 🔬 Technical Details

### Error Level Analysis (ELA)

ELA detects manipulation by analyzing JPEG compression artifacts:

1. Resave image at specific quality (90%)
2. Calculate pixel difference from original
3. Scale brightness to highlight artifacts
4. AI-generated regions show different compression patterns

### High-Compression Robustness

The dual-stream approach is robust to:
- Different compression levels
- Image resizing and cropping
- Color space transformations
- Minor post-processing

## 🔗 Spring Boot Integration

The service is designed to work with the AIVerifySnap Spring Boot backend:

```yaml
# In Spring Boot application.yml
ml:
  service:
    url: http://localhost:8000
```

The `DetectionController` forwards requests to this service.

## 📄 License

Part of the AIVerifySnap deepfake detection system.## Notes
- The default `stub` backend uses simple ELA and high-frequency heuristics and is not a production detector.
- Replace with a trained model to meet accuracy targets described in the document.

