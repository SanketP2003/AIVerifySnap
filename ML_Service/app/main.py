"""
AIVerifySnap ML Service - FastAPI Application.

This service provides deepfake detection capabilities using a dual-stream
hybrid neural network (AIVerifyNet) that combines:
1. Spatial Analysis: RGB image features via ResNet-50
2. Frequency Analysis: ELA (Error Level Analysis) features via ResNet-18

The API is designed to communicate with the Spring Boot backend and can be
deployed as a Docker container.

Endpoints:
- POST /predict: Upload an image file for deepfake detection
- POST /predict-base64: Submit base64-encoded image for detection
- GET /health: Service health check
- GET /model-info: Model architecture and status information
"""

import base64
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import settings
from app.model import DetectionModel
from app.preprocess import load_image_from_bytes, resize_max_side
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)

# Initialize detection model with dependency injection pattern
# Model is loaded once at startup, not on every request
model = DetectionModel(
    backend=settings.model_backend,
    model_path=settings.model_path,
    allow_untrained=settings.allow_untrained,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Loads the ML model at startup (dependency injection pattern).
    This ensures the model is loaded only once and reused across all requests,
    avoiding the overhead of loading on every request.
    """
    print(f"🚀 Starting AIVerifySnap ML Service...")
    print(f"   Backend: {settings.model_backend}")
    print(f"   Model Path: {settings.model_path}")
    print(f"   Allow Untrained: {settings.allow_untrained}")

    model.load()

    print(f"✅ Model loaded successfully!")
    print(f"   Status: {model.model_status}")
    print(f"   Model Loaded: {model.model_loaded}")

    yield

    print("👋 Shutting down AIVerifySnap ML Service...")


# Create FastAPI application with metadata
app = FastAPI(
    title="AIVerifySnap ML Service",
    description=(
        "AI-powered deepfake detection service using a dual-stream hybrid network. "
        "Combines spatial (RGB) and frequency (ELA) analysis for robust detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Spring Boot backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Utility Routes
# =============================================================================

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> JSONResponse:
    """Return empty response for favicon requests."""
    return JSONResponse(content={}, status_code=204)


# =============================================================================
# Health & Status Routes
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
    description="Check if the service is running and the model is loaded."
)
def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="ok" if model.model_loaded else "degraded",
        model_backend=settings.model_backend,
        model_loaded=model.model_loaded,
        model_status=model.model_status,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Health"],
    summary="Model information",
    description="Get detailed information about the loaded model architecture."
)
def model_info() -> ModelInfoResponse:
    """Return model architecture and configuration information."""
    return ModelInfoResponse(
        model_backend=settings.model_backend,
        model_path=settings.model_path,
        model_loaded=model.model_loaded,
        model_status=model.model_status,
        architecture="AIVerifyNet Dual-Stream Hybrid Network",
        spatial_stream="ResNet-50 (RGB analysis)",
        frequency_stream="ResNet-18 (ELA analysis)",
        input_size="224x224",
        supported_formats=["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"],
    )


# =============================================================================
# Prediction Routes
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Detection"],
    summary="Detect deepfake in uploaded image",
    description=(
        "Upload an image file to analyze for AI-generated or manipulated content. "
        "The service uses a dual-stream neural network combining RGB spatial analysis "
        "and ELA (Error Level Analysis) frequency analysis for robust detection."
    ),
)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    """
    Analyze an uploaded image for deepfake detection.

    The detection pipeline:
    1. Validate and load the uploaded image
    2. Generate ELA representation for frequency analysis
    3. Preprocess both RGB and ELA images (resize to 224x224, normalize)
    4. Run dual-stream inference through AIVerifyNet
    5. Return prediction with confidence score

    Args:
        file: Image file to analyze (JPEG, PNG, etc.)

    Returns:
        PredictResponse with detection results
    """
    start = time.time()

    # Validate file type
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Could not determine file type. Please upload a valid image."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )

    # Read file content
    try:
        image_bytes = file.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(exc)}"
        ) from exc

    return _predict_bytes(
        image_bytes=image_bytes,
        start_time=start,
        filename=file.filename or "uploaded_image"
    )


@app.post(
    "/predict-base64",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid base64 image"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Detection"],
    summary="Detect deepfake in base64-encoded image",
    description=(
        "Submit a base64-encoded image for deepfake detection. "
        "Useful for direct API integration without file upload."
    ),
)
def predict_base64(payload: Dict[str, str]) -> PredictResponse:
    """
    Analyze a base64-encoded image for deepfake detection.

    Args:
        payload: Dictionary with 'image_base64' key containing the encoded image

    Returns:
        PredictResponse with detection results
    """
    start = time.time()

    if "image_base64" not in payload:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'image_base64'"
        )

    try:
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        base64_string = payload["image_base64"]
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding: {str(exc)}"
        ) from exc

    return _predict_bytes(
        image_bytes=image_bytes,
        start_time=start,
        filename="base64_image"
    )


def _predict_bytes(
    image_bytes: bytes,
    start_time: float,
    filename: str
) -> PredictResponse:
    """
    Internal function to run prediction on image bytes.

    Args:
        image_bytes: Raw image bytes
        start_time: Request start timestamp for timing
        filename: Original filename for response

    Returns:
        PredictResponse with detection results
    """
    # Load and validate image
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(exc)}. Please upload a valid image file."
        ) from exc

    # Resize to maximum dimension for processing efficiency
    image = resize_max_side(image, settings.max_image_side)

    # Run model prediction
    try:
        result = model.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc

    # Calculate total elapsed time
    elapsed_ms = int((time.time() - start_time) * 1000)

    return PredictResponse(
        filename=filename,
        prediction=result.get("prediction", "Unknown"),
        is_deepfake=result["is_deepfake"],
        confidence=result["confidence"],
        raw_score=result.get("raw_score", result["confidence"]),
        processing_time_ms=result.get("processing_time_ms", elapsed_ms),
        elapsed_ms=elapsed_ms,
        model_status=result["model_status"],
        details=result.get("details", {}),
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "detail": exc.detail,
            "status_code": exc.status_code,
        }
    )


@app.exception_handler(Exception)
def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500,
        }
    )

