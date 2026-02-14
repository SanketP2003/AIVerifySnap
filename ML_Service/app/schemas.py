"""
Pydantic Schemas for AIVerifySnap ML Service API.

These schemas define the request/response models for the FastAPI endpoints,
ensuring type safety and automatic documentation generation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        default="ok",
        description="Service status: 'ok' if healthy, 'degraded' if model not loaded"
    )
    model_backend: str = Field(
        description="Active model backend (aiverifynet, torchscript, or stub)"
    )
    model_loaded: bool = Field(
        description="Whether the ML model is successfully loaded"
    )
    model_status: str = Field(
        default="unknown",
        description="Detailed model loading status"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "model_backend": "aiverifynet",
                "model_loaded": True,
                "model_status": "aiverifynet_mock (untrained)"
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Model architecture information response."""

    model_backend: str = Field(description="Active model backend")
    model_path: str = Field(description="Path to model weights file")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_status: str = Field(description="Detailed model status")
    architecture: str = Field(description="Model architecture name")
    spatial_stream: str = Field(description="Spatial stream architecture")
    frequency_stream: str = Field(description="Frequency stream architecture")
    input_size: str = Field(description="Expected input image size")
    supported_formats: List[str] = Field(description="Supported image formats")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_backend": "aiverifynet",
                "model_path": "models/aiverifynet.pth",
                "model_loaded": True,
                "model_status": "aiverifynet_trained",
                "architecture": "AIVerifyNet Dual-Stream Hybrid Network",
                "spatial_stream": "ResNet-50 (RGB analysis)",
                "frequency_stream": "ResNet-18 (ELA analysis)",
                "input_size": "224x224",
                "supported_formats": ["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"]
            }
        }
    }


class PredictResponse(BaseModel):
    """
    Deepfake detection prediction response.

    Contains the prediction result along with confidence scores,
    processing time, and additional detection details.
    """

    filename: str = Field(
        default="uploaded_image",
        description="Name of the analyzed file"
    )
    prediction: str = Field(
        description="Prediction result: 'Real' or 'Fake'"
    )
    is_deepfake: bool = Field(
        description="Boolean indicating if deepfake was detected"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction (0.0-1.0)"
    )
    raw_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Raw model output score before thresholding"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Model inference time in milliseconds"
    )
    elapsed_ms: Optional[int] = Field(
        default=None,
        description="Total request processing time in milliseconds"
    )
    model_status: str = Field(
        description="Model status used for this prediction"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional detection details (ELA statistics, backend info)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "suspect_image.jpg",
                "prediction": "Fake",
                "is_deepfake": True,
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
        }
    }


class ErrorResponse(BaseModel):
    """Error response model for API errors."""

    error: bool = Field(
        default=True,
        description="Indicates this is an error response"
    )
    detail: str = Field(
        description="Human-readable error message"
    )
    status_code: int = Field(
        description="HTTP status code"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": True,
                "detail": "Invalid file type: text/plain. Please upload an image file.",
                "status_code": 400
            }
        }
    }


class Base64ImageRequest(BaseModel):
    """Request model for base64-encoded image prediction."""

    image_base64: str = Field(
        description="Base64-encoded image data (with or without data URL prefix)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
            }
        }
    }

