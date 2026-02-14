"""
Configuration Settings for AIVerifySnap ML Service.

Environment variables:
- MODEL_BACKEND: Model backend to use ('aiverifynet', 'torchscript', 'stub')
- MODEL_PATH: Path to trained model weights file
- ALLOW_UNTRAINED: Allow running without trained weights (1=yes, 0=no)
- MAX_IMAGE_SIDE: Maximum image dimension before resizing
- CORS_ORIGINS: Comma-separated list of allowed CORS origins
- DEVICE: PyTorch device ('cuda', 'cpu', or 'auto')
"""

import os
from typing import List


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Model configuration
        # Supported backends: 'aiverifynet' (dual-stream), 'torchscript', 'stub'
        self.model_backend: str = os.getenv("MODEL_BACKEND", "aiverifynet").lower()

        # Path to trained model weights
        # For AIVerifyNet: expects a .pth file with state_dict
        # For TorchScript: expects a .pt compiled model
        self.model_path: str = os.getenv("MODEL_PATH", "models/aiverifynet.pth")

        # Allow running without trained weights (mock mode)
        # Useful for development and testing
        self.allow_untrained: bool = os.getenv("ALLOW_UNTRAINED", "1") == "1"

        # Maximum image dimension for preprocessing
        # Images larger than this will be resized proportionally
        self.max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "1024"))

        # PyTorch device configuration
        # 'auto' will use CUDA if available, otherwise CPU
        self.device: str = os.getenv("DEVICE", "auto")

        # CORS configuration for Spring Boot backend communication
        cors_origins_str = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080,http://localhost:8081"
        )
        self.cors_origins: List[str] = [
            origin.strip() for origin in cors_origins_str.split(",") if origin.strip()
        ]

        # ELA (Error Level Analysis) configuration
        # JPEG quality for ELA computation (higher = more sensitive to manipulations)
        self.ela_quality: int = int(os.getenv("ELA_QUALITY", "90"))

        # Input image size for the neural network
        self.input_size: int = int(os.getenv("INPUT_SIZE", "224"))

        # Confidence threshold for deepfake classification
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"model_backend={self.model_backend!r}, "
            f"model_path={self.model_path!r}, "
            f"allow_untrained={self.allow_untrained}, "
            f"device={self.device!r}, "
            f"max_image_side={self.max_image_side})"
        )


# Global settings instance
settings = Settings()

