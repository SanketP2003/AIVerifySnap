"""
Detection Model Module for AIVerifySnap.

This module provides the main detection interface supporting multiple backends:
1. AIVerifyNet: Production dual-stream hybrid model (RGB + ELA)
2. TorchScript: Pre-compiled PyTorch models for deployment
3. Stub: Heuristic fallback for testing without trained models

The hybrid model architecture combines spatial (RGB) and frequency (ELA) analysis
to achieve robust deepfake detection with high-compression robustness.
"""

import math
import os
import time
from typing import Any, Dict, Optional

from PIL import Image

from app.ela_utils import convert_to_ela, compute_ela_features
from app.preprocess import compute_ela, high_frequency_energy, image_stats

# Conditional imports for PyTorch
try:
    import torch
    import numpy as np
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None
    transforms = None


class DetectionModel:
    """
    Main detection model supporting multiple backends.

    Supports:
    - 'aiverifynet': Full dual-stream hybrid model (recommended for production)
    - 'torchscript': Pre-compiled TorchScript models
    - 'stub': Heuristic-based fallback (no ML required)
    """

    # ImageNet normalization for preprocessing
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        backend: str,
        model_path: str,
        allow_untrained: bool,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the detection model.

        Args:
            backend: Model backend ('aiverifynet', 'torchscript', or 'stub')
            model_path: Path to trained model weights
            allow_untrained: Allow running without trained weights (mock mode)
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.backend = backend.lower()
        self.model_path = model_path
        self.allow_untrained = allow_untrained
        self.model_loaded = False
        self.model_status = "unloaded"
        self._torch_model = None
        self._aiverifynet = None
        self._device = None
        self._transform = None
        self._requested_device = device

    def load(self) -> None:
        """Load the model based on configured backend."""
        if self.backend == "aiverifynet":
            self._load_aiverifynet()
        elif self.backend == "torchscript":
            self._load_torchscript()
        else:
            # Stub mode - no model loading required
            self.model_loaded = True
            self.model_status = "stub"

    def _setup_device(self) -> None:
        """Setup PyTorch device."""
        if not TORCH_AVAILABLE:
            return

        if self._requested_device:
            self._device = torch.device(self._requested_device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_transform(self) -> None:
        """Setup image preprocessing transform."""
        if not TORCH_AVAILABLE or transforms is None:
            return

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def _load_aiverifynet(self) -> None:
        """Load the AIVerifyNet dual-stream model."""
        if not TORCH_AVAILABLE:
            self.model_loaded = False
            self.model_status = "torch_not_installed"
            if not self.allow_untrained:
                raise ImportError("PyTorch is required for AIVerifyNet backend")
            return

        self._setup_device()
        self._setup_transform()

        try:
            from app.aiverifynet import AIVerifyNet

            # Initialize model architecture
            self._aiverifynet = AIVerifyNet(
                pretrained=True,
                dropout_rate=0.3,
                freeze_backbone=False
            )

            # Load trained weights if available
            if os.path.exists(self.model_path):
                try:
                    state_dict = torch.load(self.model_path, map_location=self._device)
                    self._aiverifynet.load_state_dict(state_dict)
                    self.model_status = "aiverifynet_trained"
                except Exception as e:
                    if not self.allow_untrained:
                        raise RuntimeError(f"Failed to load AIVerifyNet weights: {e}")
                    self.model_status = f"aiverifynet_mock (load_error: {str(e)[:30]})"
            else:
                if not self.allow_untrained:
                    raise FileNotFoundError(f"Model weights not found: {self.model_path}")
                self.model_status = "aiverifynet_mock (untrained)"

            self._aiverifynet.to(self._device)
            self._aiverifynet.eval()
            self.model_loaded = True

        except ImportError as e:
            self.model_loaded = False
            self.model_status = f"aiverifynet_import_error: {str(e)[:50]}"
            if not self.allow_untrained:
                raise

    def _load_torchscript(self) -> None:
        """Load a TorchScript compiled model."""
        if not os.path.exists(self.model_path):
            self.model_loaded = False
            self.model_status = f"missing model at {self.model_path}"
            if not self.allow_untrained:
                raise FileNotFoundError(self.model_status)
            return

        if not TORCH_AVAILABLE:
            self.model_loaded = False
            self.model_status = "torch_not_installed"
            if not self.allow_untrained:
                raise ImportError("PyTorch required for TorchScript backend")
            return

        self._setup_device()
        self._setup_transform()

        try:
            self._torch_model = torch.jit.load(self.model_path, map_location=self._device)
            self._torch_model.eval()
            self.model_loaded = True
            self.model_status = "torchscript"
        except Exception as e:
            self.model_loaded = False
            self.model_status = f"torchscript_load_error: {str(e)[:50]}"
            if not self.allow_untrained:
                raise

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run deepfake detection on an image.

        Args:
            image: PIL Image to analyze

        Returns:
            Dictionary containing prediction results
        """
        if self.backend == "aiverifynet" and self._aiverifynet is not None:
            return self._predict_aiverifynet(image)
        elif self.backend == "torchscript" and self._torch_model is not None:
            return self._predict_torchscript(image)
        return self._predict_stub(image)

    def _preprocess_image(self, image: Image.Image) -> 'torch.Tensor':
        """Preprocess image for model input."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._transform(image).unsqueeze(0).to(self._device)

    def _predict_aiverifynet(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run prediction using the AIVerifyNet dual-stream model.

        This method:
        1. Generates ELA representation of the input image
        2. Preprocesses both RGB and ELA images
        3. Runs dual-stream inference
        4. Returns prediction with confidence
        """
        start_time = time.time()

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Generate ELA image for frequency stream
        # Using quality=90 as per research recommendations for optimal artifact detection
        ela_image = convert_to_ela(image, quality=90)

        # Preprocess both streams
        rgb_tensor = self._preprocess_image(image)
        ela_tensor = self._preprocess_image(ela_image)

        # Run dual-stream inference
        with torch.no_grad():
            logits = self._aiverifynet(rgb_tensor, ela_tensor)
            probability = torch.sigmoid(logits)
            score = float(probability.squeeze().cpu().item())

        processing_time_ms = (time.time() - start_time) * 1000

        # Determine prediction
        is_deepfake = score >= 0.5
        confidence = score if is_deepfake else (1.0 - score)

        # Get ELA statistics for additional insights
        ela_mean, ela_std, ela_max_ratio = compute_ela_features(image, quality=90)

        return {
            "is_deepfake": is_deepfake,
            "prediction": "Fake" if is_deepfake else "Real",
            "confidence": float(confidence),
            "raw_score": float(score),
            "model_status": self.model_status,
            "processing_time_ms": round(processing_time_ms, 2),
            "details": {
                "backend": "aiverifynet",
                "device": str(self._device),
                "ela_mean": round(ela_mean, 6),
                "ela_std": round(ela_std, 6),
                "ela_max_ratio": round(ela_max_ratio, 4),
            },
        }

    def _predict_torchscript(self, image: Image.Image) -> Dict[str, Any]:
        """Run prediction using TorchScript model."""
        start_time = time.time()

        # Preprocess image
        tensor = self._preprocess_image(image)

        with torch.no_grad():
            logits = self._torch_model(tensor)
            score = float(torch.sigmoid(logits).squeeze().item())

        processing_time_ms = (time.time() - start_time) * 1000
        is_deepfake = score >= 0.5

        return {
            "is_deepfake": is_deepfake,
            "prediction": "Fake" if is_deepfake else "Real",
            "confidence": score if is_deepfake else (1.0 - score),
            "raw_score": score,
            "model_status": self.model_status,
            "processing_time_ms": round(processing_time_ms, 2),
            "details": {
                "backend": "torchscript",
                "device": str(self._device),
            },
        }

    def _predict_stub(self, image: Image.Image) -> Dict[str, Any]:
        """
        Heuristic-based prediction fallback.

        Uses ELA statistics and high-frequency energy analysis
        to estimate deepfake probability without a trained model.

        This is useful for:
        - Testing the pipeline without trained weights
        - Quick prototyping and development
        - Fallback when model loading fails
        """
        start_time = time.time()

        # Compute ELA and extract features
        ela_image = compute_ela(image)
        ela_mean, ela_std = image_stats(ela_image)
        hf_energy = high_frequency_energy(image)

        # Enhanced ELA features from the new utility
        try:
            ela_mean_new, ela_std_new, ela_max_ratio = compute_ela_features(image, quality=90)
        except Exception:
            ela_mean_new, ela_std_new, ela_max_ratio = ela_mean, ela_std, 1.0

        # Heuristic scoring combining multiple signals
        # Higher ELA mean/std and high-frequency energy often indicate manipulation
        score = 1 / (1 + math.exp(-(
            (ela_mean * 3.0) +
            (hf_energy * 2.0) +
            (ela_max_ratio * 0.1) -
            1.0
        )))

        processing_time_ms = (time.time() - start_time) * 1000
        is_deepfake = score >= 0.5

        return {
            "is_deepfake": is_deepfake,
            "prediction": "Fake" if is_deepfake else "Real",
            "confidence": float(min(1.0, max(0.0, score if is_deepfake else (1.0 - score)))),
            "raw_score": float(min(1.0, max(0.0, score))),
            "model_status": "stub-heuristic",
            "processing_time_ms": round(processing_time_ms, 2),
            "details": {
                "backend": "stub",
                "ela_mean": round(ela_mean, 6),
                "ela_std": round(ela_std, 6),
                "hf_energy": round(hf_energy, 6),
                "ela_max_ratio": round(ela_max_ratio, 4),
            },
        }

