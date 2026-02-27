"""
AIVerifyNet: Dual-Stream Hybrid Network for Deepfake Detection.

This module implements the core neural network architecture for AIVerifySnap.
The architecture uses a Dual-Stream approach combining:

1. Spatial Stream (RGB): Analyzes raw RGB images using pre-trained ResNet-50
   - Captures semantic features like face structure, lighting, shadows
   - Benefits from ImageNet pre-training for robust feature extraction

2. Frequency Stream (ELA): Analyzes Error Level Analysis maps using ResNet-18
   - Detects high-frequency compression artifacts
   - Specialized for catching manipulation traces invisible to human eye

3. Fusion Layer: Combines features from both streams for final classification
   - Concatenates deep features from both networks
   - Uses fully connected layers for binary classification (Real/Fake)

Research Basis:
This architecture is inspired by "Two-Stream Networks for Deepfake Detection"
and "Exploiting Visual Artifacts in Fake Face Detection" research papers.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Conditional PyTorch imports with fallback for mock mode
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class AIVerifyNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Dual-Stream Hybrid Network for Deepfake Detection.

    Architecture:
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
              │  Concatenate    │
              │  (2560 features)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FC Layer 1     │
              │  (2560 → 512)   │
              │  + ReLU + Drop  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FC Layer 2     │
              │  (512 → 128)    │
              │  + ReLU + Drop  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Output Layer   │
              │  (128 → 1)      │
              │  + Sigmoid      │
              └─────────────────┘
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize the AIVerifyNet model.

        Args:
            pretrained: Whether to use pretrained ImageNet weights for backbones
            dropout_rate: Dropout rate for regularization (0.0-1.0)
            freeze_backbone: Whether to freeze backbone weights (for fine-tuning)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for AIVerifyNet. "
                "Install with: pip install torch torchvision"
            )

        super(AIVerifyNet, self).__init__()

        self.dropout_rate = dropout_rate

        # ============================================
        # Stream 1: Spatial Stream (RGB Analysis)
        # Uses ResNet-50 pretrained on ImageNet
        # Extracts 2048-dimensional feature vector
        # ============================================
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.spatial_backbone = models.resnet50(weights=weights)
        spatial_features = self.spatial_backbone.fc.in_features  # 2048
        self.spatial_backbone.fc = nn.Identity()  # Remove final FC layer

        # ============================================
        # Stream 2: Frequency Stream (ELA Analysis)
        # Uses ResNet-18 for efficiency (ELA patterns are simpler)
        # Extracts 512-dimensional feature vector
        #
        # Why ResNet-18 for ELA?
        # - ELA images have simpler patterns than natural images
        # - Smaller model reduces overfitting on compression artifacts
        # - Faster inference for the secondary stream
        # ============================================
        ela_weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.frequency_backbone = models.resnet18(weights=ela_weights)
        frequency_features = self.frequency_backbone.fc.in_features  # 512
        self.frequency_backbone.fc = nn.Identity()  # Remove final FC layer

        # Optionally freeze backbone weights for transfer learning
        if freeze_backbone:
            for param in self.spatial_backbone.parameters():
                param.requires_grad = False
            for param in self.frequency_backbone.parameters():
                param.requires_grad = False

        # ============================================
        # Fusion Layer: Combines both streams
        # Total features: 2048 (spatial) + 512 (frequency) = 2560
        # ============================================
        combined_features = spatial_features + frequency_features  # 2560

        self.fusion = nn.Sequential(
            # First FC layer: Dimensionality reduction
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Second FC layer: Further compression
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Output layer: Binary classification
            nn.Linear(128, 1)
        )

        # Initialize fusion layer weights
        self._initialize_fusion_weights()

    def _initialize_fusion_weights(self):
        """Initialize fusion layer weights using Kaiming initialization."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        rgb_input: 'torch.Tensor',
        ela_input: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Forward pass through the dual-stream network.

        Args:
            rgb_input: RGB image tensor, shape (B, 3, 224, 224)
            ela_input: ELA image tensor, shape (B, 3, 224, 224)

        Returns:
            Tensor of shape (B, 1) with logits (before sigmoid)
        """
        # Extract spatial features from RGB stream
        spatial_features = self.spatial_backbone(rgb_input)  # (B, 2048)

        # Extract frequency features from ELA stream
        frequency_features = self.frequency_backbone(ela_input)  # (B, 512)

        # Concatenate features from both streams
        combined = torch.cat([spatial_features, frequency_features], dim=1)  # (B, 2560)

        # Pass through fusion layers
        output = self.fusion(combined)  # (B, 1)

        return output

    def predict_proba(
        self,
        rgb_input: 'torch.Tensor',
        ela_input: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Get probability predictions (applies sigmoid to logits).

        Args:
            rgb_input: RGB image tensor
            ela_input: ELA image tensor

        Returns:
            Probability tensor of shape (B, 1) in range [0, 1]
        """
        logits = self.forward(rgb_input, ela_input)
        return torch.sigmoid(logits)


class AIVerifyNetInference:
    """
    Inference wrapper for AIVerifyNet with preprocessing and postprocessing.

    This class handles:
    - Model loading (trained weights or mock mode)
    - Image preprocessing (normalization, resizing)
    - ELA computation
    - Prediction with confidence scores
    """

    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the inference wrapper.

        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            mock_mode: If True, use untrained model for architecture verification
        """
        self.mock_mode = mock_mode
        self.model: Optional[AIVerifyNet] = None
        self.device = None
        self.transform = None
        self._model_loaded = False
        self._model_status = "uninitialized"
        self._model_path = model_path
        self._requested_device = device

    def load(self) -> None:
        """Load the model and prepare for inference."""
        if not TORCH_AVAILABLE:
            self._model_status = "torch_not_available"
            if not self.mock_mode:
                raise ImportError("PyTorch is required for model inference")
            return

        # Determine device
        if self._requested_device:
            self.device = torch.device(self._requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = AIVerifyNet(pretrained=True, dropout_rate=0.3)

        # Load trained weights if available
        if self._model_path and os.path.exists(self._model_path):
            try:
                state_dict = torch.load(self._model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self._model_status = "trained"
            except Exception as e:
                if not self.mock_mode:
                    raise RuntimeError(f"Failed to load model weights: {e}")
                self._model_status = f"mock_mode (load_error: {str(e)[:50]})"
        elif self.mock_mode:
            self._model_status = "mock_mode (untrained)"
        else:
            self._model_status = "mock_mode (no_weights_file)"

        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        # Setup preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def model_status(self) -> str:
        """Get current model status."""
        return self._model_status

    def preprocess(self, image: Image.Image) -> 'torch.Tensor':
        """
        Preprocess an image for model input.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Preprocessed tensor ready for model input
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)  # Add batch dimension

    def predict(
        self,
        rgb_image: Image.Image,
        ela_image: Image.Image
    ) -> Dict[str, Any]:
        """
        Run prediction on RGB and ELA image pair.

        Args:
            rgb_image: Original RGB image
            ela_image: Corresponding ELA image

        Returns:
            Dictionary with prediction results
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not TORCH_AVAILABLE:
            # Fallback to heuristic in mock mode without torch
            return self._predict_heuristic(rgb_image, ela_image)

        # Preprocess both images
        rgb_tensor = self.preprocess(rgb_image).to(self.device)
        ela_tensor = self.preprocess(ela_image).to(self.device)

        # Run inference
        with torch.no_grad():
            probability = self.model.predict_proba(rgb_tensor, ela_tensor)
            confidence = float(probability.squeeze().cpu().item())

        # Determine prediction
        is_deepfake = confidence >= 0.5
        prediction = "Fake" if is_deepfake else "Real"

        return {
            "prediction": prediction,
            "is_deepfake": is_deepfake,
            "confidence": confidence if is_deepfake else 1.0 - confidence,
            "raw_score": confidence,
            "model_status": self._model_status,
            "device": str(self.device)
        }

    def _predict_heuristic(
        self,
        rgb_image: Image.Image,
        ela_image: Image.Image
    ) -> Dict[str, Any]:
        """Fallback heuristic prediction when torch is not available."""
        # Simple heuristic based on ELA statistics
        ela_array = np.asarray(ela_image).astype(np.float32) / 255.0
        ela_mean = float(ela_array.mean())
        ela_std = float(ela_array.std())

        # Higher ELA mean/std often indicates manipulation
        score = min(1.0, ela_mean * 5.0 + ela_std * 2.0)
        is_deepfake = score >= 0.5

        return {
            "prediction": "Fake" if is_deepfake else "Real",
            "is_deepfake": is_deepfake,
            "confidence": score if is_deepfake else 1.0 - score,
            "raw_score": score,
            "model_status": "heuristic_fallback",
            "device": "cpu"
        }


# ============================================
# Lightweight ELA Stream Alternative
# For resource-constrained environments
# ============================================

class ELAStreamCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Custom lightweight CNN for ELA stream processing.

    This can be used as an alternative to ResNet-18 when:
    - Lower memory footprint is required
    - Faster inference is needed
    - Training from scratch on ELA-specific data

    Architecture optimized for detecting compression artifacts:
    - Smaller receptive field to capture local inconsistencies
    - Fewer parameters to prevent overfitting on artifact patterns
    """

    def __init__(self, output_features: int = 512):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        super(ELAStreamCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(512, output_features)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def verify_architecture():
    """
    Verify the model architecture without trained weights.
    Useful for testing and debugging.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Architecture verification skipped.")
        return False

    print("=" * 60)
    print("AIVerifyNet Architecture Verification")
    print("=" * 60)

    try:
        # Create model instance
        model = AIVerifyNet(pretrained=False, dropout_rate=0.5)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        rgb_dummy = torch.randn(batch_size, 3, 224, 224)
        ela_dummy = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(rgb_dummy, ela_dummy)

        # Print architecture summary
        print(f"\n✓ Model created successfully")
        print(f"✓ Input shapes: RGB {rgb_dummy.shape}, ELA {ela_dummy.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        print("\n✅ Architecture verification PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Architecture verification FAILED: {e}")
        return False


if __name__ == "__main__":
    verify_architecture()

