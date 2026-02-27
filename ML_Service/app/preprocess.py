"""
Image Preprocessing Module for AIVerifySnap.

This module provides image loading, resizing, and feature extraction utilities
used by the deepfake detection pipeline.
"""

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load an image from raw bytes and convert to RGB.

    Args:
        data: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        PIL Image in RGB mode

    Raises:
        Exception: If image cannot be decoded
    """
    image = Image.open(io.BytesIO(data))
    return image.convert("RGB")


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """
    Resize image to have maximum dimension equal to max_side.

    Maintains aspect ratio. Only downsizes, never upsizes.

    Args:
        image: Input PIL Image
        max_side: Maximum allowed dimension (width or height)

    Returns:
        Resized PIL Image (or original if already smaller)
    """
    width, height = image.size
    max_current = max(width, height)
    if max_current <= max_side:
        return image
    scale = max_side / max_current
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def resize_to_square(image: Image.Image, size: int = 224) -> Image.Image:
    """
    Resize image to a square of specified size.

    Uses LANCZOS resampling for high-quality downscaling.

    Args:
        image: Input PIL Image
        size: Target size (default 224 for neural network input)

    Returns:
        Resized PIL Image of shape (size, size)
    """
    return image.resize((size, size), Image.LANCZOS)


def compute_ela(image: Image.Image, quality: int = 90) -> Image.Image:
    """
    Compute Error Level Analysis (ELA) image.

    ELA reveals compression artifacts by comparing the original image
    with a recompressed version. Areas with different compression levels
    (often indicating manipulation) show higher error values.

    Args:
        image: Input PIL Image (should be RGB)
        quality: JPEG compression quality for recompression

    Returns:
        ELA difference image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    ela_image = ImageChops.difference(image, recompressed)
    return ela_image


def image_stats(image: Image.Image) -> Tuple[float, float]:
    """
    Compute basic statistics of an image.

    Args:
        image: Input PIL Image

    Returns:
        Tuple of (mean, std) normalized to [0, 1] range
    """
    array = np.asarray(image).astype(np.float32) / 255.0
    mean = float(array.mean())
    std = float(array.std())
    return mean, std


def high_frequency_energy(image: Image.Image) -> float:
    """
    Compute high-frequency energy using Laplacian operator.

    High-frequency energy is an indicator of image sharpness and
    can reveal manipulation artifacts. Manipulated regions often
    have different high-frequency characteristics than authentic regions.

    Args:
        image: Input PIL Image

    Returns:
        Normalized high-frequency energy value in [0, 1]
    """
    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0

    # Laplacian kernel for edge detection
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    # Pad image for convolution
    padded = np.pad(gray, 1, mode="edge")

    # Compute Laplacian response
    acc = 0.0
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            region = padded[y : y + 3, x : x + 3]
            acc += float((region * kernel).sum() ** 2)

    energy = acc / (gray.shape[0] * gray.shape[1])
    return float(min(1.0, energy))


def normalize_image(image: Image.Image) -> np.ndarray:
    """
    Normalize image to [0, 1] range as numpy array.

    Args:
        image: Input PIL Image

    Returns:
        Normalized numpy array of shape (H, W, 3)
    """
    return np.asarray(image).astype(np.float32) / 255.0


def apply_imagenet_normalization(array: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalization to a numpy array.

    Uses ImageNet mean and std:
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]

    Args:
        array: Numpy array of shape (H, W, 3) in [0, 1] range

    Returns:
        Normalized array ready for model input
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (array - mean) / std


