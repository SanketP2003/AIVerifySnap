"""
Error Level Analysis (ELA) Utility Module for Deepfake Detection.

This module implements ELA, a forensic technique used to detect image manipulations.
The core principle is based on "High-Compression Robustness" research:

When a JPEG image is resaved at a specific quality level, areas that were previously
compressed at different levels (or synthetically generated) will show different
error levels compared to the rest of the image.

Key Insight for Deepfake Detection:
- AI-generated or manipulated regions often have different compression artifacts
- These artifacts appear as high-frequency noise patterns in the ELA image
- The ELA stream in our dual-network architecture learns to detect these subtle
  inconsistencies that are invisible to the human eye but detectable by CNNs

Reference: "Fighting Deepfakes Using Error Level Analysis" and
           "High-Compression Robustness" forensic imaging research.
"""

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def convert_to_ela(
    image: Image.Image,
    quality: int = 90,
    scale_factor: float = 15.0,
    enhance_contrast: bool = True
) -> Image.Image:
    """
    Convert an image to its Error Level Analysis (ELA) representation.

    This function detects regions with different compression levels, which is
    crucial for deepfake detection because:
    1. AI-generated faces often have uniform compression patterns different from
       the surrounding authentic image regions
    2. Spliced or manipulated areas show higher error levels at boundaries
    3. GAN-generated content lacks natural JPEG compression artifacts

    The ELA process:
    1. Save the original image at a specific JPEG quality (e.g., 90%)
    2. Reload the compressed version
    3. Calculate the absolute pixel difference between original and compressed
    4. Scale the brightness to make artifacts visible to the CNN

    Args:
        image: Input PIL Image (RGB)
        quality: JPEG compression quality (1-100). Higher values = less compression.
                 90 is optimal for detecting subtle manipulations while avoiding
                 introducing too many new artifacts.
        scale_factor: Brightness scaling multiplier. Higher values make artifacts
                      more visible. Default 15.0 based on empirical testing.
        enhance_contrast: Whether to apply additional contrast enhancement for
                          better feature extraction by the CNN.

    Returns:
        PIL Image: The ELA representation with scaled brightness

    Technical Notes on High-Frequency Artifact Detection:
    - JPEG compression uses DCT (Discrete Cosine Transform) which affects
      high-frequency components differently
    - Manipulated regions that were recompressed will have different DCT
      coefficients, resulting in visible ELA differences
    - The scale_factor amplifies these differences so the CNN can learn patterns
    """
    # Ensure image is in RGB mode for consistent processing
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 1: Resave the image at the specified JPEG quality
    # Using BytesIO to avoid filesystem I/O overhead
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Step 2: Reload the compressed version
    recompressed = Image.open(buffer).convert("RGB")

    # Step 3: Calculate absolute pixel difference
    # ImageChops.difference computes |original - recompressed| per pixel per channel
    ela_image = ImageChops.difference(image, recompressed)

    # Step 4: Scale the brightness to make artifacts visible
    # The raw difference values are typically very small (0-10 range)
    # We need to amplify them for the CNN to detect patterns
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])  # Get maximum difference across all channels

    if max_diff == 0:
        # No difference detected (rare case - image might be already at target quality)
        # Return a scaled version of the raw ELA
        scale = scale_factor
    else:
        # Dynamic scaling based on the maximum difference found
        # This ensures consistent brightness across different images
        scale = min(255.0 / max_diff, scale_factor)

    # Apply brightness scaling using point operation for efficiency
    ela_scaled = ela_image.point(lambda x: min(255, int(x * scale)))

    # Optional: Enhance contrast to further highlight manipulation artifacts
    # This helps the CNN distinguish between natural compression noise and
    # artificial patterns from deepfake generation
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(ela_scaled)
        ela_scaled = enhancer.enhance(1.5)

    return ela_scaled


def compute_ela_features(
    image: Image.Image,
    quality: int = 90
) -> Tuple[float, float, float]:
    """
    Compute statistical features from the ELA image.

    These features can be used for quick heuristic-based detection
    or as additional inputs to the classification model.

    Args:
        image: Input PIL Image
        quality: JPEG compression quality for ELA computation

    Returns:
        Tuple containing:
        - ela_mean: Mean ELA value (higher = more manipulation artifacts)
        - ela_std: Standard deviation (higher = more inconsistent compression)
        - ela_max_ratio: Ratio of maximum to mean (higher = localized manipulation)
    """
    ela_image = convert_to_ela(image, quality=quality, enhance_contrast=False)
    ela_array = np.asarray(ela_image).astype(np.float32) / 255.0

    ela_mean = float(ela_array.mean())
    ela_std = float(ela_array.std())
    ela_max = float(ela_array.max())

    # Ratio of max to mean - high values indicate localized bright spots
    # which often correspond to manipulated regions
    ela_max_ratio = ela_max / (ela_mean + 1e-6)

    return ela_mean, ela_std, ela_max_ratio


def compute_multi_scale_ela(
    image: Image.Image,
    qualities: Optional[list] = None
) -> Image.Image:
    """
    Compute multi-scale ELA by averaging ELA at different quality levels.

    This approach is more robust to adversarial attacks that try to fool
    single-quality ELA detection. Different quality levels capture different
    aspects of compression artifacts.

    Args:
        image: Input PIL Image
        qualities: List of JPEG quality levels. Default: [70, 80, 90]

    Returns:
        PIL Image: Averaged multi-scale ELA representation
    """
    if qualities is None:
        qualities = [70, 80, 90]

    ela_arrays = []
    for quality in qualities:
        ela = convert_to_ela(image, quality=quality, enhance_contrast=False)
        ela_arrays.append(np.asarray(ela).astype(np.float32))

    # Average across all quality levels
    avg_ela = np.mean(ela_arrays, axis=0).astype(np.uint8)
    return Image.fromarray(avg_ela)

