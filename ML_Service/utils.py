import io
from PIL import Image, ImageChops, ImageEnhance


def generate_ela(img: Image.Image, quality: int = 90) -> Image.Image:
    """Generate an ELA image to highlight compression artifacts."""
    temp_file = io.BytesIO()
    img.convert("RGB").save(temp_file, "JPEG", quality=quality)
    temp_file.seek(0)

    compressed_img = Image.open(temp_file)

    ela_img = ImageChops.difference(img.convert("RGB"), compressed_img)

    extrema = ela_img.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    return ela_img


def generate_ela_heatmap(ela_img: Image.Image) -> Image.Image:
    """Convert an ELA image into a JET-colorized heatmap showing error intensity.

    Blue regions  = low compression error (likely authentic)
    Red regions   = high compression error (potentially manipulated)
    """
    import numpy as np

    gray = np.array(ela_img.convert("L"), dtype=np.float32)

    max_val = gray.max()
    if max_val > 0:
        gray = gray / max_val

    # JET colormap: Blue → Cyan → Green → Yellow → Red
    v = np.clip(gray, 0, 1)
    r = np.clip(1.5 - np.abs(v * 4.0 - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(v * 4.0 - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(v * 4.0 - 1.0), 0, 1)

    heatmap = np.stack([r, g, b], axis=-1)
    heatmap = (heatmap * 255).astype(np.uint8)

    return Image.fromarray(heatmap)