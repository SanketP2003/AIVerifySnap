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