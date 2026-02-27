# utils.py
import io
from PIL import Image, ImageChops, ImageEnhance

def generate_ela(img: Image.Image, quality: int = 90) -> Image.Image:
    """
    Generates an Error Level Analysis (ELA) image to highlight compression artifacts.
    """
    # Save the image at a specific JPEG quality into memory
    temp_file = io.BytesIO()
    img.convert('RGB').save(temp_file, 'JPEG', quality=quality)
    temp_file.seek(0)

    # Open the compressed image
    compressed_img = Image.open(temp_file)

    # Calculate the absolute difference between original and compressed
    ela_img = ImageChops.difference(img.convert('RGB'), compressed_img)

    # Get the extreme values to calculate a scale factor for enhancement
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    # Enhance the difference so it's clearly visible/usable by the CNN
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    return ela_img