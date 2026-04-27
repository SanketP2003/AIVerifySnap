"""
Prepare CASIA v2 dataset for training.
Converts the Au/ (Authentic) and Tp/ (Tampered) structure into:
    CASIA2_prepared/
        Train/
            Fake/
            Real/
        Validation/
            Fake/
            Real/
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image

SRC = Path("CASIA2")
DST = Path("CASIA2_prepared")
VAL_RATIO = 0.15  # 15% for validation
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(folder: Path):
    """Collect all valid image file paths from a folder."""
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            # Verify the file is a valid image
            try:
                with Image.open(f) as img:
                    img.verify()
                files.append(f)
            except Exception:
                print(f"  Skipping corrupt file: {f.name}")
    return files


def copy_images(files, dst_dir: Path):
    """Copy image files to destination, converting TIF to JPEG."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.suffix.lower() in {".tif", ".tiff", ".bmp"}:
            # Convert to JPEG for consistency
            dst_path = dst_dir / (f.stem + ".jpg")
            try:
                img = Image.open(f).convert("RGB")
                img.save(dst_path, "JPEG", quality=95)
            except Exception as e:
                print(f"  Failed to convert {f.name}: {e}")
        else:
            dst_path = dst_dir / f.name
            shutil.copy2(f, dst_path)


def main():
    random.seed(SEED)

    print("Collecting Authentic (Real) images from Au/ ...")
    real_files = collect_images(SRC / "Au")
    print(f"  Found {len(real_files)} valid images")

    print("Collecting Tampered (Fake) images from Tp/ ...")
    fake_files = collect_images(SRC / "Tp")
    print(f"  Found {len(fake_files)} valid images")

    # Shuffle and split
    random.shuffle(real_files)
    random.shuffle(fake_files)

    n_val_real = int(len(real_files) * VAL_RATIO)
    n_val_fake = int(len(fake_files) * VAL_RATIO)

    val_real = real_files[:n_val_real]
    train_real = real_files[n_val_real:]
    val_fake = fake_files[:n_val_fake]
    train_fake = fake_files[n_val_fake:]

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_real)} real + {len(train_fake)} fake = {len(train_real)+len(train_fake)}")
    print(f"  Val:   {len(val_real)} real + {len(val_fake)} fake = {len(val_real)+len(val_fake)}")

    # Clear destination
    if DST.exists():
        shutil.rmtree(DST)

    print("\nCopying training Real images...")
    copy_images(train_real, DST / "Train" / "Real")
    print("Copying training Fake images...")
    copy_images(train_fake, DST / "Train" / "Fake")
    print("Copying validation Real images...")
    copy_images(val_real, DST / "Validation" / "Real")
    print("Copying validation Fake images...")
    copy_images(val_fake, DST / "Validation" / "Fake")

    # Verify counts
    for split in ["Train", "Validation"]:
        for cls in ["Real", "Fake"]:
            d = DST / split / cls
            n = len(list(d.iterdir())) if d.exists() else 0
            print(f"  {split}/{cls}: {n} images")

    print("\nDone! Dataset prepared at:", DST.resolve())


if __name__ == "__main__":
    main()
