"""Prepare CASIA2 dataset into train/val splits for training."""
import os
import random
import shutil
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2
SRC_DIR = Path("CASIA2")
DST_DIR = Path("data")

random.seed(SEED)


def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts]


def split_and_copy(files, label: str):
    random.shuffle(files)
    split_idx = int(len(files) * (1 - VAL_RATIO))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for split, subset in [("train", train_files), ("val", val_files)]:
        dst = DST_DIR / split / label
        dst.mkdir(parents=True, exist_ok=True)
        for f in subset:
            shutil.copy2(f, dst / f.name)

    return len(train_files), len(val_files)


def main():
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
        print(f"Cleared existing {DST_DIR}/")

    real_imgs = collect_images(SRC_DIR / "Au")
    fake_imgs = collect_images(SRC_DIR / "Tp")

    print(f"Found {len(real_imgs)} Real (Au) and {len(fake_imgs)} Fake (Tp) images")

    r_train, r_val = split_and_copy(real_imgs, "Real")
    f_train, f_val = split_and_copy(fake_imgs, "Fake")

    print(f"Train: {r_train} Real + {f_train} Fake = {r_train + f_train}")
    print(f"Val:   {r_val} Real + {f_val} Fake = {r_val + f_val}")
    print("Done!")


if __name__ == "__main__":
    main()
