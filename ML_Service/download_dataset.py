"""Download a deepfake-vs-real dataset from HuggingFace and organize
it into the folder structure expected by train.py:

    data/
      train/
        Fake/   (label 0)
        Real/   (label 1)
      val/
        Fake/
        Real/
"""

import os
import random
import shutil
from pathlib import Path

SEED = 42
random.seed(SEED)

DST = Path("data")

# CPU-tractable sizes
MAX_TRAIN_PER_CLASS = 1200
MAX_VAL_PER_CLASS = 250


def try_download():
    """Try multiple dataset sources until one works."""
    from datasets import load_dataset

    sources = [
        # (HF name, label_col, fake_label_value, real_label_value)
        ("dappai/Deepfake-vs-Real-v2", "label", 0, 1),
        ("zeeshanaliicreativez/AI-vs-Deepfake-vs-Real", "label", 0, 2),
        ("dragonintelligence/CIFAKE-image-dataset", "label", 0, 1),
        ("yanbax/CIFAKE_autotrain_compatible", "label", 0, 1),
        ("itsLeen/deepfake_vs_real_image_detection", "label", 0, 1),
    ]

    for hf_name, label_col, fake_val, real_val in sources:
        print(f"\nTrying: {hf_name} ...")
        try:
            ds = load_dataset(hf_name)
            print(f"  ✓ Downloaded!  Splits: {list(ds.keys())}")
            print(f"  Features: {ds[list(ds.keys())[0]].features}")

            # Get train split
            if "train" in ds:
                all_data = ds["train"]
            else:
                all_data = ds[list(ds.keys())[0]]

            return all_data, label_col, fake_val, real_val, hf_name
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:120]}")

    return None, None, None, None, None


def save_images(dataset, label_col, fake_val, real_val):
    """Save dataset images to disk in the expected folder structure."""
    from PIL import Image as PILImage

    if DST.exists():
        shutil.rmtree(DST)
        print(f"Cleared existing {DST}/")

    # Separate by class
    fake_indices = []
    real_indices = []

    print(f"Scanning {len(dataset)} images...")
    for idx in range(len(dataset)):
        label = dataset[idx][label_col]
        if label == fake_val:
            fake_indices.append(idx)
        elif label == real_val:
            real_indices.append(idx)

    print(f"Found {len(fake_indices)} Fake, {len(real_indices)} Real images")

    random.shuffle(fake_indices)
    random.shuffle(real_indices)

    # Calculate split sizes
    n_fake_train = min(MAX_TRAIN_PER_CLASS, int(len(fake_indices) * 0.8))
    n_real_train = min(MAX_TRAIN_PER_CLASS, int(len(real_indices) * 0.8))
    n_fake_val = min(MAX_VAL_PER_CLASS, len(fake_indices) - n_fake_train)
    n_real_val = min(MAX_VAL_PER_CLASS, len(real_indices) - n_real_train)

    splits = {
        "train": {
            "Fake": fake_indices[:n_fake_train],
            "Real": real_indices[:n_real_train],
        },
        "val": {
            "Fake": fake_indices[n_fake_train:n_fake_train + n_fake_val],
            "Real": real_indices[n_real_train:n_real_train + n_real_val],
        },
    }

    total = 0
    for split_name, classes in splits.items():
        for class_name, indices in classes.items():
            out_dir = DST / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for i, idx in enumerate(indices):
                try:
                    item = dataset[idx]
                    img = item["image"]
                    if not isinstance(img, PILImage.Image):
                        continue
                    img = img.convert("RGB")
                    save_path = out_dir / f"{class_name.lower()}_{i:05d}.jpg"
                    img.save(save_path, "JPEG", quality=95)
                    saved += 1
                    if saved % 200 == 0:
                        print(f"  [{split_name}/{class_name}] {saved}/{len(indices)}")
                except Exception as e:
                    print(f"  Skip {idx}: {e}")
                    continue

            total += saved
            print(f"  {split_name}/{class_name}: {saved} images saved")

    return total


def main():
    print("=" * 60)
    print("AIVerifySnap — Dataset Downloader")
    print("=" * 60)

    dataset, label_col, fake_val, real_val, name = try_download()
    if dataset is None:
        print("\n❌ All dataset sources failed!")
        print("Please download a dataset manually and place it as:")
        print("  data/train/Fake/  and  data/train/Real/")
        print("  data/val/Fake/    and  data/val/Real/")
        return

    print(f"\n--- Saving images from {name} ---")
    total = save_images(dataset, label_col, fake_val, real_val)
    print(f"\n✅ Done! {total} images saved to {DST.resolve()}")


if __name__ == "__main__":
    main()
