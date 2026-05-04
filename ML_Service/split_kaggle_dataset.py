import os
import shutil
import random
from pathlib import Path

def split_dataset():
    src_dir = Path(r"C:\Users\2019s\.cache\kagglehub\datasets\ciplab\real-and-fake-face-detection\versions\1\real_and_fake_face")
    dst_dir = Path("data")

    val_ratio = 0.15
    seed = 42
    random.seed(seed)

    for cls, folder in [("Fake", "training_fake"), ("Real", "training_real")]:
        cls_dir = src_dir / folder
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} does not exist.")
            continue

        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
        random.shuffle(files)

        n_val = int(len(files) * val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split, split_files in [("val", val_files), ("train", train_files)]:
            split_dst = dst_dir / split / cls
            split_dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, split_dst / f.name)

        print(f"{cls}: Copied {len(train_files)} to train, {len(val_files)} to val.")

if __name__ == "__main__":
    split_dataset()
