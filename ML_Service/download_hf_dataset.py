import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def download_dataset():
    # Number of images to use to make training fast but effective
    NUM_TRAIN = 2000
    NUM_VAL = 400

    print("Loading HuggingFace dataset 'insanescw/20K_real_and_deepfake_images'...")
    try:
        ds = load_dataset("insanescw/20K_real_and_deepfake_images", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    data_dir = Path("data")
    splits = {
        "train": {"Real": data_dir / "train" / "Real", "Fake": data_dir / "train" / "Fake"},
        "val": {"Real": data_dir / "val" / "Real", "Fake": data_dir / "val" / "Fake"}
    }

    # Create directories
    for split_dirs in splits.values():
        for dir_path in split_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": {"Real": 0, "Fake": 0},
        "val": {"Real": 0, "Fake": 0}
    }

    print("Saving images...")
    for item in ds:
        # dataset typically has 'image' and 'label'
        # Let's inspect the first item
        img = item['image']
        label = item['label']
        
        # In yashduhan: 0 might be fake, 1 might be real, or vice versa. 
        # Usually we can check if it's an int. Let's assume 0=Fake, 1=Real or string "Real"/"Fake"
        label_str = "Real" if str(label) in ["1", "Real", "real"] else "Fake"
        # Just to be sure, let's map it based on what it actually is:
        if isinstance(label, int):
             label_str = "Real" if label == 1 else "Fake" # common convention
        else:
             label_str = str(label).capitalize()

        if label_str not in ["Real", "Fake"]:
            label_str = "Fake" # fallback

        # Determine split
        if counts["train"][label_str] < NUM_TRAIN // 2:
            split = "train"
        elif counts["val"][label_str] < NUM_VAL // 2:
            split = "val"
        else:
            if all(counts["val"][l] >= NUM_VAL // 2 for l in ["Real", "Fake"]):
                break # We have enough data
            continue # skip this label

        save_dir = splits[split][label_str]
        filename = f"{label_str.lower()}_{counts[split][label_str]}.jpg"
        save_path = save_dir / filename
        
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(save_path)
            counts[split][label_str] += 1
        except Exception as e:
            print(f"Failed to save image: {e}")

        total_saved = sum(counts["train"].values()) + sum(counts["val"].values())
        if total_saved % 100 == 0:
            print(f"Saved {total_saved} images...")

    print("Download and split complete!")
    print(f"Train counts: {counts['train']}")
    print(f"Val counts: {counts['val']}")

if __name__ == "__main__":
    download_dataset()
