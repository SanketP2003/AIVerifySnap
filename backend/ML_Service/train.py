"""
Training Script for AIVerifyNet Dual-Stream Model.

This script provides a complete training pipeline for the AIVerifyNet
deepfake detection model, including:
- Dataset loading and preprocessing
- Training loop with validation
- Model checkpointing
- TensorBoard logging

Usage:
    python train.py --data_dir /path/to/dataset --epochs 50 --batch_size 32

Dataset Structure:
    data_dir/
        train/
            real/
                image1.jpg
                image2.jpg
            fake/
                image1.jpg
                image2.jpg
        val/
            real/
            fake/

Requirements:
    pip install torch torchvision numpy pillow
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Check for PyTorch first before any other imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("=" * 60)
    print("ERROR: PyTorch is required for training.")
    print("=" * 60)
    print("\nPlease install PyTorch with:")
    print("  pip install torch torchvision")
    print("\nOr for CUDA GPU support:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60)
    sys.exit(1)

import numpy as np
from PIL import Image

# Import from our modules
sys.path.insert(0, str(Path(__file__).parent))

from app.aiverifynet import AIVerifyNet
from app.ela_utils import convert_to_ela


class DeepfakeDataset(Dataset):
    """
    Dataset class for loading deepfake detection data.

    Expects directory structure:
        root/
            real/
                image1.jpg
                ...
            fake/
                image1.jpg
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        ela_quality: int = 90
    ):
        self.root_dir = Path(root_dir)
        self.ela_quality = ela_quality

        # Default transform with ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Load image paths and labels
        self.samples = []

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        # Real images (label = 0)
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(img_path), 0))

        # Fake images (label = 1)
        if fake_dir.exists():
            for img_path in fake_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(img_path), 1))

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Generate ELA image
        ela_image = convert_to_ela(image, quality=self.ela_quality)

        # Apply transforms
        rgb_tensor = self.transform(image)
        ela_tensor = self.transform(ela_image)

        # Label as float tensor for BCEWithLogitsLoss
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        return rgb_tensor, ela_tensor, label_tensor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for rgb_batch, ela_batch, labels in dataloader:
        rgb_batch = rgb_batch.to(device)
        ela_batch = ela_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_batch, ela_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * rgb_batch.size(0)
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_batch, ela_batch, labels in dataloader:
            rgb_batch = rgb_batch.to(device)
            ela_batch = ela_batch.to(device)
            labels = labels.to(device)

            outputs = model(rgb_batch, ela_batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * rgb_batch.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train AIVerifyNet model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")
    args = parser.parse_args()


    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = DeepfakeDataset(os.path.join(args.data_dir, "train"))
    val_dataset = DeepfakeDataset(os.path.join(args.data_dir, "val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = AIVerifyNet(
        pretrained=args.pretrained,
        dropout_rate=0.3,
        freeze_backbone=args.freeze_backbone
    )
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "aiverifynet.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")

    print("=" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'aiverifynet.pth'}")


if __name__ == "__main__":
    main()

