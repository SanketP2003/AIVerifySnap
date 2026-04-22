import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import AIVerifySnapModel
from utils import generate_ela


class RGBELADataset(Dataset):
    def __init__(self, root: str, split: str, rgb_transform, ela_transform):
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        self.base = datasets.ImageFolder(split_root)
        self.rgb_transform = rgb_transform
        self.ela_transform = ela_transform

    def __len__(self) -> int:
        return len(self.base.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.base.samples[idx]
        img = Image.open(img_path).convert("RGB")

        rgb_tensor = self.rgb_transform(img)
        ela_img = generate_ela(img, quality=90)
        ela_tensor = self.ela_transform(ela_img)

        target = torch.tensor([float(label)], dtype=torch.float32)
        return rgb_tensor, ela_tensor, target


def build_transforms(image_size: int):
    train_rgb = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_rgb = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ela = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    eval_ela = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    return train_rgb, eval_rgb, train_ela, eval_ela


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    for rgb, ela, y in loader:
        rgb = rgb.to(device)
        ela = ela.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(rgb, ela)
                loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += compute_accuracy(logits.detach(), y) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def get_pos_weight(dataset: RGBELADataset) -> torch.Tensor:
    labels = [label for _, label in dataset.base.samples]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_rgb, eval_rgb, train_ela, eval_ela = build_transforms(args.image_size)

    val_data_dir = args.val_data_dir if args.val_data_dir else args.data_dir

    train_ds = RGBELADataset(args.data_dir, args.train_split, train_rgb, train_ela)
    val_ds = RGBELADataset(val_data_dir, args.val_split, eval_rgb, eval_ela)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AIVerifySnapModel().to(device)

    pos_weight = get_pos_weight(train_ds).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False, scaler)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "image_size": args.image_size,
                "class_mapping": train_ds.base.class_to_idx,
            }
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to: {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIVerifySnap model with RGB + ELA fusion")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root containing train/ and val/")
    parser.add_argument("--val-data-dir", type=str, default=None, help="Optional separate dataset root for validation split")
    parser.add_argument("--train-split", type=str, default="train", help="Training split folder name")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split folder name")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_model(arguments)
