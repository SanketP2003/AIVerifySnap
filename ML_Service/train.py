import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import AIVerifySnapModel, AIVerifySnapModelV1
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
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    step = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (rgb, ela, y) in enumerate(loader, start=1):
        # Non-blocking copies let host-to-device transfer overlap with compute
        # when DataLoader uses pinned memory.
        rgb = rgb.to(device, non_blocking=True, memory_format=torch.channels_last)
        ela = ela.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(rgb, ela)
                loss = loss_fn(logits, y)
                if train and grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if train:
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += compute_accuracy(logits.detach(), y) * batch_size
        total_count += batch_size

    if train and step > 0 and (step % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_rgb, eval_rgb, train_ela, eval_ela = build_transforms(args.image_size)

    val_data_dir = args.val_data_dir if args.val_data_dir else args.data_dir

    train_ds = RGBELADataset(args.data_dir, args.train_split, train_rgb, train_ela)
    val_ds = RGBELADataset(val_data_dir, args.val_split, eval_rgb, eval_ela)

    if args.num_workers < 0:
        cpu_count = os.cpu_count() or 1
        if os.name == "nt":
            # Windows is more prone to shared file mapping failures with many workers.
            args.num_workers = max(1, min(2, cpu_count // 4))
        else:
            # Keep one or two logical cores available for the UI and background tasks.
            args.num_workers = max(1, cpu_count - 2)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        if os.name == "nt":
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = 1
        else:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if args.model_version == "v1":
        model = AIVerifySnapModelV1(freeze_backbone=not args.fine_tune_backbone).to(device)
        print("Using AIVerifySnapModelV1 architecture")
    else:
        model = AIVerifySnapModel(freeze_backbone=not args.fine_tune_backbone).to(device)
        print("Using AIVerifySnapModel (latest) architecture")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    pos_weight = get_pos_weight(train_ds).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("resnet."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.learning_rate * 0.1},
        {"params": head_params, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            True,
            scaler,
            args.grad_accum_steps,
        )
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False, scaler)

        scheduler.step(epoch)

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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of DataLoader workers; -1 uses most logical CPU cores")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Accumulate gradients across this many batches before optimizer step")
    parser.add_argument("--fine-tune-backbone", action="store_true", help="Train the RGB backbone instead of freezing it for speed")
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--model-version", type=str, default="latest", choices=["v1", "latest"], help="Model architecture version: v1 (simpler, matches old checkpoint) or latest")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_model(arguments)
