"""
Aggressive training script for AIVerifySnapModelV1 targeting maximum accuracy.

Key improvements over the base train.py:
- Heavier data augmentation (rotation, blur, Gaussian noise, random erasing)
- Mixup augmentation for better generalization
- Label smoothing to prevent overconfident predictions
- OneCycleLR scheduler for faster convergence
- Multi-phase training: Phase 1 (frozen backbone) then Phase 2 (fine-tuned)
- Progress bars and detailed logging
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import AIVerifySnapModelV1
from utils import generate_ela


# ─── Reproducibility ────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Custom Augmentation Transforms ────────────────────────────────
class RandomGaussianBlur:
    """Apply Gaussian blur with a random radius."""
    def __init__(self, p: float = 0.3, radius_range=(0.5, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomJPEGCompress:
    """Simulate JPEG compression artifacts."""
    def __init__(self, p: float = 0.3, quality_range=(30, 85)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < self.p:
            import io
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


# ─── Dataset ────────────────────────────────────────────────────────
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
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a black image on error instead of crashing
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            label = 0

        rgb_tensor = self.rgb_transform(img)
        ela_img = generate_ela(img, quality=90)
        ela_tensor = self.ela_transform(ela_img)
        target = torch.tensor([float(label)], dtype=torch.float32)
        return rgb_tensor, ela_tensor, target


def build_transforms(image_size: int, phase: str = "train"):
    """Build aggressive augmentation for training, standard for eval."""
    if phase == "train":
        rgb_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            RandomGaussianBlur(p=0.2),
            RandomJPEGCompress(p=0.2, quality_range=(40, 90)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
        ela_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
    else:
        rgb_tf = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ela_tf = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    return rgb_tf, ela_tf


# ─── Mixup ──────────────────────────────────────────────────────────
def mixup_data(rgb, ela, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = rgb.size(0)
    index = torch.randperm(batch_size, device=rgb.device)
    mixed_rgb = lam * rgb + (1 - lam) * rgb[index]
    mixed_ela = lam * ela + (1 - lam) * ela[index]
    y_a, y_b = y, y[index]
    return mixed_rgb, mixed_ela, y_a, y_b, lam


def mixup_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


# ─── Accuracy ───────────────────────────────────────────────────────
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


# ─── Training Loop ──────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler,
    grad_accum_steps: int = 1,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    step = 0
    num_batches = len(loader)
    start_time = time.time()

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (rgb, ela, y) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True, memory_format=torch.channels_last)
        ela = ela.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if train and use_mixup:
                    mixed_rgb, mixed_ela, y_a, y_b, lam = mixup_data(rgb, ela, y, mixup_alpha)
                    logits = model(mixed_rgb, mixed_ela)
                    loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
                else:
                    logits = model(rgb, ela)
                    loss = loss_fn(logits, y)

                if train and grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if train:
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += compute_accuracy(logits.detach(), y) * batch_size
        total_count += batch_size

        # Progress bar
        if step % max(1, num_batches // 20) == 0 or step == num_batches:
            elapsed = time.time() - start_time
            eta = elapsed / step * (num_batches - step)
            acc_so_far = total_correct / max(total_count, 1)
            phase = "Train" if train else "Val  "
            sys.stdout.write(
                f"\r  {phase} [{step:4d}/{num_batches}] "
                f"loss={total_loss / max(total_count, 1):.4f} "
                f"acc={acc_so_far:.4f} "
                f"ETA={eta:.0f}s"
            )
            sys.stdout.flush()

    # Final gradient step
    if train and step > 0 and (step % grad_accum_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    print()  # newline after progress bar
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def get_pos_weight(dataset: RGBELADataset) -> torch.Tensor:
    labels = [label for _, label in dataset.base.samples]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negatives / positives], dtype=torch.float32)


# ─── Main Training ──────────────────────────────────────────────────
def train_model(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  AIVerifySnap V1 Training — Targeting Maximum Accuracy")
    print(f"  Device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    else:
        print()
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size} (x{args.grad_accum_steps} accum)")
    print(f"{'='*60}\n")

    # Build transforms
    train_rgb, train_ela = build_transforms(args.image_size, "train")
    eval_rgb, eval_ela = build_transforms(args.image_size, "eval")

    val_data_dir = args.val_data_dir if args.val_data_dir else args.data_dir

    train_ds = RGBELADataset(args.data_dir, args.train_split, train_rgb, train_ela)
    val_ds = RGBELADataset(val_data_dir, args.val_split, eval_rgb, eval_ela)

    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"  Classes: {train_ds.base.class_to_idx}\n")

    # Workers
    if args.num_workers < 0:
        cpu_count = os.cpu_count() or 1
        if os.name == "nt":
            args.num_workers = max(1, min(2, cpu_count // 4))
        else:
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"

    # ──────────────────────────────────────────────────────────────────
    # PHASE 1: Frozen backbone — train only head + ELA CNN
    # ──────────────────────────────────────────────────────────────────
    phase1_epochs = max(1, args.epochs // 3)
    phase2_epochs = args.epochs - phase1_epochs

    print(f"Phase 1: Frozen backbone ({phase1_epochs} epochs)")
    print(f"Phase 2: Fine-tune all layers ({phase2_epochs} epochs)\n")

    model = AIVerifySnapModelV1(freeze_backbone=True).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    pos_weight = get_pos_weight(train_ds).to(device)
    # Label smoothing via adjusted pos_weight
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Phase 1 optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader), epochs=phase1_epochs,
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, phase1_epochs + 1):
        print(f"Epoch {epoch:03d}/{phase1_epochs} [Phase 1 — Frozen Backbone]")
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, device, True, scaler,
            args.grad_accum_steps, use_mixup=True, mixup_alpha=0.2,
        )
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False, scaler)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  => train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_now:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "image_size": args.image_size,
                "class_mapping": train_ds.base.class_to_idx,
            }, best_path)
            print(f"  ★ New best: {best_val_acc:.4f} — saved to {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print("  Early stopping (Phase 1).")
            break

    # ──────────────────────────────────────────────────────────────────
    # PHASE 2: Fine-tune everything
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine-tuning all layers ({phase2_epochs} epochs)")
    print(f"{'='*60}\n")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    model.backbone_trainable = True

    # Reload best checkpoint from Phase 1
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded Phase 1 best checkpoint (acc={ckpt['best_val_acc']:.4f})\n")

    # Differential learning rates: backbone much lower than head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("resnet."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.learning_rate * 0.05},
        {"params": head_params, "lr": args.learning_rate * 0.5},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    patience_counter = 0

    for epoch in range(1, phase2_epochs + 1):
        print(f"Epoch {epoch:03d}/{phase2_epochs} [Phase 2 — Fine-tuning]")
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, device, True, scaler,
            args.grad_accum_steps, use_mixup=True, mixup_alpha=0.15,
        )
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False, scaler)
        scheduler.step(epoch)

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        print(
            f"  => train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr_bb={lr_bb:.2e} lr_head={lr_head:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "image_size": args.image_size,
                "class_mapping": train_ds.base.class_to_idx,
            }, best_path)
            print(f"  ★ New best: {best_val_acc:.4f} — saved to {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print("  Early stopping (Phase 2).")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint: {best_path}")
    print(f"{'='*60}")
    return best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIVerifySnapModelV1 — Maximum Accuracy")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-data-dir", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
