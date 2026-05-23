"""
Dataset — chargement des triplets (LR masked, HR original, mask region)

Attendu (produit par face_project/preprocessing/preprocess.py) :
  processed_dir/{split}/hr/*.{jpg,png}
  processed_dir/{split}/lr_x{scale}/{mask_type}/*.{jpg,png}
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


class FaceDataset(Dataset):
    """Dataset supervisé :
      - image LR masquée (entrée)
      - image HR originale (cible)
    """

    def __init__(
        self,
        processed_dir: str,
        split: str,
        scale_factor: int = 4,
        mask_type: str = "surgical",
        image_size: int = 256,
        augment: bool = True,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        processed_path = Path(processed_dir) / split

        self.lr_dir = processed_path / f"lr_x{scale_factor}" / mask_type
        self.hr_dir = processed_path / "hr"

        self.image_size = image_size
        self.augment = augment
        self.extensions = tuple(extensions)

        # Build index from HR filenames (same stem should exist for LR)
        files = []
        for ext in self.extensions:
            files.extend(self.hr_dir.glob(f"*{ext}"))
        # Dedup + stable ordering
        self.files = sorted({p.as_posix(): p for p in files}.values(), key=lambda p: p.name)

        self.lr_transform = T.Compose([
            T.Resize((image_size // scale_factor, image_size // scale_factor)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.hr_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        if augment:
            self.aug = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.files)

    def _resolve_lr_path(self, hr_path: Path) -> Path:
        """Return the LR path matching hr stem, tolerating different extensions."""
        # Try same suffix first
        candidate = self.lr_dir / hr_path.name
        if candidate.exists():
            return candidate

        # Fallback: try common extensions
        stem = hr_path.stem
        for ext in self.extensions:
            c2 = self.lr_dir / f"{stem}{ext}"
            if c2.exists():
                return c2

        # If nothing found, return candidate (will be handled upstream)
        return candidate

    def __getitem__(self, idx):
        hr_path = self.files[idx]
        lr_path = self._resolve_lr_path(hr_path)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB") if lr_path.exists() else hr

        if self.augment:
            # Ensure deterministic flip between HR and LR
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            hr = self.aug(hr)
            torch.manual_seed(seed)
            lr = self.aug(lr)

        return {
            "lr": self.lr_transform(lr),
            "hr": self.hr_transform(hr),
            "name": hr_path.stem,
        }


def get_dataloaders(config: dict):
    processed_dir = config["data"]["processed_dir"]
    image_size = config["data"]["image_size"]
    scale_factor = config["data"]["scale_factor"]
    mask_type = config["data"]["mask_type"]

    # Kaggle config currently sets split: train.
    # We create both train and val loaders for proper training/early pruning.
    train_split = config["data"].get("train_split", config["data"].get("split", "train"))
    val_split = config["data"].get("val_split", "val")

    train_ds = FaceDataset(
        processed_dir=processed_dir,
        split=train_split,
        scale_factor=scale_factor,
        mask_type=mask_type,
        image_size=image_size,
        augment=True,
    )
    val_ds = FaceDataset(
        processed_dir=processed_dir,
        split=val_split,
        scale_factor=scale_factor,
        mask_type=mask_type,
        image_size=image_size,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, val_loader

