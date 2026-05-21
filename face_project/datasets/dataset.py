"""
Dataset — chargement des triplets (LR masked, HR original, mask region)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np


class FaceDataset(Dataset):
    """
    Dataset supervisé :
      - image LR masquée (entrée)
      - image HR originale (cible)
    """

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        scale_factor: int = 4,
        mask_type: str = "surgical",
        image_size: int = 256,
        augment: bool = True,
    ):
        self.lr_dir = Path(lr_dir) / f"x{scale_factor}" / mask_type
        self.hr_dir = Path(hr_dir) / "hr"
        self.image_size = image_size
        self.augment = augment

        self.files = sorted(self.hr_dir.glob("*.png"))

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

    def __getitem__(self, idx):
        hr_path = self.files[idx]
        lr_path = self.lr_dir / hr_path.name

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB") if lr_path.exists() else hr

        if self.augment:
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
    train_ds = FaceDataset(
        lr_dir=config["data"]["processed_dir"],
        hr_dir=config["data"]["processed_dir"],
        scale_factor=config["data"]["scale_factor"],
        mask_type=config["data"]["mask_type"],
        image_size=config["data"]["image_size"],
        augment=True,
    )
    val_ds = FaceDataset(
        lr_dir=config["data"]["processed_dir"],
        hr_dir=config["data"]["processed_dir"],
        scale_factor=config["data"]["scale_factor"],
        mask_type=config["data"]["mask_type"],
        image_size=config["data"]["image_size"],
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
