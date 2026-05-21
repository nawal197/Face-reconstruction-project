"""
Pipeline A — Pix2Pix Baseline
Generator : U-Net | Discriminator : PatchGAN
Losses : L1 + GAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm


# ─────────────────────────────────────────────
# U-Net Generator
# ─────────────────────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        layers = []
        if down:
            layers += [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        layers += [nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2) if down else nn.ReLU()]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = UNetBlock(features, features * 2)
        self.down3 = UNetBlock(features * 2, features * 4)
        self.down4 = UNetBlock(features * 4, features * 8)
        self.down5 = UNetBlock(features * 8, features * 8)
        self.down6 = UNetBlock(features * 8, features * 8)
        self.down7 = UNetBlock(features * 8, features * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = UNetBlock(features * 16, features * 8, down=False, use_dropout=True)
        self.up3 = UNetBlock(features * 16, features * 8, down=False, use_dropout=True)
        self.up4 = UNetBlock(features * 16, features * 8, down=False)
        self.up5 = UNetBlock(features * 16, features * 4, down=False)
        self.up6 = UNetBlock(features * 8, features * 2, down=False)
        self.up7 = UNetBlock(features * 4, features, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))


# ─────────────────────────────────────────────
# PatchGAN Discriminator
# ─────────────────────────────────────────────

class PatchGAN(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 4, features * 8, 4, 1, 1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 8, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Pix2PixTrainer:
    def __init__(self, config: dict, trial=None):
        self.config = config
        self.trial = trial  # Optuna trial (optional)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparams (from config or Optuna trial)
        lr_g = trial.suggest_float("lr_g", 1e-5, 1e-3, log=True) if trial else config["training"]["lr_g"]
        lr_d = trial.suggest_float("lr_d", 1e-5, 1e-3, log=True) if trial else config["training"]["lr_d"]
        lambda_l1 = trial.suggest_float("lambda_l1", 50, 200) if trial else config["training"]["lambda_l1"]
        features_g = trial.suggest_categorical("features_g", [32, 64, 128]) if trial else config["model"]["features_g"]

        self.G = UNetGenerator(features=features_g).to(self.device)
        self.D = PatchGAN().to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.writer = SummaryWriter(log_dir=config["logging"]["tensorboard_dir"])

    def train_step(self, lr, hr):
        lr, hr = lr.to(self.device), hr.to(self.device)
        fake = self.G(lr)

        # Train D
        self.opt_D.zero_grad()
        d_real = self.D(lr, hr)
        d_fake = self.D(lr, fake.detach())
        loss_D = (self.bce(d_real, torch.ones_like(d_real)) +
                  self.bce(d_fake, torch.zeros_like(d_fake))) * 0.5
        loss_D.backward()
        self.opt_D.step()

        # Train G
        self.opt_G.zero_grad()
        d_fake_g = self.D(lr, fake)
        loss_GAN = self.bce(d_fake_g, torch.ones_like(d_fake_g))
        loss_L1 = self.l1(fake, hr) * self.lambda_l1
        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        self.opt_G.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item(), "loss_L1": loss_L1.item()}

    def train(self, train_loader, val_loader=None, epochs=None):
        epochs = epochs or self.config["training"]["epochs"]
        for epoch in range(epochs):
            self.G.train()
            self.D.train()
            epoch_losses = {"loss_G": 0, "loss_D": 0}

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                losses = self.train_step(batch["lr"], batch["hr"])
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

            # Log to TensorBoard
            for k, v in epoch_losses.items():
                self.writer.add_scalar(f"Train/{k}", v / len(train_loader), epoch)

            # Optuna pruning
            if self.trial and val_loader:
                val_loss = self._validate(val_loader)
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

        self._save_models()
        return epoch_losses["loss_G"] / len(train_loader)

    def _validate(self, val_loader):
        self.G.eval()
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                fake = self.G(batch["lr"].to(self.device))
                total += self.l1(fake, batch["hr"].to(self.device)).item()
        return total / len(val_loader)

    def _save_models(self):
        out = Path(self.config["logging"]["checkpoint_dir"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.G.state_dict(), out / "pix2pix_G.pth")
        torch.save(self.D.state_dict(), out / "pix2pix_D.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pix2pix.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(config)

    trainer = Pix2PixTrainer(config)
    trainer.train(train_loader, val_loader)
