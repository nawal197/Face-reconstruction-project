"""
Pipeline G — Identity Preserving GAN
Losses : ArcFace + Perceptual + GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ─────────────────────────────────────────────
# ArcFace Identity Loss
# ─────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    Compute cosine similarity between ArcFace embeddings
    of original and reconstructed face.
    """

    def __init__(self, model_path: str = None):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
            self.encoder = InceptionResnetV1(pretrained="vggface2").eval()
        except ImportError:
            self.encoder = None
            print("[WARNING] facenet_pytorch not installed. ArcFace loss disabled.")

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return torch.tensor(0.0)
        self.encoder.eval()
        with torch.no_grad():
            emb_real = self.encoder(F.interpolate(real, size=(160, 160)))
        emb_fake = self.encoder(F.interpolate(fake, size=(160, 160)))
        cos_sim = F.cosine_similarity(emb_fake, emb_real)
        return 1 - cos_sim.mean()


# ─────────────────────────────────────────────
# Perceptual Loss (VGG)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg)[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.layers(fake), self.layers(real))


# ─────────────────────────────────────────────
# Attention U-Net Generator
# ─────────────────────────────────────────────

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, 1, 0), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, 1, 0), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, 1, 0), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        psi = self.psi(F.relu(self.W_g(g) + self.W_x(x)))
        return x * psi


class AttentionUNet(nn.Module):
    """U-Net with attention gates for identity-preserving reconstruction."""

    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(),
            )

        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.bottleneck = conv_block(base * 8, base * 16)

        self.att4 = AttentionGate(base * 16, base * 8, base * 8)
        self.att3 = AttentionGate(base * 8, base * 4, base * 4)
        self.att2 = AttentionGate(base * 4, base * 2, base * 2)
        self.att1 = AttentionGate(base * 2, base, base)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Sequential(nn.Conv2d(base, out_ch, 1), nn.Tanh())

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self.att4(d4, e4), d4], 1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(d3, e3), d3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(d2, e2), d2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.att1(d1, e1), d1], 1))
        return self.out(d1)


import torch


# ─────────────────────────────────────────────
# Identity GAN Trainer
# ─────────────────────────────────────────────

class IdentityGANTrainer:
    def __init__(self, config: dict, trial=None):
        self.config = config
        self.trial = trial
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optuna hyperparams
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) if trial else config["training"]["lr"]
        lambda_arcface = trial.suggest_float("lambda_arcface", 0.1, 10.0, log=True) if trial else config["training"].get("lambda_arcface", 1.0)
        lambda_perceptual = trial.suggest_float("lambda_perceptual", 0.1, 10.0, log=True) if trial else config["training"].get("lambda_perceptual", 1.0)
        base_features = trial.suggest_categorical("base_features", [32, 64]) if trial else 64

        self.G = AttentionUNet(base=base_features).to(self.device)
        self.loss_arcface = ArcFaceLoss().to(self.device)
        self.loss_perceptual = PerceptualLoss().to(self.device)
        self.loss_l1 = nn.L1Loss()
        self.opt = torch.optim.Adam(self.G.parameters(), lr=lr)

        self.lambda_arcface = lambda_arcface
        self.lambda_perceptual = lambda_perceptual

    def train_step(self, lr_img, hr_img):
        lr_img = lr_img.to(self.device)
        hr_img = hr_img.to(self.device)
        import torch.nn.functional as F
        lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode="bilinear", align_corners=False)

        fake = self.G(lr_up)

        loss_l1 = self.loss_l1(fake, hr_img)
        loss_perc = self.loss_perceptual(fake, hr_img)
        loss_id = self.loss_arcface(fake, hr_img)
        total = loss_l1 + self.lambda_perceptual * loss_perc + self.lambda_arcface * loss_id

        self.opt.zero_grad()
        total.backward()
        self.opt.step()

        return {
            "loss_total": total.item(),
            "loss_l1": loss_l1.item(),
            "loss_perceptual": loss_perc.item(),
            "loss_arcface": loss_id.item(),
        }
