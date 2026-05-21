"""
Métriques d'évaluation
=======================
PSNR, SSIM, LPIPS, FID, ArcFace Similarity
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val / np.sqrt(mse))


# ─────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index."""
    try:
        from piq import ssim
        return ssim(pred, target).item()
    except ImportError:
        # Fallback simple SSIM
        mu_x = pred.mean()
        mu_y = target.mean()
        sigma_x = pred.std()
        sigma_y = target.std()
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim_val = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
        return ssim_val.item()


# ─────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────

class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity."""

    def __init__(self, net: str = "vgg"):
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net)
            self.available = True
        except ImportError:
            logger.warning("lpips not installed. LPIPS metric disabled.")
            self.available = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not self.available:
            return -1.0
        with torch.no_grad():
            return self.loss_fn(pred, target).item()


# ─────────────────────────────────────────────
# FID
# ─────────────────────────────────────────────

class FIDMetric:
    """Fréchet Inception Distance."""

    def compute(self, real_dir: str, fake_dir: str) -> float:
        try:
            from pytorch_fid import fid_score
            return fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir], batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu", dims=2048
            )
        except ImportError:
            logger.warning("pytorch-fid not installed.")
            return -1.0


# ─────────────────────────────────────────────
# ArcFace Identity Similarity
# ─────────────────────────────────────────────

class ArcFaceSimilarity:
    """Mesure la conservation d'identité avec ArcFace."""

    def __init__(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained="vggface2").eval()
            self.available = True
        except ImportError:
            logger.warning("facenet_pytorch not installed.")
            self.available = False

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not self.available:
            return -1.0
        with torch.no_grad():
            pred_r = F.interpolate(pred, size=(160, 160))
            target_r = F.interpolate(target, size=(160, 160))
            e1 = self.model(pred_r)
            e2 = self.model(target_r)
            cos_sim = F.cosine_similarity(e1, e2).mean().item()
        return cos_sim


# ─────────────────────────────────────────────
# Full Evaluator
# ─────────────────────────────────────────────

class PipelineEvaluator:
    """
    Évalue un pipeline sur toutes les métriques.
    """

    def __init__(self):
        self.lpips = LPIPSMetric()
        self.arcface = ArcFaceSimilarity()
        self.fid = FIDMetric()

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute per-batch metrics."""
        # Normalize to [0, 1] if needed
        pred_norm = (pred * 0.5 + 0.5).clamp(0, 1)
        target_norm = (target * 0.5 + 0.5).clamp(0, 1)

        return {
            "psnr": compute_psnr(pred_norm, target_norm),
            "ssim": compute_ssim(pred_norm, target_norm),
            "lpips": self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1),
            "arcface_similarity": self.arcface.compute(pred_norm, target_norm),
        }

    def evaluate_dataset(
        self,
        model,
        val_loader,
        device: str = "cuda",
        pipeline_name: str = "pipeline",
    ) -> Dict[str, float]:
        model.eval()
        all_metrics: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in val_loader:
                lr = batch["lr"].to(device)
                hr = batch["hr"].to(device)
                pred = model(lr)

                batch_metrics = self.evaluate_batch(pred, hr)
                for k, v in batch_metrics.items():
                    all_metrics.setdefault(k, []).append(v)

        avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        print(f"\n=== {pipeline_name} Results ===")
        for k, v in avg.items():
            print(f"  {k:25s}: {v:.4f}")
        return avg
