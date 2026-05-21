"""
Preprocessing Pipeline — Face Detection, Alignment, Mask Generation, Degradation
Supports Kaggle dataset structure: /with_mask/ and /without_mask/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset_path():
    """Détecte si on est sur Kaggle et retourne le bon chemin"""
    if os.path.exists('/kaggle/input'):
        # Essaie le chemin avec /datasets/
        kaggle_path = Path('/kaggle/input/datasets/prasoonkottarathil/face-mask-lite-dataset')
        if kaggle_path.exists():
            logger.info(f"🎯 Kaggle dataset detected (with /datasets/): {kaggle_path}")
            return str(kaggle_path)
        
        # Fallback: essaie sans /datasets/
        kaggle_path_alt = Path('/kaggle/input/prasoonkottarathil/face-mask-lite-dataset')
        if kaggle_path_alt.exists():
            logger.info(f"🎯 Kaggle dataset detected (without /datasets/): {kaggle_path_alt}")
            return str(kaggle_path_alt)
    
    return 'datasets/raw'

def get_output_path():
    """Retourne le chemin de sortie (local ou Kaggle)"""
    if os.path.exists('/kaggle/working'):
        output_path = Path('/kaggle/working/datasets/processed')
        logger.info(f"📁 Output path (Kaggle): {output_path}")
        return str(output_path)
    return 'datasets/processed'

def detect_kaggle_structure(input_dir: str) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Détecte si le dataset a la structure Kaggle avec_mask/without_mask
    Returns: (is_kaggle_structure, without_mask_path, with_mask_path)
    """
    input_path = Path(input_dir)
    without_mask = input_path / "without_mask"
    with_mask = input_path / "with_mask"
    
    if without_mask.exists() and with_mask.exists():
        logger.info("✅ Kaggle structure detected: with_mask/ and without_mask/")
        return True, without_mask, with_mask
    return False, None, None

# ─────────────────────────────────────────────
# Face Detection
# ─────────────────────────────────────────────

class FaceDetector:
    """Supports MTCNN, RetinaFace, MediaPipe."""

    def __init__(self, backend: str = "mtcnn"):
        self.backend = backend
        self._load_model()

    def _load_model(self):
        if self.backend == "mtcnn":
            from mtcnn import MTCNN
            self.model = MTCNN()
        elif self.backend == "retinaface":
            from retinaface import RetinaFace
            self.model = RetinaFace
        elif self.backend == "mediapipe":
            import mediapipe as mp
            self.model = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    def detect(self, image: np.ndarray) -> List[dict]:
        if self.backend == "mtcnn":
            return self.model.detect_faces(image)
        elif self.backend == "retinaface":
            return self.model.detect_faces(image)
        return []


# ─────────────────────────────────────────────
# Face Alignment
# ─────────────────────────────────────────────

class FaceAligner:
    """Aligns face using dlib or FAN."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using 5-point landmarks."""
        std_landmarks = np.array([
            [0.31556875, 0.4615741],
            [0.68262291, 0.4615741],
            [0.50026505, 0.6405053],
            [0.34947589, 0.8246431],
            [0.65073005, 0.8246431],
        ]) * np.array(self.target_size)

        M, _ = cv2.estimateAffinePartial2D(landmarks, std_landmarks)
        aligned = cv2.warpAffine(image, M, self.target_size)
        return aligned


# ─────────────────────────────────────────────
# Mask Generator
# ─────────────────────────────────────────────

class MaskGenerator:
    """
    Génère des masques artificiels : chirurgical, tissu, N95, colorés.
    """

    MASK_TYPES = ["surgical", "fabric", "n95", "black", "white", "colored"]

    def __init__(self, mask_dir: Optional[str] = None):
        self.mask_dir = Path(mask_dir) if mask_dir else None

    def apply_mask(
        self,
        image: np.ndarray,
        mask_type: str = "surgical",
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a synthetic mask to a face image.
        Returns: (masked_image, mask_region)
        """
        h, w = image.shape[:2]
        mask_region = np.zeros((h, w), dtype=np.uint8)

        if landmarks is not None:
            # Use landmarks to position mask on lower face
            nose_tip = landmarks[2].astype(int)
            left_mouth = landmarks[3].astype(int)
            right_mouth = landmarks[4].astype(int)

            pts = np.array([
                [0, nose_tip[1]],
                [w, nose_tip[1]],
                [w, h],
                [0, h],
            ], dtype=np.int32)
        else:
            # Fallback: cover lower 40% of face
            y_start = int(h * 0.45)
            pts = np.array([[0, y_start], [w, y_start], [w, h], [0, h]], dtype=np.int32)

        cv2.fillPoly(mask_region, [pts], 255)

        color = self._get_mask_color(mask_type)
        overlay = image.copy()
        overlay[mask_region > 0] = color

        # Blend for realism
        alpha = 0.85
        masked_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return masked_image, mask_region

    def _get_mask_color(self, mask_type: str) -> Tuple[int, int, int]:
        colors = {
            "surgical": (200, 200, 180),
            "fabric": (100, 120, 180),
            "n95": (220, 215, 200),
            "black": (20, 20, 20),
            "white": (240, 240, 240),
            "colored": (180, 100, 120),
        }
        return colors.get(mask_type, (200, 200, 200))


# ─────────────────────────────────────────────
# Resolution Degradation
# ─────────────────────────────────────────────

class ResolutionDegrader:
    """
    Simule une basse résolution par downscale + upscale.
    Facteurs supportés : x2, x4, x8.
    """

    def __init__(self, scale_factor: int = 4):
        assert scale_factor in [2, 4, 8], "scale_factor must be 2, 4, or 8"
        self.scale_factor = scale_factor

    def degrade(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // self.scale_factor, h // self.scale_factor),
                           interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return degraded

    def get_lr(self, image: np.ndarray) -> np.ndarray:
        """Return the true LR image (before upscaling)."""
        h, w = image.shape[:2]
        return cv2.resize(image, (w // self.scale_factor, h // self.scale_factor),
                          interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────

class PreprocessingPipeline:
    """
    End-to-end preprocessing:
      HR face → detect → align → mask → degrade → save triplet
    """

    def __init__(
        self,
        detector_backend: str = "mtcnn",
        mask_types: List[str] = None,
        scale_factors: List[int] = None,
        target_size: Tuple[int, int] = (256, 256),
        use_real_masks: bool = False,
    ):
        self.detector = FaceDetector(backend=detector_backend)
        self.aligner = FaceAligner(target_size=target_size)
        self.mask_gen = MaskGenerator()
        self.scale_factors = scale_factors or [2, 4, 8]
        self.mask_types = mask_types or MaskGenerator.MASK_TYPES
        self.target_size = target_size
        self.use_real_masks = use_real_masks

    def process_image(self, image_path: Path, output_dir: Path):
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Cannot read: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect(image_rgb)

        if not faces:
            logger.warning(f"No face detected in {image_path.name}")
            return

        for i, face in enumerate(faces[:1]):  # Process first face
            aligned = cv2.resize(image_rgb, self.target_size)

            stem = image_path.stem
            # Save HR original
            hr_path = output_dir / "hr" / f"{stem}_{i}.png"
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(hr_path), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

            # Generate masked versions
            for mask_type in self.mask_types:
                masked, mask_region = self.mask_gen.apply_mask(aligned, mask_type)

                for scale in self.scale_factors:
                    degrader = ResolutionDegrader(scale_factor=scale)
                    lr_masked = degrader.get_lr(masked)

                    out_path = output_dir / f"x{scale}" / mask_type / f"{stem}_{i}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), cv2.cvtColor(lr_masked, cv2.COLOR_RGB2BGR))

    def run(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Détecte la structure Kaggle
        is_kaggle, without_mask_dir, with_mask_dir = detect_kaggle_structure(input_dir)
        
        if is_kaggle:
            # Utilise les images de without_mask/ comme source principale
            logger.info(f"📂 Processing images from: {without_mask_dir}")
            images = list(without_mask_dir.glob("*.jpg")) + list(without_mask_dir.glob("*.png"))
            images += list(without_mask_dir.glob("**/*.jpg")) + list(without_mask_dir.glob("**/*.png"))
            images = list(set(images))  # Remove duplicates
        else:
            # Traite tous les fichiers image du dossier
            images = list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.png"))

        logger.info(f"🔄 Processing {len(images)} images...")
        for img_path in tqdm(images):
            self.process_image(img_path, output_path)
        logger.info(f"✅ Preprocessing complete. Output: {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Input directory (default: auto-detect Kaggle)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: auto-detect Kaggle)")
    parser.add_argument("--detector", type=str, default="mtcnn",
                        choices=["mtcnn", "retinaface", "mediapipe"])
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--masks", nargs="+", default=MaskGenerator.MASK_TYPES)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--use-real-masks", action="store_true", 
                        help="Use real masks from with_mask/ instead of synthetic")
    args = parser.parse_args()

    # Auto-detect paths if not provided
    input_dir = args.input or get_dataset_path()
    output_dir = args.output or get_output_path()

    logger.info(f"📥 Input:  {input_dir}")
    logger.info(f"📤 Output: {output_dir}")

    pipeline = PreprocessingPipeline(
        detector_backend=args.detector,
        mask_types=args.masks,
        scale_factors=args.scales,
        target_size=(args.size, args.size),
        use_real_masks=args.use_real_masks,
    )
    pipeline.run(input_dir, output_dir)
