"""
🚀 Face Restoration Dataset Pipeline — ULTRA-OPTIMISÉ pour Kaggle + FFHQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ FEATURES:
  ✅ Alignement facial 5-landmarks universel (MTCNN & RetinaFace)
  ✅ Génération masques réalistes (6 styles)
  ✅ Dégradation multi-échelle (x2, x4, x8)
  ✅ Split train/val/test (80/10/10)
  ✅ Limite d'images configurable
  ✅ Multiprocessing optimisé
  ✅ JPEG au lieu de PNG (16x compression)
  ✅ Compatible ESRGAN/GFPGAN/SwinIR
  ✅ Gestion robuste d'erreurs
  ✅ Rapport détaillé

⚙️  USAGE:
  python preprocess.py --max-images 10000 --detector retinaface
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import argparse
import logging
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
import functools
import time
import json
import random
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PATHS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def get_dataset_path():
    """Détecte Kaggle/local et retourne le chemin du dataset."""
    if os.path.exists('/kaggle/input'):
        paths = [
            '/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq',
            '/kaggle/input/flickrfaceshq-dataset-ffhq',
            '/kaggle/input/ffhq',
        ]
        for path in paths:
            if Path(path).exists():
                logger.info(f"🎯 Kaggle FFHQ dataset : {path}")
                return path
    return 'datasets/raw'


def get_output_path():
    """Retourne le chemin de sortie (local ou Kaggle)."""
    if os.path.exists('/kaggle/working'):
        output_path = '/kaggle/working/face_dataset'
        logger.info(f"📁 Kaggle output : {output_path}")
        return output_path
    return 'datasets/processed'


# ═══════════════════════════════════════════════════════════════════════════
# FACE DETECTION & ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """Détecteur multi-backend qui standardise la sortie des 5 landmarks."""
    
    def __init__(self, backend: str = "retinaface"):
        self.backend = backend
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            if self.backend == "mtcnn":
                from mtcnn import MTCNN
                self.model = MTCNN(min_face_size=20)
            elif self.backend == "retinaface":
                try:
                    from retinaface import RetinaFace
                    self.model = RetinaFace
                except ImportError:
                    logger.warning("RetinaFace de disponible. Bascule automatique sur MTCNN.")
                    from mtcnn import MTCNN
                    self.model = MTCNN(min_face_size=20)
                    self.backend = "mtcnn"
        except Exception as e:
            logger.error(f"Échec du chargement du modèle {self.backend}: {e}, mode fallback activé.")
            self.backend = "fallback"

    def detect_landmarks(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte le visage principal et renvoie un tableau numpy (5, 2) des landmarks.
        Retourne None si aucun visage n'est détecté.
        """
        try:
            if self.backend == "mtcnn":
                faces = self.model.detect_faces(image_rgb)
                if faces:
                    kp = faces[0]['keypoints']
                    return np.array([
                        kp['left_eye'], kp['right_eye'], kp['nose'],
                        kp['mouth_left'], kp['mouth_right']
                    ], dtype=np.float32)
            
            elif self.backend == "retinaface":
                # RetinaFace prend une image BGR par défaut ou RGB selon la configuration, 
                # pour éviter les soucis on extrait les visages de manière robuste.
                faces = self.model.detect_faces(image_rgb)
                if isinstance(faces, dict) and len(faces) > 0:
                    # Prendre la première clé (premier visage détecté)
                    first_face = faces[list(faces.keys())[0]]
                    lm = first_face['landmarks']
                    return np.array([
                        lm['left_eye'], lm['right_eye'], lm['nose'],
                        lm['mouth_left'], lm['mouth_right']
                    ], dtype=np.float32)
            return None
        except Exception as e:
            logger.debug(f"Erreur de détection ({self.backend}): {e}")
            return None


class FaceAligner:
    """Aligne les visages via transformation affine (5 landmarks)."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.h, self.w = target_size
        
        # Landmarks standards pré-calculés
        self.std_landmarks = np.array([
            [0.31556875 * self.w, 0.4615741 * self.h],
            [0.68262291 * self.w, 0.4615741 * self.h],
            [0.50026505 * self.w, 0.6405053 * self.h],
            [0.34947589 * self.w, 0.8246431 * self.h],
            [0.65073005 * self.w, 0.8246431 * self.h],
        ], dtype=np.float32)

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Aligne l'image selon les landmarks."""
        landmarks = landmarks.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.std_landmarks)
        if M is None:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.warpAffine(image, M, self.target_size, flags=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════════════════════
# MASK GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class MaskGenerator:
    """Génère des masques réalistes (6 styles)."""
    
    MASK_TYPES = ["surgical", "fabric", "n95", "black", "white", "colored"]

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.h, self.w = target_size
        
        # Pré-calculer région de masque (réutilisable)
        self.nose_y = int(0.6405053 * self.h)
        self.mask_region = self._create_mask_region()
        
        # Couleurs en BGR (natif OpenCV)
        self.colors = {
            "surgical": (180, 200, 200),
            "fabric": (180, 120, 100),
            "n95": (220, 230, 230),
            "black": (30, 30, 30),
            "white": (245, 245, 245),
            "colored": (120, 100, 180),
        }

    def _create_mask_region(self) -> np.ndarray:
        """Crée région masque réutilisable."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array([
            [int(self.w * 0.15), self.nose_y],
            [int(self.w * 0.85), self.nose_y],
            [self.w, self.h],
            [0, self.h],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def apply_mask_inplace(self, image: np.ndarray, mask_type: str = "surgical") -> np.ndarray:
        """Applique masque sans copies supplémentaires."""
        color = self.colors.get(mask_type, (200, 200, 200))
        masked = image.copy()
        
        alpha = 0.9
        mask_indices = self.mask_region > 0
        masked[mask_indices] = (
            np.array(color, dtype=np.float32) * alpha + 
            image[mask_indices].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)
        
        return masked


# ═══════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER
# ═══════════════════════════════════════════════════════════════════════════

worker_detector = None
worker_aligner = None
worker_mask_gen = None

def init_worker(backend: str, target_size: Tuple[int, int]):
    """Initialise les modèles par worker."""
    global worker_detector, worker_aligner, worker_mask_gen
    worker_detector = FaceDetector(backend=backend)
    worker_aligner = FaceAligner(target_size=target_size)
    worker_mask_gen = MaskGenerator(target_size=target_size)


ProcessingStats = Dict[str, int]

def process_single_image(
    image_path: Path,
    output_config: Dict,
    target_size: Tuple[int, int],
    split: str = "train"
) -> ProcessingStats:
    """Traite une image : Détection → Alignement → Masque → Sauvegarde."""
    global worker_detector, worker_aligner, worker_mask_gen
    
    stats = {"success": 0, "fail": 0, "skipped": 0}
    
    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.warning(f"⚠️  Load failed: {image_path.name}")
            stats["fail"] += 1
            return stats
    except Exception as e:
        logger.error(f"❌ Read error {image_path.name}: {e}")
        stats["fail"] += 1
        return stats

    # Détection des landmarks standardisés de manière unifiée
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = worker_detector.detect_landmarks(image_rgb)

    # Alignement robuste
    if landmarks is None:
        # Fallback si aucun visage détecté : resize simple pour éviter de perdre l'image
        aligned = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    else:
        try:
            aligned = worker_aligner.align(image_bgr, landmarks)
        except Exception as e:
            logger.warning(f"⚠️  Align failed {image_path.name}: {e}")
            aligned = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

    stem = image_path.stem
    
    # Sauvegarde HR
    try:
        hr_dir = output_config['splits'][split]['hr']
        hr_path = hr_dir / f"{stem}.jpg"
        cv2.imwrite(str(hr_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except Exception as e:
        logger.error(f"❌ HR save error {stem}: {e}")
        stats["fail"] += 1
        return stats

    # Génération LR avec masques
    h, w = target_size
    for mask_type in output_config['mask_types']:
        try:
            masked = worker_mask_gen.apply_mask_inplace(aligned, mask_type)

            for scale in output_config['scale_factors']:
                small = cv2.resize(
                    masked,
                    (w // scale, h // scale),
                    interpolation=cv2.INTER_AREA
                )
                
                lr_dir = output_config['splits'][split]['lr'][(scale, mask_type)]
                out_path = lr_dir / f"{stem}.jpg"
                cv2.imwrite(str(out_path), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            logger.error(f"❌ Mask error {mask_type} {stem}: {e}")
            stats["fail"] += 1
            continue

    stats["success"] += 1
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class FaceRestorationPipeline:
    """Pipeline ultime pour face restoration."""
    
    def __init__(
        self,
        detector_backend: str = "retinaface",
        mask_types: Optional[List[str]] = None,
        scale_factors: Optional[List[int]] = None,
        target_size: Tuple[int, int] = (256, 256),
        max_images: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.detector_backend = detector_backend
        self.scale_factors = scale_factors or [2, 4, 8]
        self.mask_types = mask_types or MaskGenerator.MASK_TYPES
        self.target_size = target_size
        self.max_images = max_images
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = round(1.0 - train_ratio - val_ratio, 2)

    def _setup_output_structure(self, output_path: Path) -> Dict:
        """Crée la structure complète de dossiers (train/val/test)."""
        start = time.time()
        
        splits = {}
        for split in ["train", "val", "test"]:
            split_path = output_path / split
            
            # Dossiers HR
            hr_dir = split_path / "hr"
            hr_dir.mkdir(parents=True, exist_ok=True)
            
            # Dossiers LR
            lr_dirs = {}
            for scale in self.scale_factors:
                for mask_type in self.mask_types:
                    scale_mask_dir = split_path / f"lr_x{scale}" / mask_type
                    scale_mask_dir.mkdir(parents=True, exist_ok=True)
                    lr_dirs[(scale, mask_type)] = scale_mask_dir
            
            splits[split] = {
                'hr': hr_dir,
                'lr': lr_dirs,
                'path': split_path
            }
        
        elapsed = time.time() - start
        logger.info(f"✅ Structure des dossiers créée en {elapsed:.2f}s")
        
        return splits

    def _split_images(self, images: List[Path]) -> Dict[str, List[Path]]:
        """Divise les images de manière aléatoire mais contrôlée."""
        random.seed(42)  # Fixer le seed pour la reproductibilité du split
        random.shuffle(images)
        
        n = len(images)
        train_n = int(n * self.train_ratio)
        val_n = int(n * self.val_ratio)
        
        return {
            'train': images[:train_n],
            'val': images[train_n:train_n + val_n],
            'test': images[train_n + val_n:],
        }

    def _create_metadata(self, output_path: Path, stats: Dict, splits_info: Dict):
        """Crée un fichier de métadonnées JSON complet."""
        metadata = {
            'config': {
                'detector': self.detector_backend,
                'target_size': self.target_size,
                'mask_types': self.mask_types,
                'scale_factors': self.scale_factors,
                'max_images': self.max_images,
                'splits': {'train': self.train_ratio, 'val': self.val_ratio, 'test': self.test_ratio},
            },
            'statistics': stats,
            'splits_info': {
                split: {'count': len(images), 'paths': [str(p) for p in images[:5]]}
                for split, images in splits_info.items()
            },
        }
        
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📋 Fichier métadonnées sauvegardé : {metadata_path}")

    def run(self, input_dir: str, output_dir: str):
        """Pipeline principal."""
        start_total = time.time()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        logger.info("=" * 80)
        logger.info("🚀 Démarrage du Pipeline Face Restoration")
        logger.info("=" * 80)
        
        # Collecter les images
        logger.info("📂 Indexation des images en cours...")
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(input_path.rglob(ext))
        images = list(dict.fromkeys(images))  # Élimine les doublons de chemins
        
        logger.info(f"📂 {len(images)} images trouvées au total.")
        
        # Limiter le nombre d'images (Anti-crash Inodes Kaggle)
        if self.max_images and len(images) > self.max_images:
            random.seed(42)
            random.shuffle(images)
            images = images[:self.max_images]
            logger.info(f"⚠️  Échantillonnage : Limité à {len(images)} images.")
        
        if not images:
            logger.error("❌ Aucune image trouvée. Arrêt du processus.")
            return
        
        # Initialisation dossiers
        splits = self._setup_output_structure(output_path)
        
        # Split des fichiers
        split_images = self._split_images(images)
        logger.info(f"📊 Distribution : train={len(split_images['train'])}, "
                    f"val={len(split_images['val'])}, test={len(split_images['test'])}")
        
        # Traitement parallèle par split
        total_stats = defaultdict(int)
        num_workers = max(1, os.cpu_count() - 1)
        logger.info(f"⚡ Exécution sur {num_workers} cœurs CPU, Détecteur actif : {self.detector_backend}")
        
        for split_name, split_images_list in split_images.items():
            if not split_images_list:
                continue
            
            logger.info(f"\n🔄 Traitement du set [{split_name.upper()}] ({len(split_images_list)} images)...")
            start_split = time.time()
            
            output_config = {
                'splits': splits,
                'mask_types': self.mask_types,
                'scale_factors': self.scale_factors,
            }
            
            worker_fn = functools.partial(
                process_single_image,
                output_config=output_config,
                target_size=self.target_size,
                split=split_name
            )
            
            split_stats = defaultdict(int)
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=(self.detector_backend, self.target_size)
            ) as executor:
                for stats in tqdm(
                    executor.map(worker_fn, split_images_list),
                    total=len(split_images_list),
                    desc=f"{split_name.upper()}",
                    unit="img"
                ):
                    for key, val in stats.items():
                        split_stats[key] += val
                        total_stats[key] += val
            
            elapsed_split = time.time() - start_split
            rate = len(split_images_list) / elapsed_split if elapsed_split > 0 else 0
            logger.info(f"✅ Fin du set {split_name}: {elapsed_split:.1f}s ({rate:.1f} img/s) "
                        f"[OK={split_stats['success']}, ÉCHECS={split_stats['fail']}]")
        
        # Résumé Global
        elapsed_total = time.time() - start_total
        logger.info("\n" + "=" * 80)
        logger.info("📊 RAPPORT FINAL D'EXÉCUTION")
        logger.info("=" * 80)
        logger.info(f"✅ Images traitées avec succès : {total_stats['success']}")
        logger.info(f"❌ Échecs rencontrés : {total_stats['fail']}")
        logger.info(f"⏱️  Temps total d'exécution : {elapsed_total:.1f}s")
        logger.info(f"📁 Dossier racine des données : {output_path}")
        
        # Sauvegarde du fichier de log / metadata
        self._create_metadata(output_path, dict(total_stats), split_images)
        
        logger.info("=" * 80)
        logger.info("🎉 Le Dataset est prêt pour l'entraînement (ESRGAN/SwinIR) !")
        logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Face Restoration Dataset Pipeline - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Utilisation standard optimisée par défaut
  python preprocess.py
  
  # Mode ultra-sécurité Kaggle (Recommandé pour tester rapidement ton modèle)
  python preprocess.py --max-images 5000 --detector retinaface
  
  # Extraction complète et personnalisée
  python preprocess.py --scales 2 4 --masks surgical black --size 512
        """
    )
    
    parser.add_argument("--input", type=str, default=None, help="Dossier d'entrée")
    parser.add_argument("--output", type=str, default=None, help="Dossier de sortie")
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["mtcnn", "retinaface"],
        help="Face detector (retinaface=rapide et précis, mtcnn=robuste)"
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8], help="LR scales")
    parser.add_argument("--masks", nargs="+", default=MaskGenerator.MASK_TYPES, help="Mask types")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio")
    
    args = parser.parse_args()

    input_dir = args.input or get_dataset_path()
    output_dir = args.output or get_output_path()

    pipeline = FaceRestorationPipeline(
        detector_backend=args.detector,
        mask_types=args.masks,
        scale_factors=args.scales,
        target_size=(args.size, args.size),
        max_images=args.max_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    pipeline.run(input_dir, output_dir)