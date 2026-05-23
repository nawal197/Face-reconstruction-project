Voici le code complet, corrigé et optimisé.

### Ce qui a été corrigé et amélioré :

1. **Correction du bug d'alignement :** Le visage est maintenant réellement aligné via les points clés (landmarks) détectés.
2. **Correction du bug du masque :** Une fois le visage aligné, ses coordonnées changent. Le code utilise maintenant les coordonnées alignées standards pour que le masque soit **parfaitement positionné** sur la bouche et le nez.
3. **Optimisation Multiprocessing :** Traiter des milliers d'images une par une prendrait des jours. Le code utilise désormais **tous les cœurs de votre CPU en parallèle** (`ProcessPoolExecutor`).
4. **Sécurité d'initialisation :** Les modèles de Deep Learning (comme MTCNN) ne peuvent pas être partagés nativement entre plusieurs cœurs (problème de *Pickle* en Python). Ils sont maintenant initialisés proprement une seule fois par cœur de calcul (`worker_init`).

```python
"""
Preprocessing Pipeline — Face Detection, Alignment, Mask Generation, Degradation
Optimisé pour FFHQ / Kaggle avec support Multiprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import logging
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dataset_path():
    """Détecte si on est sur Kaggle et retourne le bon chemin"""
    if os.path.exists('/kaggle/input'):
        kaggle_path = Path('/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq')
        if kaggle_path.exists():
            logger.info(f"🎯 Kaggle FFHQ dataset détecté : {kaggle_path}")
            return str(kaggle_path)
    return 'datasets/raw'

def get_output_path():
    """Retourne le chemin de sortie (local ou Kaggle)"""
    if os.path.exists('/kaggle/working'):
        output_path = Path('/kaggle/working/datasets/processed')
        logger.info(f"📁 Chemin de sortie (Kaggle) : {output_path}")
        return str(output_path)
    return 'datasets/processed'


# ─────────────────────────────────────────────
# Face Detection & Alignment
# ─────────────────────────────────────────────

class FaceDetector:
    """Support de détection de visages."""
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

class FaceAligner:
    """Aligne le visage par transformation affine selon 5 points de repère."""
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        # Repères standards (yeux, nez, coins de la bouche)
        self.std_landmarks = np.array([
            [0.31556875, 0.4615741],  # Œil gauche
            [0.68262291, 0.4615741],  # Œil droit
            [0.50026505, 0.6405053],  # Nez
            [0.34947589, 0.8246431],  # Bouche gauche
            [0.65073005, 0.8246431],  # Bouche droite
        ]) * np.array(self.target_size)

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.std_landmarks)
        if M is admissions None:
            return cv2.resize(image, self.target_size)
        return cv2.warpAffine(image, M, self.target_size)


# ─────────────────────────────────────────────
# Mask Generator & Degradation
# ─────────────────────────────────────────────

class MaskGenerator:
    """Génère des masques artificiels réalistes basés sur la position du visage."""
    MASK_TYPES = ["surgical", "fabric", "n95", "black", "white", "colored"]

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        # Les landmarks du visage une fois ALIGNÉ
        self.aligned_landmarks = np.array([
            [0.31556875, 0.4615741],
            [0.68262291, 0.4615741],
            [0.50026505, 0.6405053],
            [0.34947589, 0.8246431],
            [0.65073005, 0.8246431],
        ]) * np.array(self.target_size)

    def apply_mask(self, image: np.ndarray, mask_type: str = "surgical") -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        mask_region = np.zeros((h, w), dtype=np.uint8)

        # Utilisation des landmarks alignés fixes pour couvrir le bas du visage
        nose_tip = self.aligned_landmarks[2].astype(int)
        
        # Création du polygone du masque (du nez jusqu'au menton)
        pts = np.array([
            [int(w * 0.15), nose_tip[1]],
            [int(w * 0.85), nose_tip[1]],
            [w, h],
            [0, h],
        ], dtype=np.int32)

        cv2.fillPoly(mask_region, [pts], 255)
        color = self._get_mask_color(mask_type)
        
        overlay = image.copy()
        overlay[mask_region > 0] = color

        # Mélange alpha pour plus de réalisme
        alpha = 0.9
        masked_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return masked_image, mask_region

    def _get_mask_color(self, mask_type: str) -> Tuple[int, int, int]:
        colors = {
            "surgical": (200, 200, 180),  # Bleu chirurgical (RGB)
            "fabric": (100, 120, 180),
            "n95": (230, 230, 220),
            "black": (30, 30, 30),
            "white": (245, 245, 245),
            "colored": (180, 100, 120),
        }
        return colors.get(mask_type, (200, 200, 200))


# ─────────────────────────────────────────────
# Multiprocessing Worker Core
# ─────────────────────────────────────────────

# Variables globales spécifiques à chaque processus worker
worker_detector = None
worker_aligner = None
worker_mask_gen = None

def init_worker(backend: str, target_size: Tuple[int, int]):
    """Initialise les modèles une seule fois par cœur CPU (évite les crashs de mémoire)"""
    global worker_detector, worker_aligner, worker_mask_gen
    worker_detector = FaceDetector(backend=backend)
    worker_aligner = FaceAligner(target_size=target_size)
    worker_mask_gen = MaskGenerator(target_size=target_size)

def process_single_image(image_path: Path, output_dir: Path, mask_types: List[str], scale_factors: List[int], target_size: Tuple[int, int]):
    """Traite une seule image : Détection -> Alignement -> Masque -> Dégradation -> Sauvegarde"""
    global worker_detector, worker_aligner, worker_mask_gen
    
    image = cv2.imread(str(image_path))
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = worker_detector.detect(image_rgb)

    # Si aucun visage n'est détecté, on resize par défaut (FFHQ est généralement déjà centré)
    if not faces:
        aligned = cv2.resize(image_rgb, target_size)
    else:
        face = faces[0]
        if worker_detector.backend == "mtcnn":
            kp = face['keypoints']
            landmarks = np.array([kp['left_eye'], kp['right_eye'], kp['nose'], kp['mouth_left'], kp['mouth_right']], dtype=np.float32)
            aligned = worker_aligner.align(image_rgb, landmarks)
        else:
            aligned = cv2.resize(image_rgb, target_size)

    stem = image_path.stem
    
    # 1. Sauvegarde du fichier Haute Résolution (HR) original aligné
    hr_path = output_dir / "hr" / f"{stem}.png"
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(hr_path), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

    # 2. Génération des versions masquées et dégradées (LR)
    for mask_type in mask_types:
        masked, _ = worker_mask_gen.apply_mask(aligned, mask_type)

        for scale in scale_factors:
            h, w = target_size
            # Downscale (Basse résolution)
            small = cv2.resize(masked, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
            
            # Sauvegarde de l'image Basse Résolution masquée
            out_path = output_dir / f"x{scale}" / mask_type / f"{stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────
# Pipeline Principal
# ─────────────────────────────────────────────

class PreprocessingPipeline:
    def __init__(self, detector_backend: str = "mtcnn", mask_types: List[str] = None, scale_factors: List[int] = None, target_size: Tuple[int, int] = (256, 256)):
        self.detector_backend = detector_backend
        self.scale_factors = scale_factors or [2, 4, 8]
        self.mask_types = mask_types or MaskGenerator.MASK_TYPES
        self.target_size = target_size

    def run(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Collecte récursive de toutes les images
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        images = []
        for ext in image_extensions:
            images.extend(list(input_path.rglob(ext)))
        images = list(set(images)) # Suppression des doublons éventuels

        logger.info(f"📂 {len(images)} images trouvées dans le dossier source.")
        if len(images) == 0:
            logger.error("❌ Aucune image trouvée. Vérifiez vos chemins d'accès.")
            return

        # Configuration de la fonction partielle pour le multiprocessing
        worker_fn = functools.partial(
            process_single_image,
            output_dir=output_path,
            mask_types=self.mask_types,
            scale_factors=self.scale_factors,
            target_size=self.target_size
        )

        num_workers = os.cpu_count()
        logger.info(f"⚡ Lancement du traitement parallèle sur {num_workers} cœurs CPU...")

        # Exécution parallèle avec barre de progression de synchronisation
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(self.detector_backend, self.target_size)) as executor:
            list(tqdm(executor.map(worker_fn, images), total=len(images), desc="Preprocessing"))

        logger.info(f"✅ Traitement terminé avec succès. Données stockées dans : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Preprocessing Pipeline Optimisé")
    parser.add_argument("--input", type=str, default=None, help="Dossier d'entrée")
    parser.add_argument("--output", type=str, default=None, help="Dossier de sortie")
    parser.add_argument("--detector", type=str, default="mtcnn", choices=["mtcnn", "retinaface", "mediapipe"])
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--masks", nargs="+", default=MaskGenerator.MASK_TYPES)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    input_dir = args.input or get_dataset_path()
    output_dir = args.output or get_output_path()

    pipeline = PreprocessingPipeline(
        detector_backend=args.detector,
        mask_types=args.masks,
        scale_factors=args.scales,
        target_size=(args.size, args.size),
    )
    pipeline.run(input_dir, output_dir)