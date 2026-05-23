# Rapport détaillé — Face Unmasking & Super-Resolution (Deep Learning)

## 1) Résumé du projet
Ce projet vise la **reconstruction de visages masqués** (masques basse résolution) en combinant :
- des **modèles de restauration / image-to-image translation** (GAN type *Pix2Pix*),
- des approches **super-résolution** (via pipelines SR / GAN / diffusion — présents sous forme d’ébauches),
- une contrainte de **conservation d’identité** grâce à des pertes inspirées d’**ArcFace** (implémentée via *FaceNet/ InceptionResnetV1*),
- une optimisation des **hyperparamètres** par **Optuna**.

Le projet contient une chaîne complète :
1. **Prétraitement** (détection de face, alignement, génération de masques, dégradation LR)
2. **Dataset** supervisé (paire LR masquée → HR originale)
3. **Entraînement** par pipeline
4. **Évaluation** via des métriques perceptuelles/identitaires
5. **Interface** Streamlit (démo + contrôle)
6. **Optuna** (recherche et dashboard)

> Remarque : certaines pipelines annoncées dans le README sont actuellement marquées `# TODO` dans leur implémentation Python (pix2pixhd, esrgan, transformer, diffusion). La pipeline **Pix2Pix (A)** et **Identity GAN (G)** sont pleinement codées dans le dépôt observé.

---

## 2) Spécification fonctionnelle (ce que le code fait)

### 2.1 Prétraitement — `face_project/preprocessing/preprocess.py`
Objectif : produire des données structurées en triplets (en pratique, LR masquée + HR) prêtes pour l’entraînement.

**Fonctions / classes clés :**
- `get_dataset_path()` / `get_output_path()` : détection Kaggle (`/kaggle/input` / `/kaggle/working`).
- `detect_kaggle_structure(input_dir)` : cherche `with_mask/` et `without_mask/`.
- `FaceDetector(backend)` : backends supportés :
  - `mtcnn` (MTCNN)
  - `retinaface` (RetinaFace)
  - `mediapipe`
- `FaceAligner(target_size)` : alignement par transformation affine via `cv2.estimateAffinePartial2D` (utilise 5 landmarks).
- `MaskGenerator` : génération de masques **synthétiques** (surgical/fabric/n95/black/white/colored).
  - Crée une région binaire `mask_region` (remplissage poly).
  - Applique une couleur de masque et fait un blending `alpha=0.85` pour un rendu plus réaliste.
- `ResolutionDegrader(scale_factor)` : dégradation LR par downscale/upscale pour simuler un visage basse résolution.
- `PreprocessingPipeline.run(input_dir, output_dir)` :
  - si Kaggle : parcourt surtout `without_mask/` comme source principale
  - sinon : parcourt récursivement les images du dossier
  - produit :
    - `output_dir/hr/{stem}_{i}.png`
    - `output_dir/x{scale}/{mask_type}/{stem}_{i}.png`

**Logique de génération :**
- détection → alignement (redimensionnement/warp)
- masque synthétique → dégradation
- export de plusieurs échelles (`--scales`, default `[2,4,8]`) et plusieurs types de masques.

---

### 2.2 Dataset — `face_project/datasets/dataset.py`
Objectif : charger les paires **LR masquée (entrée)** et **HR originale (cible)**.

**Classe :**
- `FaceDataset(lr_dir, hr_dir, scale_factor, mask_type, image_size, augment)`
  - `self.lr_dir = Path(lr_dir)/x{scale_factor}/{mask_type}`
  - `self.hr_dir = Path(hr_dir)/hr`
  - `self.files = sorted(self.hr_dir.glob('*.png'))`

**Transfomations :**
- LR : resize `(image_size/scale_factor)` → `ToTensor()` → normalisation `[0.5,0.5,0.5]`.
- HR : resize `image_size` → `ToTensor()` → normalisation.
- Augmentation : `RandomHorizontalFlip(p=0.5)` appliquée de façon synchronisée (même seed).

**Retour `__getitem__` :**
```python
{
  'lr': lr_tensor,
  'hr': hr_tensor,
  'name': hr_path.stem,
}
```

> Note technique : dans `get_dataloaders`, `lr_dir` et `hr_dir` utilisent `config['data']['processed_dir']`. Or `FaceDataset` ajoute en interne `.../hr` et `.../x{scale}/mask_type`. C’est cohérent avec la sortie du preprocessing.

---

## 3) Pipelines de modèles (architectures)

### 3.1 Pipeline A — Pix2Pix (`face_project/pipelines/pix2pix/train.py`)
**But :** reconstruction image-to-image avec GAN conditionnel.

**Architecture :**
- **Générateur** : `UNetGenerator`
  - encodeur (downsampling) et décodeur (upsampling)
  - skip connections via concaténation
  - activation finale `Tanh()`
- **Discriminateur** : `PatchGAN`
  - prend concat(`lr`, `y`) sur le canal (donc `in_channels=6`)
  - convolution downsampling pour produire une carte de logits.

**Losses :**
- adversarial GAN : `BCEWithLogitsLoss`
- reconstruction : `L1Loss(fake, hr) * lambda_l1`

**Entraînement :**
- boucle par epochs
- `train_step` :
  1. entraîner D (réel vs faux)
  2. entraîner G (faux pour tromper D + L1)
- logging TensorBoard `Train/loss_*`
- sauvegarde checkpoints :
  - `pix2pix_G.pth`
  - `pix2pix_D.pth`

**Intégration Optuna :**
- `Pix2PixTrainer(config, trial=None)` : si `trial` est fourni, les hyperparamètres (`lr_g`, `lr_d`, `lambda_l1`, `features_g`) sont échantillonnés.

---

### 3.2 Pipeline G — Identity Preserving GAN (`face_project/pipelines/identity_gan/train.py`)
**But :** reconstruire en préservant au mieux l’identité faciale.

**Composants :**
- `ArcFaceLoss` :
  - tente d’utiliser `facenet_pytorch.InceptionResnetV1(pretrained='vggface2')`
  - calcule `cosine_similarity(emb_fake, emb_real)` après redimensionnement en 160×160
  - perte : `1 - mean(cos_sim)`
  - si dépendance manquante : `encoder=None` et perte renvoie 0 (ArcFace loss désactivée).
- `PerceptualLoss` :
  - charge `torchvision.models.vgg16(pretrained=True).features`
  - utilise les 16 premiers niveaux
  - `L1Loss` entre features.
- `AttentionUNet` :
  - U-Net avec **gates d’attention** pour moduler le flux d’information
  - sortie finale `Tanh()`

**Entraînement (step) :**
- Upsample de `lr_img` vers la taille HR
- `loss_total = loss_l1 + lambda_perceptual * loss_perc + lambda_arcface * loss_id`
- Optimisation via `Adam`.

**Intégration Optuna :**
- hyperparamètres : `lr`, `lambda_arcface`, `lambda_perceptual`, `base_features`.

**Checkpointing :**
- le fichier observé ne montre pas de sauvegarde checkpoints dans la classe (à vérifier selon la suite du code). Toutefois, l’évaluation s’attend à un fichier `models/checkpoints/identity_gan_G.pth`.

---

### 3.3 Pipelines incomplètes / TODO
Dans le dépôt observé, ces pipelines sont actuellement des marqueurs `# TODO` :
- `pipelines/pix2pixhd/train.py`
- `pipelines/esrgan/train.py`
- `pipelines/transformer/train.py`
- `pipelines/diffusion/train.py`

Le README annonce également des approches Pix2PixHD, ESRGAN+Inpainting, MAT/SwinIR, Stable Diffusion, mais le code d’entraînement n’apparaît pas implémenté dans l’état actuel.

---

## 4) Évaluation — `face_project/evaluation/evaluate.py` + `face_project/metrics/metrics.py`

### 4.1 Évaluation comparative
`evaluation/evaluate.py` :
- crée un `PipelineEvaluator`
- teste un ensemble réduit de pipelines (selon checkpoints présents) :
  - `pix2pix`
  - `identity_gan`
- en absence de checkpoint : warning, pipeline ignorée.

Le code tente de charger dynamiquement le modèle via `importlib.util.spec_from_file_location` et de faire `model_class().load_state_dict(...)`.

Résultats :
- tableau CSV `results/comparison.csv`

---

### 4.2 Métriques disponibles — `metrics/metrics.py`
Métriques implémentées :
- `compute_psnr(pred, target)`
- `compute_ssim(pred, target)`
  - via `piq` si disponible, sinon fallback.
- `LPIPSMetric(net='vgg')`
  - désactivé si `lpips` indisponible
- `ArcFaceSimilarity` (via `facenet_pytorch`)
- `FIDMetric` (via `pytorch_fid`), implémentation présente mais non intégrée à la fonction `evaluate_batch`.

**Normalisation :**
- `pred_norm = (pred * 0.5 + 0.5).clamp(0,1)`
- `target_norm = (target * 0.5 + 0.5).clamp(0,1)`
- LPIPS utilise des tenseurs re-mappés sur [-1,1] : `pred_norm*2 - 1`.

**Sortie :**
- moyenne par dataset et impression console.

---

## 5) Optuna — Hyperparamètres et comparaison

### 5.1 Optuna pour un pipeline — `optuna_tuning/run_study.py`
Fonctionnalités :
- lance `optuna.create_study(..., storage='sqlite:///...')`
- support de pruners (`MedianPruner`, `HyperbandPruner`) et samplers (`TPESampler`, `CmaEsSampler`)

**Objectives :**
- `objective_pix2pix` : optimisation réelle (entraînement court `epochs=5`).
- `objective_identity_gan` : optimisation via `train_step` sur 1 batch (échantillonnage rapide).
- `objective_esrgan`, `objective_transformer`, `objective_diffusion` : actuellement **mock_loss** (placeholder).

Sauvegardes :
- YAML : `{pipeline}_best_params.yaml`
- HTML (si plotly disponible) : importance + courbe d’optimisation

---

### 5.2 Comparaison multi-pipelines — `optuna_tuning/multi_pipeline_study.py`
- crée un study `multi_pipeline_comparison`
- optimise un paramètre discret `pipeline` parmi :
  `pix2pix, esrgan, transformer, diffusion, identity_gan`
- réutilise les `objective_*` via import dans `run_study.py`
- sauvegarde : `multi_pipeline_best.yaml`

---

### 5.3 Dashboard Optuna — `optuna_tuning/dashboard.py`
- exécute `python -m optuna_dashboard` sur `optuna_tuning/studies.db`
- possibilité `--list` pour afficher les studies.

---

## 6) Interface — Streamlit

### 6.1 `face_project/app/app.py`
L’interface permet :
- sélection d’un pipeline (A → G)
- choix du facteur de dégradation (SR scale) et du type de masque
- upload d’une image
- génération simulée d’une image masquée LR (placeholder)
- bouton “Reconstruire” qui renvoie actuellement **une copie redimensionnée** (aucun modèle chargé)
- affichage de métriques **fictives** (valeurs démonstration)
- tableau de comparaison “demo_data” (fictif)

> Conclusion : l’UI est fonctionnelle pour la démonstration UX, mais pas encore branchée sur les modèles entraînés.

---

## 7) Configuration

### 7.1 `configs/default.yaml`
- spécifie `data` (raw_dir, processed_dir, scale_factor, mask_type, image_size)
- `training` (epochs, batch_size, lr_g, lr_d, lambda_l1, etc.)
- `model.features_g`
- logging (tensorboard_dir, checkpoint_dir, wandb_project)

### 7.2 `configs/kaggle.yaml`
- chemins Kaggle (`/kaggle/input/...`, `/kaggle/working/...`)
- variante sur les clés training (lambda_perceptual absent)

---

## 8) Structure du projet (vue d’ensemble)
D’après le dépôt :
- `datasets/` : chargement supervise (LR/HR)
- `preprocessing/` : génération HR + LR masquée
- `pipelines/` : entraînements par famille de modèles
- `metrics/` : PSNR/SSIM/LPIPS/ArcFace + banc d’essai FID
- `evaluation/` : comparaison pipeline (limité aux pipelines avec code + checkpoints)
- `optuna_tuning/` : HPO + dashboard + résultats
- `experiments/` : structure d’expériences (exp1..exp6)
- `app/` : interface Streamlit

---

## 9) Études / expériences prévues — `experiments/experiments.py`
- `exp1` : comparaison Pix2Pix vs Pix2PixHD vs ESRGAN (scale=4)
- `exp2` : CNN vs Transformers vs Diffusion
- `exp3` : impact de la résolution x2/x4/x8
- `exp4` : masques artificiels vs réels (surgical/fabric/n95/real) — attente pipeline(s) adaptées
- `exp5` : impact Identity loss (with/without ArcFace)
- `exp6` : ablation identity_gan (full/no_perceptual/no_gan/no_arcface)

---

## 10) Limites et points à améliorer (constat direct du code)
1. **Pipelines annoncées non implémentées** : pix2pixhd, esrgan, diffusion, transformer sont marquées TODO.
2. **Optuna pour pipelines TODO** : objectifs ESRGAN/Transformer/Diffusion utilisent `mock_loss` (pas de vrai entraînement).
3. **Évaluation** : ne gère que pix2pix + identity_gan (checkpoints attendus) ; FID n’est pas calculé dans `evaluate_batch`.
4. **UI** : pas encore branchée sur les modèles réels → reconstruction et métriques sont placeholders.
5. **Mécanisme de checkpoint identity_gan** : la classe observée ne montre pas la sauvegarde, alors que l’évaluation attend un checkpoint précis.

---

## 11) Recommandations pour une version “rapport de soutenance”
- Ajouter dans ce rapport :
  - un tableau “Pipeline → implémentée / non implémentée”
  - des exemples visuels (HR, LR, masque, reconstructions)
  - un exemple de logs Optuna (meilleurs trials, courbes)
  - une section “Expériences réalisées” (au lieu que seulement “prévues”) 

---

## 12) Annexes — Commandes d’exécution (telles que prévues par le README)
- Prétraitement :
  ```bash
  python preprocessing/preprocess.py --input datasets/raw --output datasets/processed
  ```
- Entraînement Pix2Pix :
  ```bash
  python pipelines/pix2pix/train.py --config configs/pix2pix.yaml
  ```
- Optuna Pix2Pix :
  ```bash
  python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 50
  ```
- Évaluation :
  ```bash
  python evaluation/evaluate.py --pipeline all --results_dir results/
  ```
- UI :
  ```bash
  streamlit run app/app.py
  ```

