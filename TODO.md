# TODO - Optimisation cohérence & perf (mémoire / temps)

## Phase 1 — Cohérence preprocessing ↔ dataset (corriger avant d’optimiser)
- [x] Lire et corriger `face_project/datasets/dataset.py` :
  - [x] Utiliser `processed_dir/{split}` (train/val/test)
  - [x] Chercher `*.jpg` (et optionnellement `*.png`) au lieu de `*.png` uniquement
  - [x] Propager `split` dans `get_dataloaders(config)`
- [ ] Vérifier que les dossiers produits par `face_project/preprocessing/preprocess.py` sont bien ceux consommés.

## Phase 2 — Mesure perf (instrumentation)
- [ ] Ajouter mesures de temps preprocessing (detect/align/mask/resize/write)
- [ ] Ajouter mesures temps/itération training + max GPU memory (si CUDA)


## Phase 3 — Réduction temps/mémoire preprocessing
- [ ] Réduire les copies dans `MaskGenerator.apply_mask_inplace()` (actuellement `image.copy()`)
- [ ] Réduire I/O : évaluer passage vers format indexé (LMDB/WebDataset) ou lazy generation LR

## Phase 4 — Optimisation training/dataloader
- [ ] Ajuster DataLoader : `persistent_workers`, `prefetch_factor`, `num_workers` adapté Kaggle
- [ ] Vérifier mixed precision dans `pipelines/*/train.py`

## Phase 5 — Validation
- [ ] Test sur petit subset (max_images=1000) :
  - [ ] dataset non vide
  - [ ] un batch passe sans erreurs
  - [ ] perf (images/s) et mémoire GPU acceptable

