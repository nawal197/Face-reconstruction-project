# Guide d’exécution du projet (Face Unmasking & Super-Resolution)

## 0) Prérequis
- Python 3.10+ conseillé
- GPU recommandé (PyTorch)
- Sur Kaggle : utilisez l’environnement notebook (GPU activé si possible)

## 1) Installation des dépendances
Depuis la racine du projet `face_unmasking_project/` :
```bash
pip install -r face_project/requirements.txt
```

> Remarque : certaines dépendances lourdes (diffusers, basicsr, etc.) peuvent prendre du temps.

## 2) Comprendre le pipeline global
1. **Prétraitement** :
   - détection/alignement visage
   - génération masques
   - génération LR (pour plusieurs `scale` et plusieurs `mask_type`)
   - stockage dans `processed_dir/{split}/hr` et `processed_dir/{split}/lr_x{scale}/{mask_type}`
2. **Entraînement** : entraînement GAN/Transformer/Diffusion selon le pipeline.
3. **Évaluation** : calcule des métriques (PSNR/SSIM/LPIPS/ArcFace/FID selon disponibilité des libs).
4. **Optuna** : recherche d’hyperparamètres (pour certains pipelines selon implémentation).
5. **Interface** : `streamlit run face_project/app/app.py`

## 3) Configuration
- `face_project/configs/default.yaml` : chemins locaux
- `face_project/configs/kaggle.yaml` : chemins Kaggle

Fichier important :
- `face_project/configs/kaggle.yaml`
  - `data.raw_dir` pointe vers : `/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq`
  - `data.processed_dir` pointe vers : `/kaggle/working/face_dataset`

## 4) Prétraitement (OBLIGATOIRE)
### 4.1 Sur Kaggle (cas normal)
Dans un notebook Kaggle (où `/kaggle/input` existe) :
```bash
python face_project/preprocessing/preprocess.py \
  --detector retinaface \
  --max-images 1000 \
  --input /kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq \
  --output /kaggle/working/face_dataset
```

Le preprocessing doit créer une structure de dossiers :
- `/kaggle/working/face_dataset/train/hr/*.jpg`
- `/kaggle/working/face_dataset/train/lr_x4/surgical/*.jpg`
- `/kaggle/working/face_dataset/val/...`
- `/kaggle/working/face_dataset/test/...`

### 4.2 Sur machine locale
Utilise `--input` vers un dossier local contenant tes images (FFHQ ou autre) :
```bash
python face_project/preprocessing/preprocess.py \
  --detector retinaface \
  --max-images 2000 \
  --input /chemin/local/vers/images \
  --output datasets/processed
```

> Si tu vois `0 images trouvées au total`, c’est que `--input` ne pointe vers aucun fichier image accessible.

## 5) Dataloaders (cohérence preprocessing ↔ training)
Le dataset attend :
- `processed_dir/{split}/hr/*.{jpg,png}`
- `processed_dir/{split}/lr_x{scale_factor}/{mask_type}/*.{jpg,png}`

Donc **ne lance l’entraînement** qu’après avoir généré `processed_dir/train` et `processed_dir/val`.

## 6) Entraînement (exemple pix2pix)
Depuis la racine :
```bash
python face_project/pipelines/pix2pix/train.py \
  --config face_project/configs/kaggle.yaml
```

Tu peux aussi tester en local :
- remplacer `kaggle.yaml` par `default.yaml`

## 7) Évaluation
```bash
python face_project/evaluation/evaluate.py \
  --config face_project/configs/default.yaml \
  --results_dir face_project/results
```

> L’évaluation charge certains checkpoints (selon implémentations disponibles) et calcule les métriques disponibles.

## 8) Optuna (recherche d’hyperparamètres)
Exemple pour pix2pix :
```bash
python face_project/optuna_tuning/run_study.py \
  --pipeline pix2pix \
  --n_trials 30
```

Dashboard :
```bash
python face_project/optuna_tuning/dashboard.py
```

Puis ouvrir : http://localhost:8080

## 9) Interface
```bash
streamlit run face_project/app/app.py
```

## 10) Checklist rapide (si ça plante)
1. `preprocess.py` : vérifier le log `images trouvées` (doit être > 0)
2. Vérifier l’existence de :
   - `processed_dir/train/hr`
   - `processed_dir/train/lr_x{scale_factor}/{mask_type}`
3. Lancer un quick sanity check dataset (optionnel) :
   - le dataloader doit renvoyer un batch non vide.


