# Face Unmasking & Super-Resolution — Deep Learning Project

## Description

Projet de reconstruction et amélioration de visages masqués basse résolution par Deep Learning.
Comparaison de pipelines GAN, Transformers et Diffusion Models avec optimisation des hyperparamètres via **Optuna**.

## Pipelines

| Pipeline | Architecture | Objectif |
|----------|-------------|----------|
| A | Pix2Pix | Baseline |
| B | Pix2PixHD | Haute résolution |
| C | ESRGAN + Inpainting | Séparation unmasking/SR |
| D | Real-ESRGAN | Robustesse bruit |
| E | MAT + SwinIR | Transformers |
| F | Stable Diffusion | Photoréalisme |
| G | Identity GAN | Conservation identité |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Prétraitement
python preprocessing/preprocess.py --input datasets/raw --output datasets/processed

# Entraînement
python pipelines/pix2pix/train.py --config configs/pix2pix.yaml

# Optimisation Optuna
python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 50

# Évaluation
python evaluation/evaluate.py --pipeline all --results_dir results/

# Interface
streamlit run app/app.py
```

## Structure

```
project/
├── datasets/           # Données brutes et traitées
├── preprocessing/      # Scripts de prétraitement
├── pipelines/          # 7 pipelines expérimentaux
├── optuna_tuning/      # Optimisation des hyperparamètres
├── evaluation/         # Scripts d'évaluation
├── metrics/            # PSNR, SSIM, LPIPS, ArcFace, FID
├── experiments/        # Configurations d'expériences
├── notebooks/          # Jupyter notebooks d'analyse
├── results/            # Résultats et visualisations
├── models/             # Poids sauvegardés
└── app/                # Interface Streamlit/Gradio
```
