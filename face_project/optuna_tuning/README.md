# Optuna — Optimisation des Hyperparamètres

## Vue d'ensemble

Ce module intègre **Optuna** pour l'optimisation automatique des hyperparamètres de chaque pipeline.

## Fichiers

| Fichier | Rôle |
|---------|------|
| `run_study.py` | Lance une étude pour un pipeline spécifique |
| `multi_pipeline_study.py` | Compare tous les pipelines en un seul study |
| `dashboard.py` | Lance le tableau de bord interactif |
| `results/` | Meilleurs hyperparamètres sauvegardés (YAML + HTML) |

## Hyperparamètres optimisés par pipeline

### Pix2Pix
- `lr_g` — learning rate générateur (1e-5 → 1e-3, log)
- `lr_d` — learning rate discriminateur (1e-5 → 1e-3, log)
- `lambda_l1` — poids de la L1 loss (50 → 200)
- `features_g` — taille features U-Net (32, 64, 128)
- `batch_size` — taille de batch (4, 8, 16)

### ESRGAN
- `lr` — learning rate (1e-5 → 1e-3, log)
- `num_feat` — features de base (32, 64)
- `num_block` — nombre de RRDB blocks (4 → 23)
- `scale` — facteur SR (2, 4)

### Transformer (SwinIR/MAT)
- `embed_dim` — dimension d'embedding (60, 96, 180)
- `num_heads` — têtes d'attention (6, 8, 12)
- `depth` — profondeur du transformer (2 → 8)
- `window_size` — taille de fenêtre (4, 8)

### Diffusion
- `guidance_scale` — guidance classifier-free (1.0 → 15.0)
- `num_inference_steps` — steps de débruitage (20 → 100)
- `strength` — force d'inpainting (0.5 → 1.0)

### Identity GAN
- `lr` — learning rate (1e-5 → 1e-3, log)
- `lambda_arcface` — poids ArcFace loss (0.1 → 10.0, log)
- `lambda_perceptual` — poids perceptual loss (0.1 → 10.0, log)

## Utilisation

```bash
# Optimiser un pipeline
python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 50

# Comparer tous les pipelines
python optuna_tuning/run_study.py --pipeline all --n_trials 30

# Avec timeout (en secondes)
python optuna_tuning/run_study.py --pipeline esrgan --n_trials 100 --timeout 3600

# Sampler alternatif
python optuna_tuning/run_study.py --pipeline transformer --sampler cmaes --pruner hyperband

# Lancer le dashboard
python optuna_tuning/dashboard.py
# → http://localhost:8080

# Lister les études
python optuna_tuning/dashboard.py --list

# Comparaison multi-pipeline
python optuna_tuning/multi_pipeline_study.py --n_trials 100
```

## Samplers disponibles

| Sampler | Usage |
|---------|-------|
| `tpe` (défaut) | Tree-structured Parzen Estimator — efficace, recommandé |
| `cmaes` | CMA-ES — bon pour espaces continus |

## Pruners disponibles

| Pruner | Usage |
|--------|-------|
| `median` (défaut) | Élimine les trials sous la médiane |
| `hyperband` | Hyperband scheduling |

## Résultats

Les meilleurs hyperparamètres sont sauvegardés dans `optuna_tuning/results/` :
- `{pipeline}_best_params.yaml` — paramètres optimaux
- `{pipeline}_param_importance.html` — importance des hyperparamètres
- `{pipeline}_optimization_history.html` — historique de convergence
- `multi_pipeline_best.yaml` — meilleur pipeline global

## Dashboard Optuna

```bash
# Installation
pip install optuna-dashboard

# Lancement direct
optuna-dashboard sqlite:///optuna_tuning/studies.db

# Ou via le script
python optuna_tuning/dashboard.py --port 8080
```

Le dashboard permet de visualiser :
- Historique d'optimisation
- Importance des hyperparamètres
- Corrélations entre paramètres
- Courbes de Pareto (multi-objectif)
- Détails de chaque trial
