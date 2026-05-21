"""
Optuna — Comparaison Multi-Pipeline
Cherche le meilleur pipeline + hyperparamètres en un seul study.
"""

import optuna
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def objective_multi_pipeline(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """
    Espace de recherche global : choisit d'abord le pipeline,
    puis optimise ses hyperparamètres spécifiques.
    """
    pipeline = trial.suggest_categorical(
        "pipeline", ["pix2pix", "esrgan", "transformer", "diffusion", "identity_gan"]
    )

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if pipeline == "pix2pix":
        from run_study import objective_pix2pix
        return objective_pix2pix(trial, config, train_loader, val_loader)

    elif pipeline == "esrgan":
        from run_study import objective_esrgan
        return objective_esrgan(trial, config, train_loader, val_loader)

    elif pipeline == "transformer":
        from run_study import objective_transformer
        return objective_transformer(trial, config, train_loader, val_loader)

    elif pipeline == "diffusion":
        from run_study import objective_diffusion
        return objective_diffusion(trial, config, train_loader, val_loader)

    elif pipeline == "identity_gan":
        from run_study import objective_identity_gan
        return objective_identity_gan(trial, config, train_loader, val_loader)

    return float("inf")


def run_multi_pipeline_study(config, train_loader, val_loader, n_trials=100):
    """Lance un study qui compare tous les pipelines."""
    storage = "sqlite:///optuna_tuning/studies.db"
    Path("optuna_tuning").mkdir(exist_ok=True)

    study = optuna.create_study(
        study_name="multi_pipeline_comparison",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective_multi_pipeline(trial, config, train_loader, val_loader),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Analyse par pipeline
    print("\n=== Multi-Pipeline Comparison Results ===")
    pipeline_results = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            p = trial.params.get("pipeline", "unknown")
            if p not in pipeline_results:
                pipeline_results[p] = []
            pipeline_results[p].append(trial.value)

    for p, values in sorted(pipeline_results.items(), key=lambda x: min(x[1])):
        print(f"{p:20s} — Best: {min(values):.6f} | Mean: {sum(values)/len(values):.6f} | Trials: {len(values)}")

    # Save comparison
    results_dir = Path("optuna_tuning/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "multi_pipeline_best.yaml", "w") as f:
        yaml.dump({
            "best_pipeline": study.best_trial.params.get("pipeline"),
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f)

    return study


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    config = {
        "data": {"processed_dir": "datasets/processed", "scale_factor": 4,
                 "mask_type": "surgical", "image_size": 256},
        "training": {"epochs": 5, "batch_size": 8, "lr_g": 2e-4, "lr_d": 2e-4,
                     "lambda_l1": 100.0, "lr": 2e-4},
        "model": {"features_g": 64},
        "logging": {"tensorboard_dir": "results/tensorboard",
                    "checkpoint_dir": "models/checkpoints"},
    }

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(config)

    run_multi_pipeline_study(config, train_loader, val_loader, n_trials=args.n_trials)
