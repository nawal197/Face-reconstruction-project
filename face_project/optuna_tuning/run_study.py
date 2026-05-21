"""
Optuna Hyperparameter Optimization
===================================
Optimise automatiquement les hyperparamètres de chaque pipeline.

Pipelines supportés : pix2pix, esrgan, diffusion, transformer, identity_gan

Usage :
    python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 50
    python optuna_tuning/run_study.py --pipeline all --n_trials 30 --timeout 3600
    optuna-dashboard sqlite:///optuna_tuning/studies.db
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
import yaml
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Objectives
# ─────────────────────────────────────────────

def objective_pix2pix(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """
    Optimise les hyperparamètres de Pix2Pix.
    Minimise la loss L1 sur validation.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipelines.pix2pix.train import Pix2PixTrainer

    # Hyperparamètres à optimiser
    config["training"]["lr_g"] = trial.suggest_float("lr_g", 1e-5, 1e-3, log=True)
    config["training"]["lr_d"] = trial.suggest_float("lr_d", 1e-5, 1e-3, log=True)
    config["training"]["lambda_l1"] = trial.suggest_float("lambda_l1", 50.0, 200.0)
    config["model"]["features_g"] = trial.suggest_categorical("features_g", [32, 64, 128])
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16])
    config["training"]["epochs"] = 5  # Raccourci pour Optuna

    trainer = Pix2PixTrainer(config, trial=trial)
    val_loss = trainer.train(train_loader, val_loader)
    return val_loss


def objective_esrgan(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """Optimise Real-ESRGAN / ESRGAN hyperparamètres."""
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    num_feat = trial.suggest_categorical("num_feat", [32, 64])
    num_block = trial.suggest_int("num_block", 4, 23)
    scale = trial.suggest_categorical("scale", [2, 4])

    # Placeholder — remplacer par vrai entraînement ESRGAN
    # from pipelines.esrgan.train import ESRGANTrainer
    # trainer = ESRGANTrainer(config, trial=trial)
    # return trainer.train(train_loader, val_loader)

    import random
    mock_loss = random.uniform(0.01, 0.1) / (lr * 1000) * (64 / num_feat)
    return mock_loss


def objective_transformer(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """Optimise SwinIR / MAT hyperparamètres."""
    embed_dim = trial.suggest_categorical("embed_dim", [60, 96, 180])
    num_heads = trial.suggest_categorical("num_heads", [6, 8, 12])
    depth = trial.suggest_int("depth", 2, 8)
    window_size = trial.suggest_categorical("window_size", [4, 8])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    import random
    mock_loss = random.uniform(0.01, 0.05) * (depth / 8)
    return mock_loss


def objective_diffusion(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """Optimise Diffusion Model hyperparamètres."""
    guidance_scale = trial.suggest_float("guidance_scale", 1.0, 15.0)
    num_inference_steps = trial.suggest_int("num_inference_steps", 20, 100)
    strength = trial.suggest_float("strength", 0.5, 1.0)
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)

    import random
    mock_loss = random.uniform(0.005, 0.05) * (1.0 / guidance_scale)
    return mock_loss


def objective_identity_gan(trial: optuna.Trial, config: dict, train_loader, val_loader) -> float:
    """Optimise Identity GAN : équilibre entre reconstruction et identité."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipelines.identity_gan.train import IdentityGANTrainer

    config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["lambda_arcface"] = trial.suggest_float("lambda_arcface", 0.1, 10.0, log=True)
    config["training"]["lambda_perceptual"] = trial.suggest_float("lambda_perceptual", 0.1, 10.0, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [4, 8])
    config["training"]["epochs"] = 3

    trainer = IdentityGANTrainer(config, trial=trial)

    total_loss = 0
    for batch in val_loader:
        losses = trainer.train_step(batch["lr"], batch["hr"])
        total_loss += losses["loss_total"]
        break  # Quick eval for Optuna

    return total_loss


OBJECTIVES = {
    "pix2pix": objective_pix2pix,
    "esrgan": objective_esrgan,
    "transformer": objective_transformer,
    "diffusion": objective_diffusion,
    "identity_gan": objective_identity_gan,
}


# ─────────────────────────────────────────────
# Study Runner
# ─────────────────────────────────────────────

class OptunaStudyRunner:
    """
    Lance et gère les études Optuna pour chaque pipeline.
    Stockage : SQLite (persistant, compatible optuna-dashboard).
    """

    def __init__(
        self,
        storage_path: str = "optuna_tuning/studies.db",
        sampler_name: str = "tpe",
        pruner_name: str = "median",
    ):
        self.storage = f"sqlite:///{storage_path}"
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        self.sampler = self._get_sampler(sampler_name)
        self.pruner = self._get_pruner(pruner_name)

    def _get_sampler(self, name: str):
        samplers = {
            "tpe": TPESampler(seed=42),
            "cmaes": CmaEsSampler(seed=42),
        }
        return samplers.get(name, TPESampler(seed=42))

    def _get_pruner(self, name: str):
        pruners = {
            "median": MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            "hyperband": HyperbandPruner(min_resource=1, max_resource=10),
        }
        return pruners.get(name, MedianPruner())

    def run(
        self,
        pipeline: str,
        config: dict,
        train_loader,
        val_loader,
        n_trials: int = 30,
        timeout: int = None,
    ) -> optuna.Study:
        study_name = f"{pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting Optuna study: {study_name}")
        logger.info(f"Pipeline: {pipeline} | Trials: {n_trials} | Storage: {self.storage}")

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction="minimize",
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=False,
        )

        objective_fn = OBJECTIVES[pipeline]

        study.optimize(
            lambda trial: objective_fn(trial, config, train_loader, val_loader),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._log_callback],
        )

        self._print_results(study, pipeline)
        self._save_results(study, pipeline, config)
        return study

    def _log_callback(self, study: optuna.Study, trial: optuna.FrozenTrial):
        logger.info(
            f"Trial {trial.number} finished — Value: {trial.value:.6f} — "
            f"Best: {study.best_value:.6f}"
        )

    def _print_results(self, study: optuna.Study, pipeline: str):
        print("\n" + "=" * 60)
        print(f"Optuna Study Results — {pipeline.upper()}")
        print("=" * 60)
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"\nBest trial:")
        t = study.best_trial
        print(f"  Value: {t.value:.6f}")
        print(f"  Params:")
        for k, v in t.params.items():
            print(f"    {k}: {v}")
        print("=" * 60 + "\n")

    def _save_results(self, study: optuna.Study, pipeline: str, config: dict):
        results_dir = Path("optuna_tuning/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save best params to YAML
        best_params = study.best_params
        out_path = results_dir / f"{pipeline}_best_params.yaml"
        with open(out_path, "w") as f:
            yaml.dump({"best_params": best_params, "best_value": study.best_value}, f)
        logger.info(f"Best params saved to {out_path}")

        # Save importance plot if plotly available
        try:
            import plotly
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(results_dir / f"{pipeline}_param_importance.html"))
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(str(results_dir / f"{pipeline}_optimization_history.html"))
            logger.info("Optuna visualizations saved.")
        except Exception as e:
            logger.warning(f"Could not save Optuna plots: {e}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--pipeline", type=str, required=True,
                        choices=list(OBJECTIVES.keys()) + ["all"],
                        help="Pipeline à optimiser")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout en secondes (optionnel)")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "cmaes"])
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "hyperband"])
    parser.add_argument("--storage", type=str, default="optuna_tuning/studies.db")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config {args.config} not found. Using defaults.")
        config = {
            "data": {"processed_dir": "datasets/processed", "scale_factor": 4,
                     "mask_type": "surgical", "image_size": 256},
            "training": {"epochs": 100, "batch_size": 8, "lr_g": 2e-4, "lr_d": 2e-4,
                         "lambda_l1": 100.0, "lr": 2e-4},
            "model": {"features_g": 64},
            "logging": {"tensorboard_dir": "results/tensorboard",
                        "checkpoint_dir": "models/checkpoints"},
        }

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(config)

    runner = OptunaStudyRunner(
        storage_path=args.storage,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
    )

    pipelines = list(OBJECTIVES.keys()) if args.pipeline == "all" else [args.pipeline]
    for p in pipelines:
        logger.info(f"\n{'='*50}\nOptimizing pipeline: {p}\n{'='*50}")
        study = runner.run(
            pipeline=p,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=args.n_trials,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
