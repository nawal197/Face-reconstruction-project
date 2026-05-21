"""
Évaluation comparative de tous les pipelines
"""

import torch
import yaml
import argparse
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def evaluate_all_pipelines(config: dict, val_loader, results_dir: str = "results"):
    from metrics.metrics import PipelineEvaluator

    evaluator = PipelineEvaluator()
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipelines_checkpoints = {
        "pix2pix": ("pipelines/pix2pix/train.py", "UNetGenerator", "models/checkpoints/pix2pix_G.pth"),
        "identity_gan": ("pipelines/identity_gan/train.py", "AttentionUNet", "models/checkpoints/identity_gan_G.pth"),
    }

    for name, (module_path, class_name, ckpt_path) in pipelines_checkpoints.items():
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            logger.warning(f"Checkpoint not found for {name}: {ckpt_path}")
            continue

        logger.info(f"Evaluating pipeline: {name}")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("module", module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model_class = getattr(mod, class_name)
            model = model_class().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            metrics = evaluator.evaluate_dataset(model, val_loader, device, name)
            results[name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    # Save comparison table
    if results:
        df = pd.DataFrame(results).T
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "comparison.csv")
        print("\n=== Pipeline Comparison ===")
        print(df.to_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from datasets.dataset import get_dataloaders
    _, val_loader = get_dataloaders(config)
    evaluate_all_pipelines(config, val_loader, args.results_dir)
