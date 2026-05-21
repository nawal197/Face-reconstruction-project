"""
Optuna Dashboard — Visualisation interactive des études
=======================================================
Lance le tableau de bord Optuna pour explorer les résultats.

Usage :
    python optuna_tuning/dashboard.py
    # Puis ouvrir http://localhost:8080

Ou directement :
    optuna-dashboard sqlite:///optuna_tuning/studies.db --host 0.0.0.0 --port 8080
"""

import subprocess
import sys
from pathlib import Path


STORAGE = "sqlite:///optuna_tuning/studies.db"


def launch_dashboard(host: str = "0.0.0.0", port: int = 8080):
    db_path = Path("optuna_tuning/studies.db")
    if not db_path.exists():
        print(f"[ERROR] No studies found at {db_path}")
        print("Run some Optuna trials first:")
        print("  python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 10")
        sys.exit(1)

    print(f"Launching Optuna Dashboard at http://{host}:{port}")
    print(f"Storage: {STORAGE}")
    print("Press Ctrl+C to stop.\n")

    cmd = [
        sys.executable, "-m", "optuna_dashboard",
        STORAGE,
        "--host", host,
        "--port", str(port),
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError:
        print("[ERROR] optuna-dashboard not installed.")
        print("Install: pip install optuna-dashboard")


def list_studies():
    """List all studies in the database."""
    import optuna
    try:
        studies = optuna.get_all_study_summaries(storage=STORAGE)
        print(f"\n{'='*60}")
        print(f"Studies in {STORAGE}")
        print(f"{'='*60}")
        for s in studies:
            print(f"  Study: {s.study_name}")
            print(f"    Direction: {s.direction}")
            print(f"    N trials: {s.n_trials}")
            if s.best_trial:
                print(f"    Best value: {s.best_trial.value:.6f}")
            print()
    except Exception as e:
        print(f"Error reading studies: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--list", action="store_true", help="List all studies")
    args = parser.parse_args()

    if args.list:
        list_studies()
    else:
        launch_dashboard(args.host, args.port)
