from pathlib import Path

# Root of the project
ROOT = Path(__file__).resolve().parent.parent

# Standard directories
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Ensure they exist
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)