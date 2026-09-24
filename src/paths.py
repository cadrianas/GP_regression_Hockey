"""Repository paths. Importing this module does not create directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Honor existing checkouts using Data/, including on case-sensitive filesystems.
DATA_DIR = ROOT / "Data" if (ROOT / "Data").is_dir() else ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "Results"
