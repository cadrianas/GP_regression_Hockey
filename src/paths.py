"""Repository paths. Importing this module does not create directories."""
from pathlib import Path

# Resolve from this file, so paths do not depend on the caller's working folder.
ROOT = Path(__file__).resolve().parent.parent
# Honor existing checkouts using Data/, including on case-sensitive filesystems.
DATA_DIR = ROOT / "Data" if (ROOT / "Data").is_dir() else ROOT / "data"
# Retained for serialised-model workflows; the current GP scripts save tables
# and figures and do not automatically load a previously stored model.
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "Results"
