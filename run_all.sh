#!/usr/bin/env bash
set -euo pipefail

python3 -m unittest discover -s tests -v
python3 src/decomposition.py --output-dir Results/corrected/decomposition
python3 src/GP_regression.py --output-dir Results/corrected/gp
python3 src/diagnostic_shot_location.py --output-dir Results/corrected/diagnostics
python3 src/results.py --output-dir Results/corrected/model_comparison
python3 src/noise_sensitivity.py --output-dir Results/corrected
python3 src/talk_sensitivity.py --output-dir Results/corrected/talk
