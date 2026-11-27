"""Download selected seasons and run the analysis in a separate output folder."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from src.config import add_config_arguments, config_from_args, sample_labels
from src.download import atomic_json, ensure_data, sha256
from src.paths import DATA_DIR, ROOT


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def execute(config, data_dir, output, refresh=False, offline=False, sensitivities=False):
    output = Path(output)
    # Fail rather than mixing new figures with an earlier or incomplete run.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config.to_dict(), "offline": offline, "sensitivities": sensitivities,
                "source": "MoneyPuck.com", "inputs": [], "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]},
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "src").glob('*.py')}}
    for key, command in [("git_commit", ["git", "rev-parse", "HEAD"]),
                         ("git_status", ["git", "status", "--porcelain"])]:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        manifest[key] = result.stdout.strip() if result.returncode == 0 else None
    atomic_json(output / "config.json", config.to_dict())
    atomic_json(output / "manifest.json", manifest)
    with (output / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            records = ensure_data(config, data_dir, refresh, offline)
            manifest["inputs"] = records
            atomic_json(output / "manifest.json", manifest)
            from src.analysis import load_dataset
            from src import decomposition, GP_regression, diagnostic_shot_location, results
            full = load_dataset(config.seasons(data_dir))
            full.attrs["evaluation_status"] = config.evaluation_status
            sample = full.groupby(["era", "season"]).agg(shots=("xGoal", "size"), games=("game_key", "nunique"))
            sample.to_csv(output / "sample_summary.csv")
            stages = [("decomposition", decomposition), ("gp", GP_regression),
                      ("diagnostics", diagnostic_shot_location), ("model_comparison", results)]
            if sensitivities:
                from src import noise_sensitivity, talk_sensitivity
                stages += [("noise_sensitivity", noise_sensitivity), ("talk", talk_sensitivity)]
            for name, module in stages:
                destination = output / name
                destination.mkdir()
                print(f"Running {name}...", flush=True)
                module.analyze(full, destination)
            labels = sample_labels(full)
            import pandas as pd
            metrics = pd.read_csv(output / "model_comparison/model_comparison.csv")
            report = (f"# Analysis of {labels['validate']}\n\n"
                      f"Early training: {labels['early']}. Recent training: {labels['recent']}.\n\n"
                      "Data: [MoneyPuck](https://moneypuck.com/data.htm). Non-commercial use; attribution required.\n\n"
                      f"Evaluation status is user-declared: **{config.evaluation_status}**. "
                      "The downloader does not certify that every scheduled game is present. "
                      "See manifest.json for the exact files and acquisition times.\n\n"
                      "| Training era | Model | RMSE | Prediction interval coverage (%) |\n"
                      "| --- | --- | ---: | ---: |\n")
            for _, row in metrics.iterrows():
                report += f"| {row['era']} | {row['model']} | {row['rmse']:.5f} | {row['prediction_coverage_pct']:.1f} |\n"
            report += ("\nCoverage describes this sample; it does not establish future calibration. "
                       "Era-boundary comparisons, when enabled, use the evaluation season and are exploratory.\n\n"
                       "![Shot quality by era](gp/figures/fig7_gp_comparison.png)\n\n"
                       "![Observed and predicted shot quality](gp/figures/fig6b_gp_recent.png)\n")
            (output / "summary.md").write_text(report)
            manifest.update(status="complete", shots=len(full), games=int(full.game_key.nunique()))
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(manifest["error"], file=sys.stderr)
            raise
        finally:
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-evaluation", action="store_true")
    parser.add_argument("--sensitivities", action="store_true", help="Also run manpower, era-boundary, and noise checks")
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output_dir or ROOT / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
        execute(config, args.data_dir, output, args.refresh_evaluation, args.offline, args.sensitivities)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Results: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
