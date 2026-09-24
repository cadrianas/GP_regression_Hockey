"""Inspect individual season files without requiring the full training sample."""
from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import filter_shots
from src.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    files = args.files or sorted(DATA_DIR.glob("shots_*.csv"))
    if not files:
        parser.exit(2, f"No shot CSVs found in {DATA_DIR}\n")
    for path in files:
        df = pd.read_csv(path)
        years = df["season"].unique()
        if len(years) != 1:
            raise ValueError(f"Expected one season in {path}, found {years}")
        selected = filter_shots(df, int(years[0]))
        print(f"{path.name}: season={years[0]}, rows={len(df):,}, "
              f"qualifying shots={len(selected):,}, games={selected.game_key.nunique():,}, "
              f"goalie-absent shots={selected.is_goalie_pull.sum():,}")


if __name__ == "__main__":
    main()
