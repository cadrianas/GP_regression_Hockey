"""Report the share of comeback shots taken with the trailing team's goalie absent."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import run_inputs
from src.diagnostic_shot_location import goalie_summary


def main():
    """Save the shared training-sample goalie summary as a small CLI report."""
    full, output = run_inputs(__doc__, "diagnostics")
    # Reuse the diagnostic helper so this standalone report cannot drift to a
    # different definition. This is the share of SELECTED SHOTS, not all games.
    summary = goalie_summary(full[full.role == "train"])
    summary.to_csv(output / "goalie_pull_frequency.csv", index=False)
    print(summary.to_string(index=False))
    print("Shot-weighted percentages through 18:30; not percentages of games or playing time.")


if __name__ == "__main__":
    main()
