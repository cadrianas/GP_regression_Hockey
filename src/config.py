"""Season selection. Years refer to the first year of an NHL season."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.paths import DATA_DIR, ROOT

DEFAULT_CONFIG = ROOT / "configs" / "study.json"


def season_label(year):
    return f"{year}-{str(year + 1)[-2:]}"


@dataclass(frozen=True)
class StudyConfig:
    train_start: int = 2014
    recent_start: int = 2022
    train_end: int = 2023
    evaluation_season: int = 2024
    evaluation_status: str = "complete"

    def __post_init__(self):
        years = [self.train_start, self.recent_start, self.train_end, self.evaluation_season]
        if any(type(y) is not int or not 2007 <= y <= 2100 for y in years):
            raise ValueError("Season years must be integers between 2007 and 2100")
        if not self.train_start < self.recent_start <= self.train_end < self.evaluation_season:
            raise ValueError("Require train_start < recent_start <= train_end < evaluation_season")
        if self.evaluation_status not in {"complete", "season-to-date", "snapshot"}:
            raise ValueError("Evaluation status must be complete, season-to-date, or snapshot")

    def seasons(self, data_dir=DATA_DIR):
        years = list(range(self.train_start, self.train_end + 1)) + [self.evaluation_season]
        return {Path(data_dir) / f"shots_{year}.csv": {
            "year": year, "label": season_label(year),
            "era": "validate" if year == self.evaluation_season else "early" if year < self.recent_start else "recent",
            "role": "validate" if year == self.evaluation_season else "train",
        } for year in years}

    def to_dict(self):
        return asdict(self)


def add_config_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for field in ["train_start", "recent_start", "train_end", "evaluation_season"]:
        parser.add_argument("--" + field.replace("_", "-"), type=int)
    parser.add_argument("--evaluation-status", choices=["complete", "season-to-date", "snapshot"])


def config_from_args(args):
    values = json.loads(args.config.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    for key in StudyConfig.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    # A new evaluation year does not inherit the published study's completeness claim.
    if args.evaluation_season is not None and args.evaluation_status is None:
        values["evaluation_status"] = "snapshot"
    return StudyConfig(**values)


def sample_labels(full):
    """Derive display labels from the actual sample, including partial-season status."""
    labels = {}
    for era, frame in full.groupby("era"):
        seasons = sorted(frame.season.unique())
        labels[era] = seasons[0] if len(seasons) == 1 else f"{seasons[0]} through {seasons[-1]}"
    status = full.attrs.get("evaluation_status", "snapshot")
    if status != "complete":
        labels["validate"] += f" ({status})"
    return labels
