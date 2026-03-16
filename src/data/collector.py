import csv
from pathlib import Path

import pandas as pd

import config
from src.data.models import HistoricalMatchup


def load_historical_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or config.HISTORICAL_DIR / "tournament_results.csv"
    return pd.read_csv(path)


def parse_historical_matchups(df: pd.DataFrame) -> list[HistoricalMatchup]:
    matchups = []
    for _, row in df.iterrows():
        matchups.append(
            HistoricalMatchup(
                year=int(row["year"]),
                round_name=row["round"],
                region=row["region"],
                seed_a=int(row["seed_a"]),
                team_a=row["team_a"],
                score_a=int(row["score_a"]),
                seed_b=int(row["seed_b"]),
                team_b=row["team_b"],
                score_b=int(row["score_b"]),
            )
        )
    return matchups


def save_historical_csv(matchups: list[HistoricalMatchup], path: Path | None = None):
    path = path or config.HISTORICAL_DIR / "tournament_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["year", "round", "region", "seed_a", "team_a", "score_a", "seed_b", "team_b", "score_b"]
        )
        for m in matchups:
            writer.writerow(
                [m.year, m.round_name, m.region, m.seed_a, m.team_a, m.score_a, m.seed_b, m.team_b, m.score_b]
            )
