from __future__ import annotations

import pandas as pd

import config
from src.data.collector import load_historical_csv

_KNOWN_SEED_WIN_RATES: dict[tuple[int, int], float] = {
    (1, 16): 0.993, (2, 15): 0.943, (3, 14): 0.857, (4, 13): 0.793,
    (5, 12): 0.647, (6, 11): 0.623, (7, 10): 0.607, (8, 9): 0.513,
    (1, 8): 0.80, (1, 9): 0.86, (2, 7): 0.71, (2, 10): 0.77,
    (3, 6): 0.58, (3, 11): 0.69, (4, 5): 0.55, (4, 12): 0.72,
    (4, 13): 0.793, (1, 4): 0.70, (1, 5): 0.78, (2, 3): 0.54,
    (2, 6): 0.65, (1, 2): 0.52, (1, 3): 0.63,
}


class HistoricalAnalyzer:
    def __init__(self, csv_path=None):
        self.df = load_historical_csv(csv_path)
        self._precompute()

    def _precompute(self):
        df = self.df.copy()
        df["higher_seed"] = df[["seed_a", "seed_b"]].min(axis=1)
        df["lower_seed"] = df[["seed_a", "seed_b"]].max(axis=1)
        df["higher_seed_won"] = (
            ((df["seed_a"] < df["seed_b"]) & (df["score_a"] > df["score_b"]))
            | ((df["seed_b"] < df["seed_a"]) & (df["score_b"] > df["score_a"]))
        )
        df.loc[df["seed_a"] == df["seed_b"], "higher_seed_won"] = df["score_a"] > df["score_b"]
        self._computed = df

    def seed_matchup_win_rate(self, seed_a: int, seed_b: int) -> float:
        higher = min(seed_a, seed_b)
        lower = max(seed_a, seed_b)

        key = (higher, lower)
        if key in _KNOWN_SEED_WIN_RATES:
            base = _KNOWN_SEED_WIN_RATES[key]
        else:
            base = self._compute_from_data(higher, lower)

        if seed_a == higher:
            return base
        return 1.0 - base

    def _compute_from_data(self, higher_seed: int, lower_seed: int) -> float:
        subset = self._computed[
            (self._computed["higher_seed"] == higher_seed)
            & (self._computed["lower_seed"] == lower_seed)
        ]
        if len(subset) == 0:
            diff = lower_seed - higher_seed
            return min(0.95, 0.5 + diff * 0.03)
        return subset["higher_seed_won"].mean()

    def upset_rate_by_round(self) -> dict[str, float]:
        results = {}
        for round_name in self._computed["round"].unique():
            rdf = self._computed[self._computed["round"] == round_name]
            if len(rdf) > 0:
                results[round_name] = 1.0 - rdf["higher_seed_won"].mean()
        return results

    def seed_advancement_rates(self) -> dict[int, dict[str, float]]:
        rates = {}
        for seed in range(1, 17):
            rates[seed] = {}
            for round_name in config.ROUNDS:
                subset = self._computed[
                    (self._computed["round"] == round_name)
                    & (
                        (self._computed["seed_a"] == seed)
                        | (self._computed["seed_b"] == seed)
                    )
                ]
                if len(subset) == 0:
                    continue
                won = subset[
                    ((subset["seed_a"] == seed) & (subset["score_a"] > subset["score_b"]))
                    | ((subset["seed_b"] == seed) & (subset["score_b"] > subset["score_a"]))
                ]
                rates[seed][round_name] = len(won) / len(subset) if len(subset) > 0 else 0.0
        return rates

    def average_margin_by_seed_diff(self) -> dict[int, float]:
        df = self._computed.copy()
        df["seed_diff"] = df["lower_seed"] - df["higher_seed"]
        df["margin"] = abs(df["score_a"] - df["score_b"])
        return df.groupby("seed_diff")["margin"].mean().to_dict()

    def get_seed_win_rate_lookup(self) -> dict[tuple[int, int], float]:
        lookup = dict(_KNOWN_SEED_WIN_RATES)
        for (higher, lower), group in self._computed.groupby(["higher_seed", "lower_seed"]):
            key = (int(higher), int(lower))
            if key not in lookup and len(group) >= 3:
                lookup[key] = group["higher_seed_won"].mean()
        return lookup

    def team_tournament_experience(self, team_names: list[str], lookback_years: int = 10) -> dict[str, float]:
        max_year = self.df["year"].max()
        cutoff = max_year - lookback_years
        recent = self.df[self.df["year"] > cutoff]

        csv_teams = set(recent["team_a"].unique()) | set(recent["team_b"].unique())

        bracket_to_csv = {}
        for bracket_name in team_names:
            if bracket_name in csv_teams:
                bracket_to_csv[bracket_name] = bracket_name
                continue
            for csv_name in csv_teams:
                if (bracket_name.lower() in csv_name.lower()
                        or csv_name.lower() in bracket_name.lower()):
                    bracket_to_csv[bracket_name] = csv_name
                    break

        _NAME_OVERRIDES = {
            "UConn": "Connecticut",
            "Miami FL": "Miami (FL)",
            "UCF": "Central Florida",
            "SMU": "Southern Methodist",
            "VCU": "Virginia Commonwealth",
            "Saint Mary's": "Saint Mary's (CA)",
            "Penn": "Pennsylvania",
            "McNeese": "McNeese State",
            "UMBC": "Maryland-Baltimore County",
            "BYU": "Brigham Young",
            "Cal Baptist": "California Baptist",
        }
        for bracket_name, csv_name in _NAME_OVERRIDES.items():
            if bracket_name in team_names:
                bracket_to_csv[bracket_name] = csv_name

        result = {}
        for bracket_name in team_names:
            csv_name = bracket_to_csv.get(bracket_name)
            if not csv_name:
                result[bracket_name] = 0.0
                continue
            mask = (recent["team_a"] == csv_name) | (recent["team_b"] == csv_name)
            appearances = recent[mask]["year"].nunique()
            wins = 0
            for _, row in recent[mask].iterrows():
                winner = row["team_a"] if row["score_a"] > row["score_b"] else row["team_b"]
                if winner == csv_name:
                    wins += 1
            result[bracket_name] = appearances + wins * 0.5

        return result
