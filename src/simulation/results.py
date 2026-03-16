from __future__ import annotations

import config


class SimulationResults:
    def __init__(self, raw: dict):
        self.iterations = raw["iterations"]
        self.team_round_counts = raw["team_round_counts"]
        self.championship_wins = raw["championship_wins"]
        self.final_four_appearances = raw.get("final_four_appearances", {})
        self.final_four_combos = raw.get("final_four_combos", {})
        self.championship_matchups = raw.get("championship_matchups", {})

    def team_advancement_probabilities(self, team: str) -> dict[str, float]:
        counts = self.team_round_counts.get(team, {})
        return {
            round_name: counts.get(round_name, 0) / self.iterations * 100
            for round_name in config.ROUNDS
        }

    def all_advancement_probabilities(self) -> dict[str, dict[str, float]]:
        return {
            team: self.team_advancement_probabilities(team)
            for team in self.team_round_counts
        }

    def championship_probabilities(self) -> dict[str, float]:
        return {
            team: wins / self.iterations * 100
            for team, wins in sorted(self.championship_wins.items(), key=lambda x: -x[1])
        }

    def final_four_probabilities(self) -> dict[str, float]:
        return {
            team: count / self.iterations * 100
            for team, count in sorted(self.final_four_appearances.items(), key=lambda x: -x[1])
        }

    def most_likely_champion(self) -> tuple[str, float]:
        if not self.championship_wins:
            return ("Unknown", 0.0)
        best = max(self.championship_wins.items(), key=lambda x: x[1])
        return (best[0], best[1] / self.iterations * 100)

    def confidence_tiers(self) -> dict[str, list[dict]]:
        tiers = {"locks": [], "likely": [], "coin_flips": [], "upsets_to_watch": []}
        probs = self.all_advancement_probabilities()

        for team, rounds in probs.items():
            r64 = rounds.get("Round of 64", 0)
            if r64 >= 90:
                tiers["locks"].append({"team": team, "r64_prob": r64})
            elif r64 >= 65:
                tiers["likely"].append({"team": team, "r64_prob": r64})
            elif r64 >= 40:
                tiers["coin_flips"].append({"team": team, "r64_prob": r64})
            elif r64 > 0:
                tiers["upsets_to_watch"].append({"team": team, "r64_prob": r64})

        for key in tiers:
            tiers[key].sort(key=lambda x: -x["r64_prob"])

        return tiers

    def predicted_bracket(self) -> dict[str, dict[str, list[str]]]:
        bracket = {}
        probs = self.all_advancement_probabilities()

        for team, rounds in probs.items():
            for round_name, pct in rounds.items():
                if pct > 50:
                    bracket.setdefault(round_name, []).append((team, pct))

        for round_name in bracket:
            bracket[round_name].sort(key=lambda x: -x[1])
            bracket[round_name] = [
                {"team": t, "probability": round(p, 1)} for t, p in bracket[round_name]
            ]

        return bracket

    def to_summary_dict(self) -> dict:
        champion, champ_pct = self.most_likely_champion()
        return {
            "iterations": self.iterations,
            "predicted_champion": {"team": champion, "probability": round(champ_pct, 1)},
            "final_four_probabilities": {
                t: round(p, 1)
                for t, p in list(self.final_four_probabilities().items())[:8]
            },
            "championship_probabilities": {
                t: round(p, 1)
                for t, p in list(self.championship_probabilities().items())[:10]
            },
            "tiers": self.confidence_tiers(),
        }
