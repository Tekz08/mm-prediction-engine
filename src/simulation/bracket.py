from __future__ import annotations

import json
from pathlib import Path

import config
from src.data.models import BracketEntry, Team


REGION_MATCHUP_ORDER = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def load_bracket(year: int | None = None, path: Path | None = None) -> dict[str, list[BracketEntry]]:
    year = year or config.CURRENT_YEAR
    path = path or config.BRACKETS_DIR / f"{year}_bracket.json"
    with open(path) as f:
        data = json.load(f)
    bracket = {}
    for region, entries in data.items():
        bracket[region] = [
            BracketEntry(
                seed=e["seed"],
                team=e["team"],
                region=region,
                play_in=e.get("play_in"),
            )
            for e in entries
        ]
    return bracket


def get_first_four_matchups(bracket: dict[str, list[BracketEntry]]) -> list[tuple[BracketEntry, BracketEntry]]:
    matchups = []
    for region, entries in bracket.items():
        for entry in entries:
            if entry.play_in:
                opponent = BracketEntry(
                    seed=entry.seed,
                    team=entry.play_in,
                    region=region,
                )
                matchups.append((entry, opponent))
    return matchups


def build_region_matchups(entries: list[BracketEntry]) -> list[tuple[BracketEntry, BracketEntry]]:
    by_seed = {e.seed: e for e in entries}
    matchups = []
    for high, low in REGION_MATCHUP_ORDER:
        if high in by_seed and low in by_seed:
            matchups.append((by_seed[high], by_seed[low]))
    return matchups


def entry_to_team(entry: BracketEntry, teams_by_name: dict[str, Team]) -> Team:
    if entry.team in teams_by_name:
        return teams_by_name[entry.team]
    return Team(name=entry.team, seed=entry.seed, region=entry.region)


class BracketSimulator:
    def __init__(self, bracket: dict[str, list[BracketEntry]], teams_by_name: dict[str, Team]):
        self.bracket = bracket
        self.teams_by_name = teams_by_name
        self.regions = list(bracket.keys())

    def get_first_four(self) -> list[tuple[Team, Team]]:
        ff = get_first_four_matchups(self.bracket)
        return [
            (entry_to_team(a, self.teams_by_name), entry_to_team(b, self.teams_by_name))
            for a, b in ff
        ]

    def resolve_first_four(self, winners: dict[str, str]):
        for region, entries in self.bracket.items():
            for i, entry in enumerate(entries):
                if entry.play_in:
                    winning_team = winners.get(entry.team) or winners.get(entry.play_in)
                    if winning_team:
                        self.bracket[region][i] = BracketEntry(
                            seed=entry.seed,
                            team=winning_team,
                            region=region,
                        )

    def get_round_of_64_matchups(self) -> dict[str, list[tuple[Team, Team]]]:
        result = {}
        for region, entries in self.bracket.items():
            pairs = build_region_matchups(entries)
            result[region] = [
                (entry_to_team(a, self.teams_by_name), entry_to_team(b, self.teams_by_name))
                for a, b in pairs
            ]
        return result

    def get_next_round_matchups(self, winners: list[Team]) -> list[tuple[Team, Team]]:
        matchups = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                matchups.append((winners[i], winners[i + 1]))
        return matchups

    def get_final_four_matchups(self, region_winners: dict[str, Team]) -> list[tuple[Team, Team]]:
        regions = self.regions
        if len(regions) < 4:
            return []
        semi1 = (region_winners.get(regions[0]), region_winners.get(regions[1]))
        semi2 = (region_winners.get(regions[2]), region_winners.get(regions[3]))
        matchups = []
        if semi1[0] and semi1[1]:
            matchups.append(semi1)
        if semi2[0] and semi2[1]:
            matchups.append(semi2)
        return matchups
