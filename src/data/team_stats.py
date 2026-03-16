from __future__ import annotations

import json
from pathlib import Path

import config
from src.data.models import Team, TeamStats, PlayerStats, ScheduleGame, CoachInfo


def load_teams(year: int | None = None, path: Path | None = None) -> list[Team]:
    year = year or config.CURRENT_YEAR
    path = path or config.TEAMS_DIR / f"{year}_teams.json"
    with open(path) as f:
        data = json.load(f)
    teams = []
    for entry in data:
        stats = TeamStats(**entry.get("stats", {}))
        coach = CoachInfo(**entry.get("coach", {}))
        players = [PlayerStats(**p) for p in entry.get("players", [])]
        schedule = [ScheduleGame(**g) for g in entry.get("schedule", [])]
        teams.append(
            Team(
                name=entry["name"],
                seed=entry["seed"],
                region=entry["region"],
                conference=entry.get("conference", ""),
                stats=stats,
                coach=coach,
                players=players,
                schedule=schedule,
            )
        )
    return teams


def load_teams_by_name(year: int | None = None, path: Path | None = None) -> dict[str, Team]:
    return {t.name: t for t in load_teams(year, path)}


def save_teams(teams: list[Team], year: int | None = None, path: Path | None = None):
    year = year or config.CURRENT_YEAR
    path = path or config.TEAMS_DIR / f"{year}_teams.json"
    data = [t.model_dump() for t in teams]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
