from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
TEAMS_DIR = DATA_DIR / "teams"
BRACKETS_DIR = DATA_DIR / "brackets"

CURRENT_YEAR = 2026
DEFAULT_ITERATIONS = 10_000

WEIGHT_HISTORICAL = 0.15
WEIGHT_EFFICIENCY = 0.45
WEIGHT_PROFILE = 0.20
WEIGHT_MATCHUP = 0.20

LOGISTIC_SCALE = 10.0

GAME_STDEV = 9.0
LEAGUE_AVG_EFFICIENCY = 100.0
SOS_ADJUSTMENT_FACTOR = 0.5

ROUND_VARIANCE_MULT = {
    "First Four": 1.0,
    "Round of 64": 1.0,
    "Round of 32": 0.95,
    "Sweet 16": 0.90,
    "Elite 8": 0.85,
    "Final Four": 0.80,
    "Championship": 0.75,
}

ROUND_EXPERIENCE_PREMIUM = {
    "First Four": 0.0,
    "Round of 64": 0.0,
    "Round of 32": 0.2,
    "Sweet 16": 0.5,
    "Elite 8": 0.8,
    "Final Four": 1.0,
    "Championship": 1.2,
}

ROUND_DEFENSE_PREMIUM = {
    "First Four": 0.0,
    "Round of 64": 0.0,
    "Round of 32": 0.02,
    "Sweet 16": 0.04,
    "Elite 8": 0.06,
    "Final Four": 0.08,
    "Championship": 0.10,
}

STAR_DEPENDENCY_STDEV_MULT = 0.08
BENCH_DEPTH_ROUND_BONUS = {
    "First Four": 0.0,
    "Round of 64": 0.0,
    "Round of 32": 0.0,
    "Sweet 16": 0.01,
    "Elite 8": 0.02,
    "Final Four": 0.03,
    "Championship": 0.04,
}

ROUNDS = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
]
