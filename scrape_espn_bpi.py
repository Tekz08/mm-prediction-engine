import json
from pathlib import Path

import requests

import config

API_URL = "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/powerindex"

BPI_CATEGORY = "bpi"
RESUME_CATEGORY = "resume"
TOURNAMENT_CATEGORY = "tournament"

BPI_FIELDS = ["bpi", "bpirank", "bpisevendaychangerank", "bpioffense", "bpidefense",
               "chancewinconfortie", "sosremrank", "wins", "losses", "projtotalwins",
               "projtotallosses", "confwins", "conflosses", "projconfwins", "projconflosses"]

RESUME_FIELDS = ["sorrank", "projectedtournamentseed", "projectedtournamentorder",
                  "top50bpiwins", "top50bpilosses", "sospastrank", "sosoutofconfpastrank"]

ESPN_NAME_TO_BRACKET = {
    "UConn": "UConn",
    "St. John's": "St. John's",
    "St John's": "St. John's",
    "Miami": "Miami FL",
    "Miami (OH)": "Miami OH",
    "BYU": "BYU",
    "TCU": "TCU",
    "UCF": "UCF",
    "SMU": "SMU",
    "VCU": "VCU",
    "NC State": "NC State",
    "Saint Mary's": "Saint Mary's",
    "Texas A&M": "Texas A&M",
    "Utah State": "Utah State",
    "Michigan St": "Michigan State",
    "Iowa State": "Iowa State",
    "Texas Tech": "Texas Tech",
    "Ohio State": "Ohio State",
    "Cal Baptist": "Cal Baptist",
    "CA Baptist": "Cal Baptist",
    "North Dakota St": "North Dakota State",
    "N Dakota St": "North Dakota State",
    "South Florida": "South Florida",
    "Northern Iowa": "Northern Iowa",
    "High Point": "High Point",
    "Kennesaw St": "Kennesaw State",
    "Long Island": "Long Island",
    "Wright St": "Wright State",
    "Tennessee St": "Tennessee State",
    "Prairie View": "Prairie View A&M",
    "North Carolina": "North Carolina",
    "McNeese": "McNeese",
    "Saint Louis": "Saint Louis",
    "Santa Clara": "Santa Clara",
    "Hawai'i": "Hawaii",
}


def fetch_all_teams(year: int) -> list[dict]:
    all_teams = []
    page = 1
    while True:
        params = {
            "region": "us",
            "lang": "en",
            "season": year,
            "limit": 50,
            "page": page,
        }
        print(f"  Fetching BPI page {page}...")
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        teams = data.get("teams", [])
        if not teams:
            break
        all_teams.extend(teams)

        pagination = data.get("pagination", {})
        if page >= pagination.get("pages", 1):
            break
        page += 1

    print(f"  -> {len(all_teams)} total teams fetched from ESPN BPI")
    return all_teams


def extract_category_values(categories: list[dict], category_name: str, field_names: list[str]) -> dict:
    for cat in categories:
        if cat.get("name") == category_name:
            values = cat.get("values", [])
            return dict(zip(field_names, values))
    return {}


def normalize_espn_name(espn_short_name: str) -> str:
    return ESPN_NAME_TO_BRACKET.get(espn_short_name, espn_short_name)


def build_bpi_stats(bpi_data: dict, resume_data: dict) -> dict:
    return {
        "bpi": round(bpi_data.get("bpi", 0.0), 2),
        "bpi_offensive": round(bpi_data.get("bpioffense", 0.0), 2),
        "bpi_defensive": round(bpi_data.get("bpidefense", 0.0), 2),
        "bpi_rank": int(bpi_data.get("bpirank", 0)),
        "sor_rank": int(resume_data.get("sorrank", 0)),
        "quality_wins": int(resume_data.get("top50bpiwins", 0)),
        "quality_losses": int(resume_data.get("top50bpilosses", 0)),
        "espn_sos_rank": int(resume_data.get("sospastrank", 0)),
    }


def main():
    year = config.CURRENT_YEAR

    bracket_path = config.BRACKETS_DIR / f"{year}_bracket.json"
    with open(bracket_path) as f:
        bracket = json.load(f)

    tournament_teams = set()
    for region, entries in bracket.items():
        for entry in entries:
            tournament_teams.add(entry["team"])
            if entry.get("play_in"):
                tournament_teams.add(entry["play_in"])

    print(f"Tournament teams: {len(tournament_teams)}")
    print(f"Fetching ESPN BPI data for {year}...\n")

    espn_teams = fetch_all_teams(year)

    espn_by_bracket_name: dict[str, dict] = {}
    for espn_team in espn_teams:
        team_info = espn_team.get("team", {})
        short_name = team_info.get("shortDisplayName", "")
        bracket_name = normalize_espn_name(short_name)

        if bracket_name in tournament_teams:
            categories = espn_team.get("categories", [])
            bpi_data = extract_category_values(categories, BPI_CATEGORY, BPI_FIELDS)
            resume_data = extract_category_values(categories, RESUME_CATEGORY, RESUME_FIELDS)
            espn_by_bracket_name[bracket_name] = build_bpi_stats(bpi_data, resume_data)

    print(f"\nMatched {len(espn_by_bracket_name)} of {len(tournament_teams)} tournament teams")
    missing = tournament_teams - set(espn_by_bracket_name.keys())
    if missing:
        print(f"Missing: {missing}")
        all_espn_names = [t.get("team", {}).get("shortDisplayName", "") for t in espn_teams]
        for m in sorted(missing):
            candidates = [n for n in all_espn_names if m.lower() in n.lower() or n.lower() in m.lower()]
            if candidates:
                print(f"  Possible match for '{m}': {candidates}")

    teams_path = config.TEAMS_DIR / f"{year}_teams.json"
    with open(teams_path) as f:
        teams_json = json.load(f)

    updated = 0
    for team in teams_json:
        name = team["name"]
        if name in espn_by_bracket_name:
            team["stats"].update(espn_by_bracket_name[name])
            updated += 1

    with open(teams_path, "w") as f:
        json.dump(teams_json, f, indent=2)
    print(f"\nUpdated {updated} teams in {teams_path}")

    for team in teams_json[:5]:
        s = team["stats"]
        if s.get("bpi"):
            print(f"\n  {team['seed']}-{team['name']} ({team['region']})")
            print(f"    BPI: {s['bpi']} (off: {s['bpi_offensive']}, def: {s['bpi_defensive']}) rank: {s['bpi_rank']}")
            print(f"    SOR rank: {s['sor_rank']}, Quality: {s['quality_wins']}-{s['quality_losses']}")
            print(f"    ESPN SOS rank: {s['espn_sos_rank']}")


if __name__ == "__main__":
    main()
