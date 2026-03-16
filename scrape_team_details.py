"""
Scrape individual team pages from Sports Reference to get:
- Player roster with per-game stats
- Full season schedule with results
- Computed: last-10 record, record vs ranked opponents

Merges into the existing 2026_teams.json produced by scrape_real_data.py.
Rate-limited to respect Sports Reference (3s between requests).
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.sports-reference.com"
DELAY = 3.5

SR_NAME_TO_BRACKET = {
    "Connecticut": "UConn",
    "St. John's (NY)": "St. John's",
    "Texas Christian": "TCU",
    "TCU": "TCU",
    "Central Florida": "UCF",
    "UCF": "UCF",
    "South Florida": "South Florida",
    "Northern Iowa": "Northern Iowa",
    "California Baptist": "Cal Baptist",
    "North Dakota State": "North Dakota State",
    "Brigham Young": "BYU",
    "Miami (FL)": "Miami FL",
    "Utah State": "Utah State",
    "High Point": "High Point",
    "Kennesaw State": "Kennesaw State",
    "Long Island University": "Long Island",
    "Iowa State": "Iowa State",
    "Texas Tech": "Texas Tech",
    "Saint Louis": "Saint Louis",
    "Santa Clara": "Santa Clara",
    "Southern Methodist": "SMU",
    "Wright State": "Wright State",
    "Tennessee State": "Tennessee State",
    "Saint Mary's (CA)": "Saint Mary's",
    "Texas A&M": "Texas A&M",
    "Virginia Commonwealth": "VCU",
    "McNeese State": "McNeese",
    "Prairie View": "Prairie View A&M",
    "Pennsylvania": "Penn",
    "Queens (NC)": "Queens",
    "Michigan State": "Michigan State",
    "Ohio State": "Ohio State",
    "North Carolina": "North Carolina",
    "North Dakota State": "North Dakota State",
    "Howard": "Howard",
}

BRACKET_TO_SLUG = {
    "Duke": "duke", "UConn": "connecticut", "Michigan State": "michigan-state",
    "Kansas": "kansas", "St. John's": "st-johns-ny", "Louisville": "louisville",
    "UCLA": "ucla", "Ohio State": "ohio-state", "TCU": "texas-christian",
    "UCF": "central-florida", "South Florida": "south-florida",
    "Northern Iowa": "northern-iowa", "Cal Baptist": "california-baptist",
    "North Dakota State": "north-dakota-state", "Furman": "furman", "Siena": "siena",
    "Arizona": "arizona", "Purdue": "purdue", "Gonzaga": "gonzaga",
    "Arkansas": "arkansas", "Wisconsin": "wisconsin", "BYU": "brigham-young",
    "Miami FL": "miami-fl", "Villanova": "villanova", "Utah State": "utah-state",
    "Missouri": "missouri", "Texas": "texas", "High Point": "high-point",
    "Hawaii": "hawaii", "Kennesaw State": "kennesaw-state", "Queens": "queens-nc",
    "Long Island": "long-island-university",
    "Michigan": "michigan", "Iowa State": "iowa-state", "Virginia": "virginia",
    "Alabama": "alabama", "Texas Tech": "texas-tech", "Tennessee": "tennessee",
    "Kentucky": "kentucky", "Georgia": "georgia", "Saint Louis": "saint-louis",
    "Santa Clara": "santa-clara", "SMU": "southern-methodist", "Akron": "akron",
    "Hofstra": "hofstra", "Wright State": "wright-state",
    "Tennessee State": "tennessee-state", "Howard": "howard",
    "Florida": "florida", "Houston": "houston", "Illinois": "illinois",
    "Nebraska": "nebraska", "Vanderbilt": "vanderbilt",
    "North Carolina": "north-carolina", "Saint Mary's": "saint-marys-ca",
    "Clemson": "clemson", "Iowa": "iowa", "Texas A&M": "texas-am",
    "VCU": "virginia-commonwealth", "McNeese": "mcneese-state",
    "Troy": "troy", "Penn": "pennsylvania", "Idaho": "idaho",
    "Prairie View A&M": "prairie-view",
    "NC State": "north-carolina-state",
    "Miami OH": "miami-oh",
    "UMBC": "maryland-baltimore-county",
    "Lehigh": "lehigh",
}


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if v == v else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def scrape_team_page(slug: str, year: int) -> dict:
    url = f"{BASE_URL}/cbb/schools/{slug}/men/{year}.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    conference = ""
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if f"/cbb/conferences/" in href and f"/{year}" in href:
            conference = link.text.strip()
            break

    players = []
    per_game = soup.find("table", {"id": "players_per_game"})
    if per_game:
        tbody = per_game.find("tbody")
        for row in tbody.find_all("tr"):
            tds = row.find_all("td")
            data = {td.get("data-stat", ""): td.text.strip() for td in tds}
            name = data.get("name_display", "")
            if not name:
                continue
            players.append({
                "name": name,
                "position": data.get("pos", ""),
                "games": safe_int(data.get("games")),
                "games_started": safe_int(data.get("games_started")),
                "mpg": safe_float(data.get("mp_per_g")),
                "ppg": safe_float(data.get("pts_per_g")),
                "rpg": safe_float(data.get("trb_per_g")),
                "apg": safe_float(data.get("ast_per_g")),
                "spg": safe_float(data.get("stl_per_g")),
                "bpg": safe_float(data.get("blk_per_g")),
                "topg": safe_float(data.get("tov_per_g")),
                "fg_pct": safe_float(data.get("fg_pct")) * 100 if safe_float(data.get("fg_pct")) <= 1 else safe_float(data.get("fg_pct")),
                "three_pct": safe_float(data.get("fg3_pct")) * 100 if safe_float(data.get("fg3_pct")) <= 1 else safe_float(data.get("fg3_pct")),
                "ft_pct": safe_float(data.get("ft_pct")) * 100 if safe_float(data.get("ft_pct")) <= 1 else safe_float(data.get("ft_pct")),
                "efg_pct": safe_float(data.get("efg_pct")) * 100 if safe_float(data.get("efg_pct")) <= 1 else safe_float(data.get("efg_pct")),
                "orpg": safe_float(data.get("orb_per_g")),
                "drpg": safe_float(data.get("drb_per_g")),
            })

    roster = soup.find("table", {"id": "roster"})
    if roster:
        roster_map = {}
        for row in roster.find("tbody").find_all("tr"):
            tds = row.find_all("td")
            rdata = {td.get("data-stat", ""): td.text.strip() for td in tds}
            pname = rdata.get("player", "")
            if pname:
                roster_map[pname] = {
                    "year": rdata.get("class", ""),
                    "height": rdata.get("height", ""),
                }
        for p in players:
            if p["name"] in roster_map:
                p["year"] = roster_map[p["name"]]["year"]
                p["height"] = roster_map[p["name"]]["height"]

    coach_slug = None
    for a in soup.select('a[href*="/cbb/coaches/"]'):
        href = a.get("href", "")
        if href != "/cbb/coaches/" and "/cbb/coaches/" in href:
            coach_slug = href.rstrip("/").split("/")[-1]
            break

    return {"players": players, "conference": conference, "coach_slug": coach_slug}


def scrape_coach(coach_slug: str) -> dict:
    if coach_slug.endswith(".html"):
        coach_slug = coach_slug[:-5]
    url = f"{BASE_URL}/cbb/coaches/{coach_slug}.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    info = {
        "name": "",
        "career_wins": 0,
        "career_losses": 0,
        "career_win_pct": 0.0,
        "years_coaching": 0,
        "tourney_appearances": 0,
        "tourney_wins": 0,
        "tourney_losses": 0,
        "final_fours": 0,
        "championships": 0,
    }

    coach_name_el = soup.select_one("h1 span")
    if coach_name_el:
        info["name"] = coach_name_el.get_text(strip=True)

    for p in soup.select("#info p, #meta p"):
        text = p.get_text(" ", strip=True)
        m = re.search(r"(\d+)\s+Years?,\s*([\d,]+)-([\d,]+),\s*\.(\d+)", text)
        if m and "Career Record" in text:
            info["years_coaching"] = int(m.group(1))
            info["career_wins"] = int(m.group(2).replace(",", ""))
            info["career_losses"] = int(m.group(3).replace(",", ""))
            info["career_win_pct"] = float(f"0.{m.group(4)}")

        if "NCAA Tournament" in text:
            tm = re.search(r"(\d+)\s+Years?\s*\((\d+)-(\d+)\)", text)
            if tm:
                info["tourney_appearances"] = int(tm.group(1))
                info["tourney_wins"] = int(tm.group(2))
                info["tourney_losses"] = int(tm.group(3))
            ff = re.search(r"(\d+)\s+Final Four", text)
            if ff:
                info["final_fours"] = int(ff.group(1))
            ch = re.search(r"(\d+)\s+Championship", text)
            if ch:
                info["championships"] = int(ch.group(1))

    return info


def scrape_schedule(slug: str, year: int) -> list[dict]:
    url = f"{BASE_URL}/cbb/schools/{slug}/men/{year}-schedule.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    schedule = []
    table = soup.find("table", {"id": "schedule"})
    if not table:
        return schedule

    tbody = table.find("tbody")
    for row in tbody.find_all("tr"):
        cells = row.find_all(["td", "th"])
        data = {c.get("data-stat", ""): c.text.strip() for c in cells}

        date = data.get("date_game", "")
        if not date or date == "Date":
            continue

        result_raw = data.get("game_result", "")
        if not result_raw:
            continue

        result = "W" if result_raw.startswith("W") else "L" if result_raw.startswith("L") else ""
        pts = safe_int(data.get("pts"))
        opp_pts = safe_int(data.get("opp_pts"))
        location = data.get("game_location", "")
        opp_name = data.get("opp_name", data.get("opp_id", ""))

        is_ranked = bool(re.search(r"\(\d+\)", opp_name))
        opp_clean = re.sub(r"\s*\(\d+\)\s*", "", opp_name).strip()
        opp_clean = re.sub(r"[^a-zA-Z0-9\s.&'\-]", "", opp_clean).strip()

        schedule.append({
            "date": date,
            "location": "@" if location == "@" else "N" if location == "N" else "H",
            "opponent": opp_clean,
            "result": result,
            "team_score": pts,
            "opp_score": opp_pts,
            "is_ranked_opp": is_ranked,
        })

    return schedule


def compute_last10_and_ranked(schedule: list[dict]) -> dict:
    valid_games = [g for g in schedule if g.get("result") in ("W", "L")]

    last10 = valid_games[-10:] if len(valid_games) >= 10 else valid_games
    l10_wins = sum(1 for g in last10 if g["result"] == "W")
    l10_losses = len(last10) - l10_wins

    ranked_games = [g for g in valid_games if g.get("is_ranked_opp")]
    ranked_wins = sum(1 for g in ranked_games if g["result"] == "W")
    ranked_losses = len(ranked_games) - ranked_wins

    return {
        "last10_wins": l10_wins,
        "last10_losses": l10_losses,
        "record_vs_top25": f"{ranked_wins}-{ranked_losses}",
    }


def main():
    import sys
    only_teams = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only_teams = set(sys.argv[idx + 1:])

    year = config.CURRENT_YEAR
    teams_path = config.TEAMS_DIR / f"{year}_teams.json"

    with open(teams_path) as f:
        teams = json.load(f)

    teams_by_name = {t["name"]: t for t in teams}
    total = len(teams)

    print(f"Scraping individual team data for {total} teams...")
    if only_teams:
        print(f"  Filtering to: {only_teams}")
    print(f"This will make ~{total * 3} requests with {DELAY}s delay = ~{int(total * 3 * DELAY / 60)} minutes\n")

    for i, team in enumerate(teams):
        name = team["name"]
        if only_teams and name not in only_teams:
            continue
        slug = BRACKET_TO_SLUG.get(name)
        if not slug:
            print(f"  [{i+1}/{total}] {name}: NO SLUG MAPPING, skipping")
            continue

        print(f"  [{i+1}/{total}] {name} ({slug})...", end=" ", flush=True)

        try:
            team_data = scrape_team_page(slug, year)
            time.sleep(DELAY)

            schedule = scrape_schedule(slug, year)
            time.sleep(DELAY)

            coach_info = {}
            if team_data.get("coach_slug"):
                coach_info = scrape_coach(team_data["coach_slug"])
                time.sleep(DELAY)

            team["players"] = team_data["players"]
            if team_data["conference"]:
                team["conference"] = team_data["conference"]
            if coach_info:
                team["coach"] = coach_info

            team["schedule"] = schedule

            derived = compute_last10_and_ranked(schedule)
            if "stats" not in team:
                team["stats"] = {}
            team["stats"]["last10_wins"] = derived["last10_wins"]
            team["stats"]["last10_losses"] = derived["last10_losses"]
            team["stats"]["record_vs_top25"] = derived["record_vs_top25"]

            top_player = max(team_data["players"], key=lambda p: p["ppg"]) if team_data["players"] else None
            top_str = f"{top_player['name']} {top_player['ppg']}ppg" if top_player else "no players"
            coach_str = f"{coach_info.get('name', '?')} ({coach_info.get('career_wins', 0)}-{coach_info.get('career_losses', 0)}, {coach_info.get('tourney_appearances', 0)} tourney)" if coach_info else "no coach"
            print(f"{len(team_data['players'])} players, {len(schedule)} games, "
                  f"L10: {derived['last10_wins']}-{derived['last10_losses']}, "
                  f"vs Ranked: {derived['record_vs_top25']}, "
                  f"top: {top_str}, "
                  f"coach: {coach_str}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    with open(teams_path, "w") as f:
        json.dump(teams, f, indent=2)

    print(f"\nDone! Updated {teams_path}")

    print("\n--- Sample: Duke ---")
    duke = teams_by_name.get("Duke", {})
    if duke.get("players"):
        for p in duke["players"][:5]:
            print(f"  {p['name']} ({p.get('year','')}, {p['position']}): "
                  f"{p['ppg']}ppg {p['rpg']}rpg {p['apg']}apg "
                  f"{p['fg_pct']:.1f}FG% {p['three_pct']:.1f}3P%")
    if duke.get("schedule"):
        for g in duke["schedule"][-5:]:
            print(f"  {g['date']} {g['location']} {g['opponent']}: {g['result']} {g['team_score']}-{g['opp_score']}")
    print(f"  Last 10: {duke.get('stats',{}).get('last10_wins')}-{duke.get('stats',{}).get('last10_losses')}")
    print(f"  vs Ranked: {duke.get('stats',{}).get('record_vs_top25')}")


if __name__ == "__main__":
    main()
