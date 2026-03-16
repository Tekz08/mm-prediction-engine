"""
Scrape real 2025-26 team stats from Sports Reference for all 68 tournament teams.
Pulls from 4 pages: basic school stats, basic opponent stats, advanced school stats,
advanced opponent stats. Merges into a single dataset matching our TeamStats model.
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

URLS = {
    "basic": "https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html",
    "basic_opp": "https://www.sports-reference.com/cbb/seasons/men/{year}-opponent-stats.html",
    "advanced": "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html",
    "advanced_opp": "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-opponent-stats.html",
}

TABLE_IDS = {
    "basic": "basic_school_stats",
    "basic_opp": "basic_opp_stats",
    "advanced": "adv_school_stats",
    "advanced_opp": "adv_opp_stats",
}

SR_NAME_TO_BRACKET = {
    "Connecticut": "UConn",
    "Michigan State": "Michigan State",
    "St. John's (NY)": "St. John's",
    "Ohio State": "Ohio State",
    "Texas Christian": "TCU",
    "Central Florida": "UCF",
    "South Florida": "South Florida",
    "Northern Iowa": "Northern Iowa",
    "California Baptist": "Cal Baptist",
    "North Dakota State": "North Dakota State",
    "Brigham Young": "BYU",
    "Miami (FL)": "Miami FL",
    "Utah State": "Utah State",
    "High Point": "High Point",
    "Hawaii": "Hawaii",
    "Kennesaw State": "Kennesaw State",
    "Long Island University": "Long Island",
    "Iowa State": "Iowa State",
    "Texas Tech": "Texas Tech",
    "Saint Louis": "Saint Louis",
    "Santa Clara": "Santa Clara",
    "Southern Methodist": "SMU",
    "Wright State": "Wright State",
    "Tennessee State": "Tennessee State",
    "Howard": "Howard",
    "North Carolina": "North Carolina",
    "Saint Mary's (CA)": "Saint Mary's",
    "Texas A&M": "Texas A&M",
    "Virginia Commonwealth": "VCU",
    "McNeese State": "McNeese",
    "Prairie View": "Prairie View A&M",
    "Pennsylvania": "Penn",
    "Queens (NC)": "Queens",
    "NC State": "NC State",
    "North Carolina State": "NC State",
    "Miami (OH)": "Miami OH",
    "Maryland-Baltimore County": "UMBC",
    "UMBC": "UMBC",
}


def scrape_table(url: str, table_id: str) -> dict[str, dict]:
    print(f"  Fetching {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": table_id})
    if not table:
        raise ValueError(f"Table {table_id} not found")

    rows = table.find("tbody").find_all("tr", class_=lambda x: x != "thead")
    data = {}
    for row in rows:
        cells = row.find_all(["td", "th"])
        entry = {}
        school_name = None
        for cell in cells:
            stat = cell.get("data-stat", "")
            if stat == "DUMMY" or not stat:
                continue
            val = cell.text.strip()
            if stat == "school_name":
                link = cell.find("a")
                school_name = link.text.strip() if link else val
            else:
                entry[stat] = val
        if school_name:
            data[school_name] = entry
    return data


def normalize_name(sr_name: str) -> str:
    if sr_name in SR_NAME_TO_BRACKET:
        return SR_NAME_TO_BRACKET[sr_name]
    return sr_name


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def build_team_stats(basic, basic_opp, advanced, advanced_opp, games):
    off_rtg = safe_float(advanced.get("off_rtg"))
    opp_off_rtg = safe_float(advanced_opp.get("opp_off_rtg"))
    pace = safe_float(advanced.get("pace"))
    sos = safe_float(basic.get("sos"))

    tov = safe_int(basic.get("tov"))
    fg = safe_int(basic.get("fg"))
    fga = safe_int(basic.get("fga"))
    fta = safe_int(basic.get("fta"))
    possessions_approx = fga - safe_int(basic.get("orb")) + tov + 0.475 * fta if fga > 0 else 1

    opp_tov = safe_int(basic_opp.get("opp_tov"))
    opp_fga = safe_int(basic_opp.get("opp_fga"))
    opp_fta = safe_int(basic_opp.get("opp_fta"))
    opp_orb = safe_int(basic_opp.get("opp_orb"))
    opp_poss = opp_fga - opp_orb + opp_tov + 0.475 * opp_fta if opp_fga > 0 else 1

    to_rate_off = (tov / possessions_approx * 100) if possessions_approx > 0 else 0
    to_rate_def = (opp_tov / opp_poss * 100) if opp_poss > 0 else 0

    orb = safe_int(basic.get("orb"))
    trb = safe_int(basic.get("trb"))
    opp_trb = safe_int(basic_opp.get("opp_trb"))
    opp_orb_val = safe_int(basic_opp.get("opp_orb"))

    total_reb_chances = trb + opp_trb if (trb + opp_trb) > 0 else 1
    orb_rate = safe_float(advanced.get("orb_pct"))
    drb_rate = 100 - safe_float(advanced_opp.get("opp_orb_pct")) if advanced_opp.get("opp_orb_pct") else 70.0

    fg3_pct = safe_float(basic.get("fg3_pct")) * 100 if safe_float(basic.get("fg3_pct")) < 1 else safe_float(basic.get("fg3_pct"))
    ft_pct = safe_float(basic.get("ft_pct")) * 100 if safe_float(basic.get("ft_pct")) < 1 else safe_float(basic.get("ft_pct"))
    opp_fg3_pct = safe_float(basic_opp.get("opp_fg3_pct")) * 100 if safe_float(basic_opp.get("opp_fg3_pct")) < 1 else safe_float(basic_opp.get("opp_fg3_pct"))

    fg3a = safe_int(basic.get("fg3a"))
    ft_rate = (fta / fga * 100) if fga > 0 else 0

    return {
        "adj_offensive_efficiency": round(off_rtg, 1),
        "adj_defensive_efficiency": round(opp_off_rtg, 1),
        "tempo": round(pace, 1),
        "strength_of_schedule": round(sos, 1),
        "record_vs_top25": "",
        "turnover_rate_off": round(to_rate_off, 1),
        "turnover_rate_def": round(to_rate_def, 1),
        "off_rebound_rate": round(orb_rate, 1),
        "def_rebound_rate": round(drb_rate, 1),
        "ft_rate": round(ft_rate, 1),
        "ft_pct": round(ft_pct, 1),
        "three_pt_pct": round(fg3_pct, 1),
        "three_pt_def_pct": round(opp_fg3_pct, 1),
        "wins": safe_int(basic.get("wins")),
        "losses": safe_int(basic.get("losses")),
        "last10_wins": 0,
        "last10_losses": 0,
        "tournament_experience": 0.0,
    }


def main():
    year = config.CURRENT_YEAR
    bracket_path = config.BRACKETS_DIR / f"{year}_bracket.json"
    with open(bracket_path) as f:
        bracket = json.load(f)

    tournament_teams = {}
    for region, entries in bracket.items():
        for entry in entries:
            tournament_teams[entry["team"]] = {
                "seed": entry["seed"],
                "region": region,
            }
            if entry.get("play_in"):
                tournament_teams[entry["play_in"]] = {
                    "seed": entry["seed"],
                    "region": region,
                }

    print(f"Tournament teams: {len(tournament_teams)}")
    print(f"Scraping Sports Reference for {year} data...\n")

    all_data = {}
    for key, url_template in URLS.items():
        url = url_template.format(year=year)
        all_data[key] = scrape_table(url, TABLE_IDS[key])
        time.sleep(3)
        print(f"  -> {len(all_data[key])} schools loaded\n")

    sr_names = set()
    for key in all_data:
        sr_names.update(all_data[key].keys())

    bracket_to_sr = {}
    for sr_name in sr_names:
        normalized = normalize_name(sr_name)
        if normalized in tournament_teams:
            bracket_to_sr[normalized] = sr_name

    print(f"\nMatched {len(bracket_to_sr)} of {len(tournament_teams)} tournament teams")
    missing = set(tournament_teams.keys()) - set(bracket_to_sr.keys())
    if missing:
        print(f"Missing: {missing}")
        for sr_name in sr_names:
            for m in list(missing):
                if m.lower() in sr_name.lower() or sr_name.lower() in m.lower():
                    print(f"  Possible match: '{m}' -> '{sr_name}'")

    teams_json = []
    for bracket_name, info in tournament_teams.items():
        sr_name = bracket_to_sr.get(bracket_name)
        if sr_name:
            basic = all_data["basic"].get(sr_name, {})
            basic_opp = all_data["basic_opp"].get(sr_name, {})
            advanced = all_data["advanced"].get(sr_name, {})
            advanced_opp = all_data["advanced_opp"].get(sr_name, {})
            games = safe_int(basic.get("g", 30))
            stats = build_team_stats(basic, basic_opp, advanced, advanced_opp, games)
        else:
            print(f"  WARNING: No data for {bracket_name}, using empty stats")
            stats = {}

        teams_json.append({
            "name": bracket_name,
            "seed": info["seed"],
            "region": info["region"],
            "conference": "",
            "stats": stats,
        })

    teams_json.sort(key=lambda t: (t["region"], t["seed"]))

    output_path = config.TEAMS_DIR / f"{year}_teams.json"
    with open(output_path, "w") as f:
        json.dump(teams_json, f, indent=2)
    print(f"\nWrote {len(teams_json)} teams to {output_path}")

    for t in teams_json[:5]:
        print(f"\n  {t['seed']}-{t['name']} ({t['region']})")
        s = t["stats"]
        if s:
            print(f"    Off Eff: {s.get('adj_offensive_efficiency')}, Def Eff: {s.get('adj_defensive_efficiency')}")
            print(f"    Tempo: {s.get('tempo')}, SOS: {s.get('strength_of_schedule')}")
            print(f"    3PT%: {s.get('three_pt_pct')}, FT%: {s.get('ft_pct')}")
            print(f"    Record: {s.get('wins')}-{s.get('losses')}")


if __name__ == "__main__":
    main()
