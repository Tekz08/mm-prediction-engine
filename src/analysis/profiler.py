from __future__ import annotations

from src.analysis.historical import HistoricalAnalyzer
from src.data.models import Team, TeamProfile


def _normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 50.0
    return max(0.0, min(100.0, (value - low) / (high - low) * 100.0))


def compute_roster_metrics(team: Team) -> dict:
    players_with_minutes = [p for p in team.players if p.mpg >= 5]
    if not players_with_minutes:
        return {"top_scorer_share": 0.0, "top_two_share": 0.0, "deep_rotation_count": 0}

    total_ppg = sum(p.ppg for p in players_with_minutes)
    if total_ppg == 0:
        return {"top_scorer_share": 0.0, "top_two_share": 0.0, "deep_rotation_count": 0}

    sorted_by_ppg = sorted(players_with_minutes, key=lambda p: -p.ppg)
    top_scorer_share = sorted_by_ppg[0].ppg / total_ppg
    top_two_share = sum(p.ppg for p in sorted_by_ppg[:2]) / total_ppg if len(sorted_by_ppg) >= 2 else top_scorer_share
    deep_rotation_count = sum(1 for p in team.players if p.mpg >= 10)

    return {
        "top_scorer_share": round(top_scorer_share, 3),
        "top_two_share": round(top_two_share, 3),
        "deep_rotation_count": deep_rotation_count,
    }


def build_profile(team: Team, all_teams: list[Team]) -> TeamProfile:
    s = team.stats

    off_effs = [t.stats.adj_offensive_efficiency for t in all_teams if t.stats.adj_offensive_efficiency > 0]
    def_effs = [t.stats.adj_defensive_efficiency for t in all_teams if t.stats.adj_defensive_efficiency > 0]

    off_score = _normalize(s.adj_offensive_efficiency, min(off_effs, default=90), max(off_effs, default=130)) if off_effs else 50.0
    def_score = _normalize(
        max(def_effs, default=110) - s.adj_defensive_efficiency,
        0,
        max(def_effs, default=110) - min(def_effs, default=85),
    ) if def_effs else 50.0

    intangible = _compute_intangible(team, all_teams)
    composite = off_score * 0.4 + def_score * 0.4 + intangible * 0.2

    strengths = _identify_strengths(team, all_teams)
    weaknesses = _identify_weaknesses(team, all_teams)

    return TeamProfile(
        team=team,
        offensive_score=round(off_score, 1),
        defensive_score=round(def_score, 1),
        intangible_score=round(intangible, 1),
        composite_score=round(composite, 1),
        strengths=strengths,
        weaknesses=weaknesses,
    )


def _compute_intangible(team: Team, all_teams: list[Team]) -> float:
    s = team.stats
    c = team.coach
    factors = []

    sos_values = [t.stats.strength_of_schedule for t in all_teams if t.stats.strength_of_schedule != 0]
    if sos_values:
        factors.append(_normalize(s.strength_of_schedule, min(sos_values), max(sos_values)))

    exp_values = [t.stats.tournament_experience for t in all_teams]
    if any(v > 0 for v in exp_values):
        factors.append(_normalize(s.tournament_experience, 0, max(exp_values)))

    if s.last10_wins + s.last10_losses > 0:
        form = s.last10_wins / (s.last10_wins + s.last10_losses) * 100
        factors.append(form)

    if s.wins + s.losses > 0:
        win_pct = s.wins / (s.wins + s.losses) * 100
        factors.append(win_pct)

    if c.name:
        coach_tourney_apps = [t.coach.tourney_appearances for t in all_teams if t.coach.name]
        if coach_tourney_apps and max(coach_tourney_apps) > 0:
            factors.append(_normalize(c.tourney_appearances, 0, max(coach_tourney_apps)))

        coach_win_pcts = [t.coach.career_win_pct for t in all_teams if t.coach.career_win_pct > 0]
        if coach_win_pcts:
            factors.append(_normalize(c.career_win_pct, min(coach_win_pcts), max(coach_win_pcts)))

    sor_ranks = [t.stats.sor_rank for t in all_teams if t.stats.sor_rank > 0]
    if s.sor_rank > 0 and sor_ranks:
        factors.append(_normalize(max(sor_ranks) - s.sor_rank, 0, max(sor_ranks) - min(sor_ranks)))

    if s.quality_wins + s.quality_losses > 0:
        factors.append(s.quality_win_ratio * 100)

    roster = compute_roster_metrics(team)
    if roster["top_scorer_share"] > 0:
        balance_score = (1.0 - roster["top_scorer_share"]) * 100
        factors.append(balance_score)

    if roster["deep_rotation_count"] > 0:
        all_depths = [compute_roster_metrics(t)["deep_rotation_count"] for t in all_teams if t.players]
        if all_depths and max(all_depths) > 0:
            factors.append(_normalize(roster["deep_rotation_count"], min(all_depths), max(all_depths)))

    return sum(factors) / len(factors) if factors else 50.0


def _percentile_rank(value: float, values: list[float], higher_is_better: bool = True) -> float:
    if not values:
        return 50.0
    sorted_vals = sorted(values)
    rank = sum(1 for v in sorted_vals if v <= value) / len(sorted_vals) * 100
    return rank if higher_is_better else 100 - rank


def _identify_strengths(team: Team, all_teams: list[Team]) -> list[str]:
    s = team.stats
    strengths = []

    off_rank = _percentile_rank(s.adj_offensive_efficiency, [t.stats.adj_offensive_efficiency for t in all_teams])
    if off_rank >= 80:
        strengths.append("Elite offense")

    def_rank = _percentile_rank(s.adj_defensive_efficiency, [t.stats.adj_defensive_efficiency for t in all_teams], higher_is_better=False)
    if def_rank >= 80:
        strengths.append("Elite defense")

    if s.three_pt_pct >= 37:
        strengths.append("Strong 3-point shooting")

    if s.off_rebound_rate >= 32:
        strengths.append("Dominant offensive rebounding")

    if s.ft_pct >= 77:
        strengths.append("Reliable free throw shooting")

    to_rank = _percentile_rank(s.turnover_rate_off, [t.stats.turnover_rate_off for t in all_teams], higher_is_better=False)
    if to_rank >= 80:
        strengths.append("Takes care of the ball")

    if s.tournament_experience >= 3:
        strengths.append("Tournament-tested program")

    if s.last10_wins >= 9:
        strengths.append("Hot streak entering tournament")

    c = team.coach
    if c.tourney_appearances >= 15:
        strengths.append("Elite tournament coach")
    elif c.tourney_appearances >= 8:
        strengths.append("Experienced tournament coach")

    if c.final_fours >= 3:
        strengths.append("Coach has Final Four pedigree")

    if c.championships >= 1:
        strengths.append("Coach is a national champion")

    if s.bpi_rank > 0 and s.bpi_rank <= 10:
        strengths.append("Top-10 BPI power rating")

    if s.sor_rank > 0 and s.sor_rank <= 10:
        strengths.append("Elite strength of record")

    if s.quality_wins >= 12:
        strengths.append("Battle-tested vs top-50 opponents")

    roster = compute_roster_metrics(team)
    if roster["deep_rotation_count"] >= 8:
        strengths.append("Deep bench")
    if roster["top_scorer_share"] > 0 and roster["top_scorer_share"] <= 0.22:
        strengths.append("Balanced scoring attack")

    return strengths


def _identify_weaknesses(team: Team, all_teams: list[Team]) -> list[str]:
    s = team.stats
    weaknesses = []

    off_rank = _percentile_rank(s.adj_offensive_efficiency, [t.stats.adj_offensive_efficiency for t in all_teams])
    if off_rank <= 30:
        weaknesses.append("Below-average offense")

    def_rank = _percentile_rank(s.adj_defensive_efficiency, [t.stats.adj_defensive_efficiency for t in all_teams], higher_is_better=False)
    if def_rank <= 30:
        weaknesses.append("Porous defense")

    if s.three_pt_pct > 0 and s.three_pt_pct < 32:
        weaknesses.append("Poor 3-point shooting")

    if s.ft_pct > 0 and s.ft_pct < 68:
        weaknesses.append("Weak free throw shooting")

    if s.turnover_rate_off >= 20:
        weaknesses.append("Turnover-prone")

    if s.strength_of_schedule < -2:
        weaknesses.append("Weak schedule")

    if s.last10_wins <= 5 and s.last10_losses >= 5:
        weaknesses.append("Struggling entering tournament")

    c = team.coach
    if c.name and c.tourney_appearances == 0:
        weaknesses.append("Coach has zero tournament experience")
    elif c.name and c.years_coaching >= 5 and c.tourney_appearances <= 1:
        weaknesses.append("Coach has minimal tournament experience")

    if s.quality_wins + s.quality_losses > 0 and s.quality_win_ratio < 0.33:
        weaknesses.append("Poor record vs top-50 opponents")

    roster = compute_roster_metrics(team)
    if roster["deep_rotation_count"] > 0 and roster["deep_rotation_count"] <= 5:
        weaknesses.append("Thin rotation")
    if roster["top_scorer_share"] >= 0.35:
        weaknesses.append("Over-reliant on one scorer")

    return weaknesses


def build_all_profiles(teams: list[Team], historical: HistoricalAnalyzer | None = None) -> dict[str, TeamProfile]:
    if historical:
        team_names = [t.name for t in teams]
        exp_scores = historical.team_tournament_experience(team_names)
        for team in teams:
            team.stats.tournament_experience = exp_scores.get(team.name, 0.0)
    return {team.name: build_profile(team, teams) for team in teams}
