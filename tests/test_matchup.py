import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.models import Team, TeamStats, TeamProfile, PlayerStats, ScheduleGame, CoachInfo
from src.analysis.historical import HistoricalAnalyzer
from src.analysis.matchup import MatchupEvaluator


def _make_team(name, seed, region="East", off_eff=115.0, def_eff=100.0, **kwargs):
    stats = TeamStats(adj_offensive_efficiency=off_eff, adj_defensive_efficiency=def_eff, **kwargs)
    return Team(name=name, seed=seed, region=region, stats=stats)


def _make_profiles(teams):
    return {
        t.name: TeamProfile(team=t, composite_score=50 + t.stats.net_efficiency)
        for t in teams
    }


def test_higher_seed_favored():
    one_seed = _make_team("TopDog", 1, off_eff=125, def_eff=92)
    sixteen_seed = _make_team("Underdog", 16, off_eff=100, def_eff=107)
    teams = [one_seed, sixteen_seed]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(one_seed, sixteen_seed)
    assert prob > 0.85, f"1-seed should be heavily favored, got {prob}"


def test_equal_seeds_near_fifty():
    team_a = _make_team("TeamA", 8, off_eff=112, def_eff=100)
    team_b = _make_team("TeamB", 9, off_eff=112, def_eff=100)
    teams = [team_a, team_b]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(team_a, team_b)
    assert 0.35 < prob < 0.65, f"Equal teams should be near 50/50, got {prob}"


def test_efficiency_matters():
    strong = _make_team("Strong", 5, off_eff=125, def_eff=90)
    weak = _make_team("Weak", 5, off_eff=105, def_eff=105)
    teams = [strong, weak]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(strong, weak)
    assert prob > 0.6, f"Much better efficiency should win, got {prob}"


def test_probability_bounds():
    best = _make_team("Best", 1, off_eff=130, def_eff=85)
    worst = _make_team("Worst", 16, off_eff=95, def_eff=110)
    teams = [best, worst]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(best, worst)
    assert 0.02 <= prob <= 0.98, f"Probability should be bounded, got {prob}"


def test_matchup_breakdown_has_components():
    a = _make_team("A", 3, off_eff=118, def_eff=95)
    b = _make_team("B", 6, off_eff=113, def_eff=98)
    teams = [a, b]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    breakdown = evaluator.get_matchup_breakdown(a, b)
    assert "components" in breakdown
    assert "historical" in breakdown["components"]
    assert "efficiency" in breakdown["components"]
    assert "profile" in breakdown["components"]
    assert "matchup" in breakdown["components"]
    assert breakdown["win_prob_a"] + breakdown["win_prob_b"] - 1.0 < 0.01


def test_perimeter_mismatch_matters():
    shooter = _make_team("Shooter", 5, off_eff=112, def_eff=100, three_pt_pct=38.0, three_pt_def_pct=32.0, tempo=68.0)

    bad_def = _make_team("BadDef", 5, off_eff=112, def_eff=100, three_pt_pct=33.0, three_pt_def_pct=36.0, tempo=68.0)
    good_def = _make_team("GoodDef", 5, off_eff=112, def_eff=100, three_pt_pct=33.0, three_pt_def_pct=28.0, tempo=68.0)

    historical = HistoricalAnalyzer()
    teams1 = [shooter, bad_def]
    teams2 = [shooter, good_def]
    profiles1 = _make_profiles(teams1)
    profiles2 = _make_profiles(teams2)

    ev1 = MatchupEvaluator(historical, profiles1)
    ev2 = MatchupEvaluator(historical, profiles2)

    prob_vs_bad = ev1.win_probability(shooter, bad_def)
    prob_vs_good = ev2.win_probability(shooter, good_def)
    assert prob_vs_bad > prob_vs_good, f"Shooter should do better vs bad perimeter D ({prob_vs_bad} vs {prob_vs_good})"


def test_turnover_battle_matters():
    careless = _make_team("Careless", 5, off_eff=112, def_eff=100, turnover_rate_off=20.0, turnover_rate_def=15.0, tempo=68.0)
    presser = _make_team("Presser", 5, off_eff=112, def_eff=100, turnover_rate_off=14.0, turnover_rate_def=20.0, tempo=68.0)
    passive = _make_team("Passive", 5, off_eff=112, def_eff=100, turnover_rate_off=14.0, turnover_rate_def=13.0, tempo=68.0)

    historical = HistoricalAnalyzer()
    teams1 = [careless, presser]
    teams2 = [careless, passive]
    profiles1 = _make_profiles(teams1)
    profiles2 = _make_profiles(teams2)

    ev1 = MatchupEvaluator(historical, profiles1)
    ev2 = MatchupEvaluator(historical, profiles2)

    prob_vs_press = ev1.win_probability(careless, presser)
    prob_vs_passive = ev2.win_probability(careless, passive)
    assert prob_vs_press < prob_vs_passive, f"Careless team should do worse vs pressing D ({prob_vs_press} vs {prob_vs_passive})"


def test_matchup_component_bounded():
    extreme_a = _make_team("ExtremeA", 1, off_eff=135, def_eff=80,
                           three_pt_pct=42.0, three_pt_def_pct=28.0,
                           turnover_rate_off=10.0, turnover_rate_def=25.0,
                           off_rebound_rate=40.0, def_rebound_rate=80.0,
                           ft_rate=45.0, ft_pct=82.0, tempo=75.0, strength_of_schedule=10.0)
    extreme_b = _make_team("ExtremeB", 16, off_eff=90, def_eff=115,
                           three_pt_pct=28.0, three_pt_def_pct=38.0,
                           turnover_rate_off=22.0, turnover_rate_def=12.0,
                           off_rebound_rate=22.0, def_rebound_rate=60.0,
                           ft_rate=20.0, ft_pct=60.0, tempo=60.0, strength_of_schedule=-5.0)

    teams = [extreme_a, extreme_b]
    historical = HistoricalAnalyzer()
    profiles = _make_profiles(teams)
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(extreme_a, extreme_b)
    assert 0.02 <= prob <= 0.98, f"Probability must stay bounded, got {prob}"


def test_head_to_head_boosts_probability():
    base = dict(off_eff=112, def_eff=100, tempo=68.0)
    team_a = _make_team("TeamA", 5, **base)
    team_b_no_h2h = _make_team("TeamB", 5, **base)

    team_a_with = Team(name="TeamA", seed=5, region="East",
                       stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68.0),
                       schedule=[ScheduleGame(date="2026-02-15", opponent="TeamB", result="W", team_score=80, opp_score=65)])
    team_b_with = Team(name="TeamB", seed=5, region="East",
                       stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68.0),
                       schedule=[ScheduleGame(date="2026-02-15", opponent="TeamA", result="L", team_score=65, opp_score=80)])

    historical = HistoricalAnalyzer()
    profiles_no = _make_profiles([team_a, team_b_no_h2h])
    profiles_with = _make_profiles([team_a_with, team_b_with])

    ev_no = MatchupEvaluator(historical, profiles_no)
    ev_with = MatchupEvaluator(historical, profiles_with)

    prob_no = ev_no.win_probability(team_a, team_b_no_h2h)
    prob_with = ev_with.win_probability(team_a_with, team_b_with)
    assert prob_with > prob_no, f"H2H win should boost probability ({prob_with} vs {prob_no})"


def test_common_opponents_factor():
    base_stats = TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68.0)

    team_a = Team(name="TeamA", seed=5, region="East", stats=base_stats.model_copy(),
                  schedule=[ScheduleGame(date="2026-01-10", opponent="CommonOpp", result="W", team_score=90, opp_score=70)])
    team_b = Team(name="TeamB", seed=5, region="East", stats=base_stats.model_copy(),
                  schedule=[ScheduleGame(date="2026-01-15", opponent="CommonOpp", result="L", team_score=65, opp_score=75)])

    historical = HistoricalAnalyzer()
    profiles = _make_profiles([team_a, team_b])
    evaluator = MatchupEvaluator(historical, profiles)

    prob = evaluator.win_probability(team_a, team_b)
    assert prob > 0.5, f"Team that crushed common opponent should be favored, got {prob}"


def test_momentum_matters():
    winning_sched = [
        ScheduleGame(date=f"2026-02-{i:02d}", opponent=f"Opp{i}", result="W", team_score=80, opp_score=65)
        for i in range(1, 9)
    ]
    losing_sched = [
        ScheduleGame(date=f"2026-02-{i:02d}", opponent=f"Opp{i}", result="W" if i <= 4 else "L",
                     team_score=70 if i <= 4 else 60, opp_score=65 if i <= 4 else 72)
        for i in range(1, 11)
    ]

    base_stats = TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68.0)
    hot_team = Team(name="HotTeam", seed=5, region="East", stats=base_stats.model_copy(), schedule=winning_sched)
    cold_team = Team(name="ColdTeam", seed=5, region="East", stats=base_stats.model_copy(), schedule=losing_sched)
    neutral = _make_team("Neutral", 5, off_eff=112, def_eff=100)

    historical = HistoricalAnalyzer()
    profiles_hot = _make_profiles([hot_team, neutral])
    profiles_cold = _make_profiles([cold_team, neutral])

    ev_hot = MatchupEvaluator(historical, profiles_hot)
    ev_cold = MatchupEvaluator(historical, profiles_cold)

    prob_hot = ev_hot.win_probability(hot_team, neutral)
    prob_cold = ev_cold.win_probability(cold_team, neutral)
    assert prob_hot > prob_cold, f"Hot team should be favored over cold team ({prob_hot} vs {prob_cold})"


if __name__ == "__main__":
    test_higher_seed_favored()
    test_equal_seeds_near_fifty()
    test_efficiency_matters()
    test_probability_bounds()
    test_matchup_breakdown_has_components()
    test_perimeter_mismatch_matters()
    test_turnover_battle_matters()
    test_matchup_component_bounded()
    test_head_to_head_boosts_probability()
    test_common_opponents_factor()
    test_momentum_matters()
    print("All matchup tests passed!")
