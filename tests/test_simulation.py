import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.models import Team, TeamStats, TeamProfile, CoachInfo
from src.data.team_stats import load_teams, load_teams_by_name
from src.analysis.historical import HistoricalAnalyzer
from src.analysis.profiler import build_all_profiles
from src.analysis.matchup import MatchupEvaluator
from src.simulation.bracket import BracketSimulator, load_bracket
from src.simulation.engine import MonteCarloEngine
from src.simulation.results import SimulationResults


def _build_test_engine(iterations=100, seed=42):
    teams = load_teams()
    teams_by_name = {t.name: t for t in teams}
    bracket = load_bracket()
    historical = HistoricalAnalyzer()
    profiles = build_all_profiles(teams, historical)
    evaluator = MatchupEvaluator(historical, profiles)
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=iterations, seed=seed)
    return engine, teams, teams_by_name, profiles


def test_simulation_produces_champion():
    engine, _, _, _ = _build_test_engine(iterations=100)
    raw = engine.run()
    assert raw["iterations"] == 100
    assert len(raw["championship_wins"]) > 0
    assert sum(raw["championship_wins"].values()) == 100


def test_simulation_deterministic_with_seed():
    engine1, _, _, _ = _build_test_engine(iterations=50, seed=123)
    engine2, _, _, _ = _build_test_engine(iterations=50, seed=123)
    raw1 = engine1.run()
    raw2 = engine2.run()
    assert raw1["championship_wins"] == raw2["championship_wins"]


def test_results_aggregation():
    engine, _, _, _ = _build_test_engine(iterations=200)
    raw = engine.run()
    results = SimulationResults(raw)

    champion, pct = results.most_likely_champion()
    assert champion != "Unknown"
    assert pct > 0

    tiers = results.confidence_tiers()
    assert "locks" in tiers
    assert len(tiers["locks"]) > 0

    summary = results.to_summary_dict()
    assert "predicted_champion" in summary
    assert "championship_probabilities" in summary


def test_one_seed_dominates():
    engine, _, _, _ = _build_test_engine(iterations=500)
    raw = engine.run()
    results = SimulationResults(raw)
    champ_probs = results.championship_probabilities()

    one_seeds = {"Duke", "Arizona", "Michigan", "Florida"}
    one_seed_total = sum(champ_probs.get(t, 0) for t in one_seeds)
    assert one_seed_total > 20, f"1-seeds should have significant championship share, got {one_seed_total}%"


def test_scores_in_realistic_range():
    one_seed = Team(name="TopDog", seed=1, region="East",
                    stats=TeamStats(adj_offensive_efficiency=120, adj_defensive_efficiency=92, tempo=70))
    sixteen_seed = Team(name="Underdog", seed=16, region="East",
                        stats=TeamStats(adj_offensive_efficiency=100, adj_defensive_efficiency=107, tempo=65))

    historical = HistoricalAnalyzer()
    profiles = {
        t.name: TeamProfile(team=t, composite_score=50 + t.stats.net_efficiency)
        for t in [one_seed, sixteen_seed]
    }
    evaluator = MatchupEvaluator(historical, profiles)
    bracket = load_bracket()
    teams_by_name = {one_seed.name: one_seed, sixteen_seed.name: sixteen_seed}
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=1, seed=42)

    for _ in range(100):
        _, score_a, score_b = engine.simulate_game(one_seed, sixteen_seed, "Round of 64")
        assert 35 <= score_a <= 120, f"Score out of range: {score_a}"
        assert 35 <= score_b <= 120, f"Score out of range: {score_b}"


def test_upsets_rare_for_mismatches():
    one_seed = Team(name="TopDog", seed=1, region="East",
                    stats=TeamStats(adj_offensive_efficiency=125, adj_defensive_efficiency=90, tempo=70))
    sixteen_seed = Team(name="Underdog", seed=16, region="East",
                        stats=TeamStats(adj_offensive_efficiency=98, adj_defensive_efficiency=108, tempo=64))

    historical = HistoricalAnalyzer()
    profiles = {
        t.name: TeamProfile(team=t, composite_score=50 + t.stats.net_efficiency)
        for t in [one_seed, sixteen_seed]
    }
    evaluator = MatchupEvaluator(historical, profiles)
    bracket = load_bracket()
    teams_by_name = {one_seed.name: one_seed, sixteen_seed.name: sixteen_seed}
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=1, seed=42)

    sixteen_wins = 0
    for _ in range(1000):
        winner, _, _ = engine.simulate_game(one_seed, sixteen_seed, "Round of 64")
        if winner.name == sixteen_seed.name:
            sixteen_wins += 1

    pct = sixteen_wins / 10.0
    assert 0 <= pct <= 10.0, f"16-seed should win <10% of games, got {pct}%"


def test_close_matchups_have_variance():
    team_a = Team(name="TeamA", seed=8, region="East",
                  stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68))
    team_b = Team(name="TeamB", seed=9, region="East",
                  stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100, tempo=68))

    historical = HistoricalAnalyzer()
    profiles = {
        t.name: TeamProfile(team=t, composite_score=50 + t.stats.net_efficiency)
        for t in [team_a, team_b]
    }
    evaluator = MatchupEvaluator(historical, profiles)
    bracket = load_bracket()
    teams_by_name = {team_a.name: team_a, team_b.name: team_b}
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=1, seed=42)

    a_wins = 0
    for _ in range(1000):
        winner, _, _ = engine.simulate_game(team_a, team_b, "Round of 64")
        if winner.name == team_a.name:
            a_wins += 1

    pct = a_wins / 10.0
    assert 30 <= pct <= 70, f"Close matchup should be roughly even, got {pct}% for team A"


def test_experienced_coach_benefits_late():
    base_stats = TeamStats(adj_offensive_efficiency=115, adj_defensive_efficiency=98, tempo=68)
    experienced_coach = CoachInfo(name="VetCoach", tourney_appearances=20, final_fours=5,
                                  championships=2, career_win_pct=0.72, years_coaching=25,
                                  career_wins=600, career_losses=200)
    rookie_coach = CoachInfo(name="NewCoach", tourney_appearances=0, final_fours=0,
                              championships=0, career_win_pct=0.55, years_coaching=3,
                              career_wins=50, career_losses=40)

    exp_team = Team(name="VetTeam", seed=3, region="East", stats=base_stats.model_copy(), coach=experienced_coach)
    rookie_team = Team(name="NewTeam", seed=3, region="East", stats=base_stats.model_copy(), coach=rookie_coach)
    opp_coach = CoachInfo(name="OppCoach", tourney_appearances=5, career_win_pct=0.60, years_coaching=10,
                          career_wins=200, career_losses=130)
    opponent = Team(name="Opponent", seed=3, region="East", stats=base_stats.model_copy(), coach=opp_coach)

    historical = HistoricalAnalyzer()
    profiles = {
        t.name: TeamProfile(team=t, composite_score=65)
        for t in [exp_team, rookie_team, opponent]
    }
    evaluator = MatchupEvaluator(historical, profiles)
    bracket = load_bracket()
    teams_by_name = {t.name: t for t in [exp_team, rookie_team, opponent]}
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=1, seed=42)

    n = 3000
    exp_wins_late = sum(
        1 for _ in range(n)
        if engine.simulate_game(exp_team, opponent, "Elite 8")[0].name == "VetTeam"
    )
    rookie_wins_late = sum(
        1 for _ in range(n)
        if engine.simulate_game(rookie_team, opponent, "Elite 8")[0].name == "NewTeam"
    )

    assert exp_wins_late > rookie_wins_late, \
        f"Experienced coach should win more in E8 ({exp_wins_late} vs {rookie_wins_late})"


if __name__ == "__main__":
    test_simulation_produces_champion()
    test_simulation_deterministic_with_seed()
    test_results_aggregation()
    test_one_seed_dominates()
    test_scores_in_realistic_range()
    test_upsets_rare_for_mismatches()
    test_close_matchups_have_variance()
    test_experienced_coach_benefits_late()
    print("All simulation tests passed!")
