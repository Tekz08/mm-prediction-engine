import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.team_stats import load_teams
from src.analysis.historical import HistoricalAnalyzer
from src.analysis.profiler import build_all_profiles
from src.analysis.matchup import MatchupEvaluator
from src.analysis.advisor import BracketAdvisor
from src.simulation.bracket import BracketSimulator, load_bracket
from src.simulation.engine import MonteCarloEngine
from src.simulation.results import SimulationResults


def _build_advisor(iterations=200, seed=42):
    teams = load_teams()
    teams_by_name = {t.name: t for t in teams}
    bracket = load_bracket()
    historical = HistoricalAnalyzer()
    profiles = build_all_profiles(teams, historical)
    evaluator = MatchupEvaluator(historical, profiles)
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=iterations, seed=seed)
    raw = engine.run()
    sim_results = SimulationResults(raw)
    consensus = engine.predict_consensus_bracket()
    adv = BracketAdvisor(evaluator, sim_results, teams_by_name, profiles)
    picks = adv.generate_bracket(consensus)
    return adv, picks


def test_complete_bracket():
    adv, picks = _build_advisor()
    assert len(picks) == 67, f"Expected 67 picks (4 FF + 32 R64 + 16 R32 + 8 S16 + 4 E8 + 2 FF + 1 Champ), got {len(picks)}"


def test_all_fields_populated():
    adv, picks = _build_advisor()
    for p in picks:
        assert p.pick, f"Pick should not be empty for {p.team_a} vs {p.team_b}"
        assert p.confidence, f"Confidence should not be empty for {p.team_a} vs {p.team_b}"
        assert p.reasoning, f"Reasoning should not be empty for {p.team_a} vs {p.team_b}"


def test_upsets_recommended():
    adv, picks = _build_advisor()
    upsets = [p for p in picks if p.is_upset]
    assert len(upsets) >= 1, "Advisor should recommend at least 1 upset"


def test_champion_is_realistic():
    adv, picks = _build_advisor()
    champ = adv.get_champion(picks)
    assert champ is not None, "Should have a championship pick"
    assert champ.pick_seed <= 4, f"Champion should typically be a top-4 seed, got {champ.pick_seed}-seed {champ.pick}"


def test_confidence_levels_valid():
    adv, picks = _build_advisor()
    valid = {"Lock", "Strong", "Lean", "Toss-up"}
    for p in picks:
        assert p.confidence in valid, f"Invalid confidence '{p.confidence}' for {p.team_a} vs {p.team_b}"


def test_to_dict_structure():
    adv, picks = _build_advisor()
    result = adv.to_dict(picks)
    assert "champion" in result
    assert "final_four" in result
    assert "smart_upsets" in result
    assert "rounds" in result
    assert "total_picks" in result
    assert result["total_picks"] == 67
    assert len(result["final_four"]) == 4


if __name__ == "__main__":
    test_complete_bracket()
    test_all_fields_populated()
    test_upsets_recommended()
    test_champion_is_realistic()
    test_confidence_levels_valid()
    test_to_dict_structure()
    print("All advisor tests passed!")
