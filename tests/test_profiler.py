import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.models import Team, TeamStats, PlayerStats
from src.data.team_stats import load_teams
from src.analysis.profiler import build_profile, build_all_profiles, compute_roster_metrics


def test_profile_scores_bounded():
    teams = load_teams()
    profiles = build_all_profiles(teams)

    for name, prof in profiles.items():
        assert 0 <= prof.offensive_score <= 100, f"{name} offensive_score out of range: {prof.offensive_score}"
        assert 0 <= prof.defensive_score <= 100, f"{name} defensive_score out of range: {prof.defensive_score}"
        assert 0 <= prof.composite_score <= 100, f"{name} composite_score out of range: {prof.composite_score}"


def test_top_seed_has_high_composite():
    teams = load_teams()
    profiles = build_all_profiles(teams)

    duke = profiles.get("Duke")
    assert duke is not None
    assert duke.composite_score > 60, f"Duke (1-seed) should have high composite, got {duke.composite_score}"


def test_low_seed_has_lower_composite():
    teams = load_teams()
    profiles = build_all_profiles(teams)

    duke = profiles["Duke"]
    siena = profiles.get("Siena")
    assert siena is not None
    assert duke.composite_score > siena.composite_score, "1-seed should rate higher than 16-seed"


def test_strengths_identified():
    teams = load_teams()
    profiles = build_all_profiles(teams)

    duke = profiles["Duke"]
    assert len(duke.strengths) > 0, "Top team should have identified strengths"


def test_weaknesses_for_weak_teams():
    teams = load_teams()
    team_16 = Team(
        name="TestWeak",
        seed=16,
        region="East",
        stats=TeamStats(
            adj_offensive_efficiency=98,
            adj_defensive_efficiency=110,
            three_pt_pct=28,
            ft_pct=62,
            turnover_rate_off=22,
            strength_of_schedule=-8,
            last10_wins=3,
            last10_losses=7,
        ),
    )
    all_teams = teams + [team_16]
    prof = build_profile(team_16, all_teams)
    assert len(prof.weaknesses) > 0, "Weak team should have identified weaknesses"


def test_all_68_teams_profiled():
    teams = load_teams()
    profiles = build_all_profiles(teams)
    assert len(profiles) == 68, f"Expected 68 teams, got {len(profiles)}"


def test_star_dependency_computed():
    star_heavy_players = [
        PlayerStats(name="Star", ppg=22.0, mpg=35.0),
        PlayerStats(name="P2", ppg=5.0, mpg=20.0),
        PlayerStats(name="P3", ppg=5.0, mpg=18.0),
        PlayerStats(name="P4", ppg=5.0, mpg=15.0),
        PlayerStats(name="P5", ppg=5.0, mpg=12.0),
    ]
    balanced_players = [
        PlayerStats(name="P1", ppg=10.0, mpg=30.0),
        PlayerStats(name="P2", ppg=10.0, mpg=28.0),
        PlayerStats(name="P3", ppg=10.0, mpg=26.0),
        PlayerStats(name="P4", ppg=10.0, mpg=24.0),
        PlayerStats(name="P5", ppg=10.0, mpg=22.0),
    ]

    star_team = Team(name="StarTeam", seed=5, region="East", players=star_heavy_players)
    balanced_team = Team(name="BalancedTeam", seed=5, region="East", players=balanced_players)

    star_metrics = compute_roster_metrics(star_team)
    balanced_metrics = compute_roster_metrics(balanced_team)

    assert star_metrics["top_scorer_share"] > balanced_metrics["top_scorer_share"], \
        f"Star team should have higher top_scorer_share ({star_metrics['top_scorer_share']} vs {balanced_metrics['top_scorer_share']})"


def test_bench_depth_computed():
    deep_players = [PlayerStats(name=f"P{i}", ppg=8.0, mpg=12.0 + i) for i in range(9)]
    thin_players = [PlayerStats(name=f"P{i}", ppg=12.0, mpg=25.0) for i in range(4)] + \
                   [PlayerStats(name=f"B{i}", ppg=2.0, mpg=5.0) for i in range(4)]

    deep_team = Team(name="DeepTeam", seed=5, region="East", players=deep_players,
                     stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100))
    thin_team = Team(name="ThinTeam", seed=5, region="East", players=thin_players,
                     stats=TeamStats(adj_offensive_efficiency=112, adj_defensive_efficiency=100))

    deep_metrics = compute_roster_metrics(deep_team)
    thin_metrics = compute_roster_metrics(thin_team)

    assert deep_metrics["deep_rotation_count"] >= 8
    assert thin_metrics["deep_rotation_count"] <= 5

    all_teams = [deep_team, thin_team]
    deep_prof = build_profile(deep_team, all_teams)
    thin_prof = build_profile(thin_team, all_teams)

    assert any("Deep bench" in s for s in deep_prof.strengths), f"Deep team should have 'Deep bench' strength: {deep_prof.strengths}"
    assert any("Thin rotation" in w for w in thin_prof.weaknesses), f"Thin team should have 'Thin rotation' weakness: {thin_prof.weaknesses}"


if __name__ == "__main__":
    test_profile_scores_bounded()
    test_top_seed_has_high_composite()
    test_low_seed_has_lower_composite()
    test_strengths_identified()
    test_weaknesses_for_weak_teams()
    test_all_68_teams_profiled()
    test_star_dependency_computed()
    test_bench_depth_computed()
    print("All profiler tests passed!")
