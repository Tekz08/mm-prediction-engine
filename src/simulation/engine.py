from __future__ import annotations

import copy
import random
from collections import defaultdict

import numpy as np

import config
from src.analysis.matchup import MatchupEvaluator
from src.analysis.profiler import compute_roster_metrics
from src.data.models import BracketEntry, Team
from src.simulation.bracket import BracketSimulator, build_region_matchups, entry_to_team


class MonteCarloEngine:
    def __init__(
        self,
        bracket_sim: BracketSimulator,
        evaluator: MatchupEvaluator,
        iterations: int | None = None,
        seed: int | None = None,
    ):
        self.bracket_sim = bracket_sim
        self.evaluator = evaluator
        self.iterations = iterations or config.DEFAULT_ITERATIONS
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._roster_cache: dict[str, dict] = {}

    def _get_roster_metrics(self, team: Team) -> dict:
        if team.name not in self._roster_cache:
            self._roster_cache[team.name] = compute_roster_metrics(team)
        return self._roster_cache[team.name]

    def _team_stdev(self, team: Team, round_name: str) -> float:
        base = config.GAME_STDEV
        round_mult = config.ROUND_VARIANCE_MULT.get(round_name, 1.0)
        base *= round_mult

        roster = self._get_roster_metrics(team)
        star_share = roster.get("top_scorer_share", 0.0)
        if star_share > 0.25:
            base *= 1.0 + (star_share - 0.25) * config.STAR_DEPENDENCY_STDEV_MULT / 0.1

        depth = roster.get("deep_rotation_count", 6)
        depth_bonus = config.BENCH_DEPTH_ROUND_BONUS.get(round_name, 0.0)
        if depth >= 8:
            base *= (1.0 - depth_bonus)
        elif depth <= 5:
            base *= (1.0 + depth_bonus)

        return base

    def _experience_bonus(self, team: Team, round_name: str) -> float:
        premium = config.ROUND_EXPERIENCE_PREMIUM.get(round_name, 0.0)
        if premium == 0:
            return 0.0
        coach_score = 0.0
        c = team.coach
        if c.name:
            coach_score = min(c.tourney_appearances / 20.0, 1.0) * 0.5
            coach_score += min(c.final_fours / 5.0, 1.0) * 0.3
            coach_score += min(c.championships / 3.0, 1.0) * 0.2
        program_exp = min(team.stats.tournament_experience / 5.0, 1.0) if team.stats.tournament_experience > 0 else 0.0
        total = (coach_score * 0.6 + program_exp * 0.4)
        return total * premium

    def _defense_premium(self, team: Team, round_name: str) -> float:
        premium = config.ROUND_DEFENSE_PREMIUM.get(round_name, 0.0)
        if premium == 0 or team.stats.adj_defensive_efficiency == 0:
            return 0.0
        def_quality = max(0, 105 - team.stats.adj_defensive_efficiency) / 20.0
        return def_quality * premium * 2.0

    def simulate_game(self, team_a: Team, team_b: Team, round_name: str) -> tuple[Team, int, int]:
        sa, sb = team_a.stats, team_b.stats

        if sa.tempo > 0 and sb.tempo > 0:
            possessions = (sa.tempo + sb.tempo) / 2.0
        else:
            possessions = 68.0

        league_avg = config.LEAGUE_AVG_EFFICIENCY
        off_a = sa.adj_offensive_efficiency if sa.adj_offensive_efficiency > 0 else league_avg
        def_a = sa.adj_defensive_efficiency if sa.adj_defensive_efficiency > 0 else league_avg
        off_b = sb.adj_offensive_efficiency if sb.adj_offensive_efficiency > 0 else league_avg
        def_b = sb.adj_defensive_efficiency if sb.adj_defensive_efficiency > 0 else league_avg

        expected_a = possessions * off_a * (def_b / league_avg) / 100.0
        expected_b = possessions * off_b * (def_a / league_avg) / 100.0

        matchup_result = self.evaluator.evaluate(team_a, team_b, round_name)
        matchup_edge = (matchup_result.win_prob_a - 0.5) * 6.0
        expected_a += matchup_edge / 2.0
        expected_b -= matchup_edge / 2.0

        exp_bonus_a = self._experience_bonus(team_a, round_name)
        exp_bonus_b = self._experience_bonus(team_b, round_name)
        expected_a += exp_bonus_a
        expected_b += exp_bonus_b

        def_prem_a = self._defense_premium(team_a, round_name)
        def_prem_b = self._defense_premium(team_b, round_name)
        expected_b -= def_prem_a
        expected_a -= def_prem_b

        stdev_a = self._team_stdev(team_a, round_name)
        stdev_b = self._team_stdev(team_b, round_name)

        score_a = self.np_rng.normal(expected_a, stdev_a)
        score_b = self.np_rng.normal(expected_b, stdev_b)

        score_a = max(35, round(score_a))
        score_b = max(35, round(score_b))

        if score_a == score_b:
            if self.rng.random() < matchup_result.win_prob_a:
                score_a += 1
            else:
                score_b += 1

        if score_a > score_b:
            return team_a, score_a, score_b
        else:
            return team_b, score_b, score_a

    def simulate_round(self, matchups: list[tuple[Team, Team]], round_name: str) -> list[Team]:
        return [self.simulate_game(a, b, round_name)[0] for a, b in matchups]

    def _resolve_bracket_copy(self) -> dict[str, list[BracketEntry]]:
        bracket = copy.deepcopy(self.bracket_sim.bracket)
        teams = self.bracket_sim.teams_by_name
        first_four_results = {}

        for region, entries in bracket.items():
            for i, entry in enumerate(entries):
                if entry.play_in:
                    team_a = entry_to_team(entry, teams)
                    opp_entry = BracketEntry(seed=entry.seed, team=entry.play_in, region=region)
                    team_b = entry_to_team(opp_entry, teams)
                    winner, _, _ = self.simulate_game(team_a, team_b, "First Four")
                    first_four_results[f"{team_a.name} vs {team_b.name}"] = winner.name
                    bracket[region][i] = BracketEntry(
                        seed=entry.seed, team=winner.name, region=region,
                    )

        return bracket, first_four_results

    def simulate_tournament(self) -> dict:
        bracket, first_four_results = self._resolve_bracket_copy()
        teams = self.bracket_sim.teams_by_name
        round_names = config.ROUNDS
        results = {"rounds": {}, "champion": None, "first_four_results": first_four_results}
        region_winners = {}

        for region, entries in bracket.items():
            pairs = build_region_matchups(entries)
            matchups = [
                (entry_to_team(a, teams), entry_to_team(b, teams))
                for a, b in pairs
            ]
            region_results = {}
            current_matchups = matchups
            round_idx = 0

            while len(current_matchups) > 0:
                rname = round_names[round_idx] if round_idx < len(round_names) else f"Round {round_idx}"
                winners = self.simulate_round(current_matchups, rname)
                region_results[rname] = [w.name for w in winners]

                if len(winners) == 1:
                    region_winners[region] = winners[0]
                    break

                current_matchups = []
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        current_matchups.append((winners[i], winners[i + 1]))
                round_idx += 1

            results["rounds"][region] = region_results

        ff_matchups = self.bracket_sim.get_final_four_matchups(region_winners)
        if len(ff_matchups) == 2:
            ff_winners = self.simulate_round(ff_matchups, "Final Four")
            results["final_four"] = [rw.name for rw in region_winners.values()]
            results["championship_matchup"] = [w.name for w in ff_winners]

            if len(ff_winners) == 2:
                champion, _, _ = self.simulate_game(ff_winners[0], ff_winners[1], "Championship")
                results["champion"] = champion.name

        return results

    def predict_consensus_bracket(self) -> dict:
        bracket = copy.deepcopy(self.bracket_sim.bracket)
        teams = self.bracket_sim.teams_by_name
        round_names = config.ROUNDS
        result = {"first_four": [], "regions": {}, "final_four": [], "championship": None}

        for region, entries in bracket.items():
            for i, entry in enumerate(entries):
                if entry.play_in:
                    team_a = entry_to_team(entry, teams)
                    opp_entry = BracketEntry(seed=entry.seed, team=entry.play_in, region=region)
                    team_b = entry_to_team(opp_entry, teams)
                    prob_a = self.evaluator.win_probability(team_a, team_b, "First Four")
                    winner = team_a if prob_a >= 0.5 else team_b
                    result["first_four"].append({
                        "team_a": {"name": team_a.name, "seed": team_a.seed},
                        "team_b": {"name": team_b.name, "seed": team_b.seed},
                        "winner": winner.name,
                        "prob": round(max(prob_a, 1 - prob_a) * 100, 1),
                        "region": region,
                    })
                    bracket[region][i] = BracketEntry(seed=entry.seed, team=winner.name, region=region)

        region_winners = {}
        for region, entries in bracket.items():
            pairs = build_region_matchups(entries)
            matchups = [
                (entry_to_team(a, teams), entry_to_team(b, teams))
                for a, b in pairs
            ]
            region_data = []
            current_matchups = matchups
            round_idx = 0

            while current_matchups:
                rname = round_names[round_idx] if round_idx < len(round_names) else f"Round {round_idx}"
                round_games = []
                winners = []
                for team_a, team_b in current_matchups:
                    prob_a = self.evaluator.win_probability(team_a, team_b, rname)
                    winner = team_a if prob_a >= 0.5 else team_b
                    round_games.append({
                        "team_a": {"name": team_a.name, "seed": team_a.seed},
                        "team_b": {"name": team_b.name, "seed": team_b.seed},
                        "winner": winner.name,
                        "prob": round(max(prob_a, 1 - prob_a) * 100, 1),
                    })
                    winners.append(winner)

                region_data.append({"round": rname, "games": round_games})

                if len(winners) == 1:
                    region_winners[region] = winners[0]
                    break

                current_matchups = [
                    (winners[i], winners[i + 1])
                    for i in range(0, len(winners) - 1, 2)
                ]
                round_idx += 1

            result["regions"][region] = region_data

        ff_matchups = self.bracket_sim.get_final_four_matchups(region_winners)
        if len(ff_matchups) == 2:
            ff_games = []
            ff_winners = []
            for team_a, team_b in ff_matchups:
                prob_a = self.evaluator.win_probability(team_a, team_b, "Final Four")
                winner = team_a if prob_a >= 0.5 else team_b
                ff_games.append({
                    "team_a": {"name": team_a.name, "seed": team_a.seed},
                    "team_b": {"name": team_b.name, "seed": team_b.seed},
                    "winner": winner.name,
                    "prob": round(max(prob_a, 1 - prob_a) * 100, 1),
                })
                ff_winners.append(winner)
            result["final_four"] = ff_games

            if len(ff_winners) == 2:
                prob_a = self.evaluator.win_probability(ff_winners[0], ff_winners[1], "Championship")
                champ = ff_winners[0] if prob_a >= 0.5 else ff_winners[1]
                result["championship"] = {
                    "team_a": {"name": ff_winners[0].name, "seed": ff_winners[0].seed},
                    "team_b": {"name": ff_winners[1].name, "seed": ff_winners[1].seed},
                    "winner": champ.name,
                    "prob": round(max(prob_a, 1 - prob_a) * 100, 1),
                }

        return result

    def run(self, progress_callback=None) -> dict:
        team_round_counts = defaultdict(lambda: defaultdict(int))
        championship_wins = defaultdict(int)
        final_four_appearances = defaultdict(int)
        final_four_combos = defaultdict(int)
        championship_matchups = defaultdict(int)

        for i in range(self.iterations):
            result = self.simulate_tournament()

            if result.get("champion"):
                championship_wins[result["champion"]] += 1

            if result.get("final_four"):
                for team in result["final_four"]:
                    final_four_appearances[team] += 1
                combo_key = tuple(sorted(result["final_four"]))
                final_four_combos[combo_key] += 1

            if result.get("championship_matchup"):
                cm_key = tuple(sorted(result["championship_matchup"]))
                championship_matchups[cm_key] += 1

            for region, rounds in result.get("rounds", {}).items():
                for round_name, winners in rounds.items():
                    for team in winners:
                        team_round_counts[team][round_name] += 1

            if progress_callback and (i + 1) % max(1, self.iterations // 100) == 0:
                progress_callback(i + 1, self.iterations)

        return {
            "iterations": self.iterations,
            "team_round_counts": dict(team_round_counts),
            "championship_wins": dict(championship_wins),
            "final_four_appearances": dict(final_four_appearances),
            "final_four_combos": {
                str(k): v for k, v in sorted(final_four_combos.items(), key=lambda x: -x[1])[:10]
            },
            "championship_matchups": {
                str(k): v for k, v in sorted(championship_matchups.items(), key=lambda x: -x[1])[:10]
            },
        }
