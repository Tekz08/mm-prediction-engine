from __future__ import annotations

import copy
import os
import random
import threading
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import config
from src.analysis.matchup import MatchupEvaluator
from src.analysis.profiler import compute_roster_metrics
from src.data.models import BracketEntry, Team
from src.simulation.bracket import BracketSimulator, build_region_matchups, entry_to_team

_worker_data = None


def _init_worker(bracket_sim, evaluator):
    global _worker_data
    _worker_data = (bracket_sim, evaluator)


def _run_chunk(args):
    iterations, seed = args
    bracket_sim, evaluator = _worker_data
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=iterations, seed=seed)
    return engine._run_batch()


class MonteCarloEngine:
    @staticmethod
    def _matchup_key(round_name: str, team_a: str, team_b: str) -> str:
        a, b = sorted((team_a, team_b))
        return f"{round_name}|{a}|{b}"

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
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._roster_cache: dict[str, dict] = {}
        self._cancel_event = threading.Event()
        self._pool = None

    def cancel(self):
        self._cancel_event.set()
        if self._pool:
            try:
                self._pool.terminate()
            except Exception:
                pass

    @property
    def cancelled(self):
        return self._cancel_event.is_set()

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
        off_a = sa.sos_adjusted_oe if sa.adj_offensive_efficiency > 0 else league_avg
        def_a = sa.sos_adjusted_de if sa.adj_defensive_efficiency > 0 else league_avg
        off_b = sb.sos_adjusted_oe if sb.adj_offensive_efficiency > 0 else league_avg
        def_b = sb.sos_adjusted_de if sb.adj_defensive_efficiency > 0 else league_avg

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
        first_four_games = []

        for region, entries in bracket.items():
            for i, entry in enumerate(entries):
                if entry.play_in:
                    team_a = entry_to_team(entry, teams)
                    opp_entry = BracketEntry(seed=entry.seed, team=entry.play_in, region=region)
                    team_b = entry_to_team(opp_entry, teams)
                    winner, _, _ = self.simulate_game(team_a, team_b, "First Four")
                    first_four_results[f"{team_a.name} vs {team_b.name}"] = winner.name
                    first_four_games.append({
                        "team_a": team_a.name,
                        "team_b": team_b.name,
                        "winner": winner.name,
                    })
                    bracket[region][i] = BracketEntry(
                        seed=entry.seed, team=winner.name, region=region,
                    )

        return bracket, first_four_results, first_four_games

    def simulate_tournament(self) -> dict:
        bracket, first_four_results, first_four_games = self._resolve_bracket_copy()
        teams = self.bracket_sim.teams_by_name
        round_names = config.ROUNDS
        results = {
            "rounds": {},
            "games": {},
            "champion": None,
            "first_four_results": first_four_results,
        }
        if first_four_games:
            results["games"]["First Four"] = first_four_games
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
                winners = []
                for team_a, team_b in current_matchups:
                    winner, _, _ = self.simulate_game(team_a, team_b, rname)
                    winners.append(winner)
                    results["games"].setdefault(rname, []).append({
                        "team_a": team_a.name,
                        "team_b": team_b.name,
                        "winner": winner.name,
                    })
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
            ff_winners = []
            for team_a, team_b in ff_matchups:
                winner, _, _ = self.simulate_game(team_a, team_b, "Final Four")
                ff_winners.append(winner)
                results["games"].setdefault("Final Four", []).append({
                    "team_a": team_a.name,
                    "team_b": team_b.name,
                    "winner": winner.name,
                })
            results["final_four"] = [rw.name for rw in region_winners.values()]
            results["championship_matchup"] = [w.name for w in ff_winners]

            if len(ff_winners) == 2:
                champion, _, _ = self.simulate_game(ff_winners[0], ff_winners[1], "Championship")
                results["champion"] = champion.name
                results["games"].setdefault("Championship", []).append({
                    "team_a": ff_winners[0].name,
                    "team_b": ff_winners[1].name,
                    "winner": champion.name,
                })

        return results

    def _pick_winner(self, team_a: Team, team_b: Team, prob_a: float, round_name: str, upset_bias: float) -> tuple[str, float]:
        if upset_bias <= 0:
            winner = team_a if prob_a >= 0.5 else team_b
            display_prob = max(prob_a, 1 - prob_a) * 100
            return winner.name, round(display_prob, 1)

        threshold = 0.5 + upset_bias * 0.20
        fav_prob = max(prob_a, 1 - prob_a)
        favorite = team_a if prob_a >= 0.5 else team_b
        underdog = team_b if prob_a >= 0.5 else team_a

        if fav_prob >= threshold:
            return favorite.name, round(fav_prob * 100, 1)
        return underdog.name, round((1 - fav_prob) * 100, 1)

    def predict_consensus_bracket(self, upset_bias: float = 0.0) -> dict:
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
                    winner_name, display_prob = self._pick_winner(team_a, team_b, prob_a, "First Four", upset_bias)
                    winner = team_a if winner_name == team_a.name else team_b
                    result["first_four"].append({
                        "team_a": {"name": team_a.name, "seed": team_a.seed},
                        "team_b": {"name": team_b.name, "seed": team_b.seed},
                        "winner": winner_name,
                        "prob": display_prob,
                        "region": region,
                    })
                    bracket[region][i] = BracketEntry(seed=entry.seed, team=winner_name, region=region)

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
                    winner_name, display_prob = self._pick_winner(team_a, team_b, prob_a, rname, upset_bias)
                    winner = team_a if winner_name == team_a.name else team_b
                    round_games.append({
                        "team_a": {"name": team_a.name, "seed": team_a.seed},
                        "team_b": {"name": team_b.name, "seed": team_b.seed},
                        "winner": winner_name,
                        "prob": display_prob,
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
                winner_name, display_prob = self._pick_winner(team_a, team_b, prob_a, "Final Four", upset_bias)
                winner = team_a if winner_name == team_a.name else team_b
                ff_games.append({
                    "team_a": {"name": team_a.name, "seed": team_a.seed},
                    "team_b": {"name": team_b.name, "seed": team_b.seed},
                    "winner": winner_name,
                    "prob": display_prob,
                })
                ff_winners.append(winner)
            result["final_four"] = ff_games

            if len(ff_winners) == 2:
                prob_a = self.evaluator.win_probability(ff_winners[0], ff_winners[1], "Championship")
                champ_name, display_prob = self._pick_winner(ff_winners[0], ff_winners[1], prob_a, "Championship", upset_bias)
                champ = ff_winners[0] if champ_name == ff_winners[0].name else ff_winners[1]
                result["championship"] = {
                    "team_a": {"name": ff_winners[0].name, "seed": ff_winners[0].seed},
                    "team_b": {"name": ff_winners[1].name, "seed": ff_winners[1].seed},
                    "winner": champ_name,
                    "prob": display_prob,
                }

        return result

    def run(self, progress_callback=None, workers=None) -> dict:
        if workers is None:
            workers = os.cpu_count() or 1

        if workers <= 1 or self.iterations < 100:
            return self._run_single(progress_callback)

        return self._run_parallel(workers, progress_callback)

    def _run_batch(self) -> dict:
        team_round_counts = defaultdict(lambda: defaultdict(int))
        championship_wins = defaultdict(int)
        final_four_appearances = defaultdict(int)
        final_four_combos = defaultdict(int)
        championship_matchups = defaultdict(int)
        matchup_meet_counts = defaultdict(int)
        matchup_win_counts = defaultdict(lambda: defaultdict(int))

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

            if result.get("final_four"):
                for team in result["final_four"]:
                    team_round_counts[team]["Final Four"] += 1
            if result.get("championship_matchup"):
                for team in result["championship_matchup"]:
                    team_round_counts[team]["Championship"] += 1
            for round_name, games in result.get("games", {}).items():
                for game in games:
                    key = self._matchup_key(round_name, game["team_a"], game["team_b"])
                    matchup_meet_counts[key] += 1
                    matchup_win_counts[key][game["winner"]] += 1

        return {
            "iterations": self.iterations,
            "team_round_counts": {t: dict(r) for t, r in team_round_counts.items()},
            "championship_wins": dict(championship_wins),
            "final_four_appearances": dict(final_four_appearances),
            "final_four_combos": dict(final_four_combos),
            "championship_matchups": dict(championship_matchups),
            "matchup_meet_counts": dict(matchup_meet_counts),
            "matchup_win_counts": {k: dict(v) for k, v in matchup_win_counts.items()},
        }

    def _run_single(self, progress_callback=None) -> dict:
        team_round_counts = defaultdict(lambda: defaultdict(int))
        championship_wins = defaultdict(int)
        final_four_appearances = defaultdict(int)
        final_four_combos = defaultdict(int)
        championship_matchups = defaultdict(int)
        matchup_meet_counts = defaultdict(int)
        matchup_win_counts = defaultdict(lambda: defaultdict(int))

        for i in range(self.iterations):
            if self.cancelled:
                break
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

            if result.get("final_four"):
                for team in result["final_four"]:
                    team_round_counts[team]["Final Four"] += 1
            if result.get("championship_matchup"):
                for team in result["championship_matchup"]:
                    team_round_counts[team]["Championship"] += 1
            for round_name, games in result.get("games", {}).items():
                for game in games:
                    key = self._matchup_key(round_name, game["team_a"], game["team_b"])
                    matchup_meet_counts[key] += 1
                    matchup_win_counts[key][game["winner"]] += 1

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
            "matchup_meet_counts": dict(matchup_meet_counts),
            "matchup_win_counts": {k: dict(v) for k, v in matchup_win_counts.items()},
        }

    def _run_parallel(self, workers, progress_callback=None) -> dict:
        num_chunks = min(workers * 8, self.iterations)
        base_size = self.iterations // num_chunks
        remainder = self.iterations % num_chunks

        rng = np.random.RandomState(self.seed)
        chunk_seeds = [int(rng.randint(0, 2**31)) for _ in range(num_chunks)]

        chunk_args = []
        for i in range(num_chunks):
            n = base_size + (1 if i < remainder else 0)
            if n > 0:
                chunk_args.append((n, chunk_seeds[i]))

        partial_results = []
        completed_iters = 0

        pool = Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.bracket_sim, self.evaluator),
        )
        self._pool = pool
        try:
            for result in pool.imap_unordered(_run_chunk, chunk_args):
                if self.cancelled:
                    break
                partial_results.append(result)
                completed_iters += result["iterations"]
                if progress_callback:
                    progress_callback(completed_iters, self.iterations)
        finally:
            if self.cancelled:
                pool.terminate()
            else:
                pool.close()
            pool.join()
            self._pool = None

        return self._merge_results(partial_results) if partial_results else {
            "iterations": 0,
            "team_round_counts": {},
            "championship_wins": {},
            "final_four_appearances": {},
            "final_four_combos": {},
            "championship_matchups": {},
            "matchup_meet_counts": {},
            "matchup_win_counts": {},
        }

    @staticmethod
    def _merge_results(partial_results: list[dict]) -> dict:
        team_round_counts = defaultdict(lambda: defaultdict(int))
        championship_wins = defaultdict(int)
        final_four_appearances = defaultdict(int)
        final_four_combos = defaultdict(int)
        championship_matchups = defaultdict(int)
        matchup_meet_counts = defaultdict(int)
        matchup_win_counts = defaultdict(lambda: defaultdict(int))

        for r in partial_results:
            for team, wins in r["championship_wins"].items():
                championship_wins[team] += wins
            for team, count in r["final_four_appearances"].items():
                final_four_appearances[team] += count
            for team, rounds in r["team_round_counts"].items():
                for rnd, count in rounds.items():
                    team_round_counts[team][rnd] += count
            for key, count in r["final_four_combos"].items():
                final_four_combos[key] += count
            for key, count in r["championship_matchups"].items():
                championship_matchups[key] += count
            for key, count in r.get("matchup_meet_counts", {}).items():
                matchup_meet_counts[key] += count
            for key, wins in r.get("matchup_win_counts", {}).items():
                for team, count in wins.items():
                    matchup_win_counts[key][team] += count

        total_iterations = sum(r["iterations"] for r in partial_results)

        return {
            "iterations": total_iterations,
            "team_round_counts": {t: dict(r) for t, r in team_round_counts.items()},
            "championship_wins": dict(championship_wins),
            "final_four_appearances": dict(final_four_appearances),
            "final_four_combos": {
                str(k): v for k, v in sorted(final_four_combos.items(), key=lambda x: -x[1])[:10]
            },
            "championship_matchups": {
                str(k): v for k, v in sorted(championship_matchups.items(), key=lambda x: -x[1])[:10]
            },
            "matchup_meet_counts": dict(matchup_meet_counts),
            "matchup_win_counts": {k: dict(v) for k, v in matchup_win_counts.items()},
        }
