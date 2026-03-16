from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.analysis.matchup import MatchupEvaluator
from src.data.models import Team, TeamProfile
from src.simulation.results import SimulationResults


@dataclass
class GamePick:
    round_name: str
    region: str
    team_a: str
    seed_a: int
    team_b: str
    seed_b: int
    pick: str
    pick_seed: int
    confidence: str
    reasoning: str
    is_upset: bool
    win_rate: float
    matchup_details: dict = field(default_factory=dict)


class BracketAdvisor:
    def __init__(
        self,
        evaluator: MatchupEvaluator,
        sim_results: SimulationResults,
        teams_by_name: dict[str, Team],
        profiles: dict[str, TeamProfile],
    ):
        self.evaluator = evaluator
        self.sim_results = sim_results
        self.teams_by_name = teams_by_name
        self.profiles = profiles
        self.adv_probs = sim_results.all_advancement_probabilities()

    def _confidence_label(self, win_rate: float) -> str:
        if win_rate >= 85:
            return "Lock"
        if win_rate >= 65:
            return "Strong"
        if win_rate >= 55:
            return "Lean"
        return "Toss-up"

    def _build_reasoning(self, team_a: Team, team_b: Team, pick: Team, breakdown: dict) -> str:
        parts = []
        loser = team_b if pick.name == team_a.name else team_a

        pa = self.profiles.get(pick.name)
        pl = self.profiles.get(loser.name)

        if pa and pa.strengths:
            top_strength = pa.strengths[0]
            parts.append(f"{pick.name}'s {top_strength.lower()}")

        if pl and pl.weaknesses:
            top_weakness = pl.weaknesses[0]
            parts.append(f"{loser.name}'s {top_weakness.lower()}")

        if pick.coach.name and loser.coach.name:
            if pick.coach.tourney_appearances > loser.coach.tourney_appearances + 5:
                parts.append(f"{pick.coach.name} has far more tournament experience than {loser.coach.name}")

        eff_diff = pick.stats.net_efficiency - loser.stats.net_efficiency
        if abs(eff_diff) > 10:
            parts.append(f"significant efficiency edge (+{eff_diff:.1f})")

        if not parts:
            parts.append(f"{pick.name} is the stronger team overall")

        return "; ".join(parts[:3])

    def generate_bracket(self, consensus: dict) -> list[GamePick]:
        picks = []

        for ff_game in consensus.get("first_four", []):
            ta_name = ff_game["team_a"]["name"]
            tb_name = ff_game["team_b"]["name"]
            winner_name = ff_game["winner"]
            ta = self.teams_by_name.get(ta_name)
            tb = self.teams_by_name.get(tb_name)
            if not ta or not tb:
                continue
            pick_team = ta if winner_name == ta_name else tb
            loser_team = tb if winner_name == ta_name else ta

            breakdown = self.evaluator.get_matchup_breakdown(ta, tb)
            prob = ff_game["prob"]
            is_upset = pick_team.seed > loser_team.seed

            sim_rate = self._get_sim_win_rate(ta_name, tb_name, "Round of 64")
            effective_rate = (prob + sim_rate) / 2 if sim_rate > 0 else prob

            picks.append(GamePick(
                round_name="First Four",
                region=ff_game.get("region", ""),
                team_a=ta_name,
                seed_a=ta.seed,
                team_b=tb_name,
                seed_b=tb.seed,
                pick=winner_name,
                pick_seed=pick_team.seed,
                confidence=self._confidence_label(effective_rate),
                reasoning=self._build_reasoning(ta, tb, pick_team, breakdown),
                is_upset=is_upset,
                win_rate=round(effective_rate, 1),
                matchup_details=breakdown,
            ))

        for region, rounds in consensus.get("regions", {}).items():
            for round_data in rounds:
                rname = round_data["round"]
                for game in round_data["games"]:
                    ta_name = game["team_a"]["name"]
                    tb_name = game["team_b"]["name"]
                    winner_name = game["winner"]
                    ta = self.teams_by_name.get(ta_name)
                    tb = self.teams_by_name.get(tb_name)
                    if not ta or not tb:
                        continue
                    pick_team = ta if winner_name == ta_name else tb
                    loser_team = tb if winner_name == ta_name else ta

                    breakdown = self.evaluator.get_matchup_breakdown(ta, tb)
                    prob = game["prob"]
                    is_upset = pick_team.seed > loser_team.seed

                    sim_rate = self._get_sim_win_rate(ta_name, tb_name, rname)
                    effective_rate = (prob + sim_rate) / 2 if sim_rate > 0 else prob

                    picks.append(GamePick(
                        round_name=rname,
                        region=region,
                        team_a=ta_name,
                        seed_a=ta.seed,
                        team_b=tb_name,
                        seed_b=tb.seed,
                        pick=winner_name,
                        pick_seed=pick_team.seed,
                        confidence=self._confidence_label(effective_rate),
                        reasoning=self._build_reasoning(ta, tb, pick_team, breakdown),
                        is_upset=is_upset,
                        win_rate=round(effective_rate, 1),
                        matchup_details=breakdown,
                    ))

        for ff_game in consensus.get("final_four", []):
            ta_name = ff_game["team_a"]["name"]
            tb_name = ff_game["team_b"]["name"]
            winner_name = ff_game["winner"]
            ta = self.teams_by_name.get(ta_name)
            tb = self.teams_by_name.get(tb_name)
            if not ta or not tb:
                continue
            pick_team = ta if winner_name == ta_name else tb
            loser_team = tb if winner_name == ta_name else ta

            breakdown = self.evaluator.get_matchup_breakdown(ta, tb)
            prob = ff_game["prob"]
            is_upset = pick_team.seed > loser_team.seed

            sim_rate = self._get_sim_win_rate(ta_name, tb_name, "Final Four")
            effective_rate = (prob + sim_rate) / 2 if sim_rate > 0 else prob

            picks.append(GamePick(
                round_name="Final Four",
                region="",
                team_a=ta_name,
                seed_a=ta.seed,
                team_b=tb_name,
                seed_b=tb.seed,
                pick=winner_name,
                pick_seed=pick_team.seed,
                confidence=self._confidence_label(effective_rate),
                reasoning=self._build_reasoning(ta, tb, pick_team, breakdown),
                is_upset=is_upset,
                win_rate=round(effective_rate, 1),
                matchup_details=breakdown,
            ))

        champ = consensus.get("championship")
        if champ:
            ta_name = champ["team_a"]["name"]
            tb_name = champ["team_b"]["name"]
            winner_name = champ["winner"]
            ta = self.teams_by_name.get(ta_name)
            tb = self.teams_by_name.get(tb_name)
            if ta and tb:
                pick_team = ta if winner_name == ta_name else tb
                loser_team = tb if winner_name == ta_name else ta
                breakdown = self.evaluator.get_matchup_breakdown(ta, tb)
                prob = champ["prob"]
                is_upset = pick_team.seed > loser_team.seed

                champ_probs = self.sim_results.championship_probabilities()
                pick_champ_rate = champ_probs.get(winner_name, prob)
                effective_rate = (prob + pick_champ_rate) / 2

                picks.append(GamePick(
                    round_name="Championship",
                    region="",
                    team_a=ta_name,
                    seed_a=ta.seed,
                    team_b=tb_name,
                    seed_b=tb.seed,
                    pick=winner_name,
                    pick_seed=pick_team.seed,
                    confidence=self._confidence_label(effective_rate),
                    reasoning=self._build_reasoning(ta, tb, pick_team, breakdown),
                    is_upset=is_upset,
                    win_rate=round(effective_rate, 1),
                    matchup_details=breakdown,
                ))

        return picks

    def _get_sim_win_rate(self, team_a: str, team_b: str, round_name: str) -> float:
        a_adv = self.adv_probs.get(team_a, {}).get(round_name, 0)
        b_adv = self.adv_probs.get(team_b, {}).get(round_name, 0)
        total = a_adv + b_adv
        if total == 0:
            return 0.0
        return a_adv / total * 100

    def get_smart_upsets(self, picks: list[GamePick], max_upsets: int = 5) -> list[GamePick]:
        upsets = [p for p in picks if p.is_upset]
        upsets.sort(key=lambda p: -p.win_rate)
        return upsets[:max_upsets]

    def get_champion(self, picks: list[GamePick]) -> GamePick | None:
        for p in picks:
            if p.round_name == "Championship":
                return p
        return None

    def get_final_four(self, picks: list[GamePick]) -> list[str]:
        ff_picks = [p for p in picks if p.round_name == "Final Four"]
        teams = set()
        for p in ff_picks:
            teams.add(p.team_a)
            teams.add(p.team_b)
        return sorted(teams)

    def to_dict(self, picks: list[GamePick]) -> dict:
        champ = self.get_champion(picks)
        ff = self.get_final_four(picks)
        smart_upsets = self.get_smart_upsets(picks)

        rounds: dict[str, list] = {}
        for p in picks:
            rounds.setdefault(p.round_name, []).append({
                "region": p.region,
                "team_a": p.team_a,
                "seed_a": p.seed_a,
                "team_b": p.team_b,
                "seed_b": p.seed_b,
                "pick": p.pick,
                "pick_seed": p.pick_seed,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
                "is_upset": p.is_upset,
                "win_rate": p.win_rate,
            })

        return {
            "champion": {
                "team": champ.pick if champ else "",
                "seed": champ.pick_seed if champ else 0,
                "confidence": champ.confidence if champ else "",
                "reasoning": champ.reasoning if champ else "",
            },
            "final_four": ff,
            "smart_upsets": [
                {
                    "round": u.round_name,
                    "region": u.region,
                    "pick": u.pick,
                    "seed": u.pick_seed,
                    "over": u.team_a if u.pick == u.team_b else u.team_b,
                    "confidence": u.confidence,
                    "reasoning": u.reasoning,
                    "win_rate": u.win_rate,
                }
                for u in smart_upsets
            ],
            "rounds": rounds,
            "total_picks": len(picks),
            "total_upsets": sum(1 for p in picks if p.is_upset),
        }
