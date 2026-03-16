from __future__ import annotations

import numpy as np
from scipy.special import expit

import config
from src.analysis.historical import HistoricalAnalyzer
from src.data.models import MatchupResult, Team, TeamProfile


SR_SCHEDULE_NAME_MAP = {
    "Connecticut": "UConn",
    "St. John's (NY)": "St. John's",
    "Texas Christian": "TCU",
    "Central Florida": "UCF",
    "Southern Methodist": "SMU",
    "Brigham Young": "BYU",
    "Miami (FL)": "Miami FL",
    "North Carolina State": "NC State",
    "Maryland-Baltimore County": "UMBC",
    "McNeese State": "McNeese",
    "Prairie View": "Prairie View A&M",
    "Pennsylvania": "Penn",
    "Queens (NC)": "Queens",
    "California Baptist": "Cal Baptist",
    "Long Island University": "Long Island",
    "Saint Mary's (CA)": "Saint Mary's",
    "Virginia Commonwealth": "VCU",
    "Miami (OH)": "Miami OH",
}


def _normalize_opponent(name: str) -> str:
    return SR_SCHEDULE_NAME_MAP.get(name, name)


class MatchupEvaluator:
    def __init__(
        self,
        historical: HistoricalAnalyzer,
        profiles: dict[str, TeamProfile],
        weight_historical: float | None = None,
        weight_efficiency: float | None = None,
        weight_profile: float | None = None,
        weight_matchup: float | None = None,
    ):
        self.historical = historical
        self.profiles = profiles
        self.w_hist = weight_historical or config.WEIGHT_HISTORICAL
        self.w_eff = weight_efficiency or config.WEIGHT_EFFICIENCY
        self.w_prof = weight_profile or config.WEIGHT_PROFILE
        self.w_match = weight_matchup or config.WEIGHT_MATCHUP

    def evaluate(self, team_a: Team, team_b: Team, round_name: str = "Round of 64") -> MatchupResult:
        hist_prob = self._historical_component(team_a, team_b)
        eff_prob = self._efficiency_component(team_a, team_b)
        prof_prob = self._profile_component(team_a, team_b)
        match_prob = self._matchup_component(team_a, team_b)

        raw = (
            self.w_hist * hist_prob
            + self.w_eff * eff_prob
            + self.w_prof * prof_prob
            + self.w_match * match_prob
        )
        raw += self._coach_nudge(team_a, team_b)
        win_prob_a = float(np.clip(raw, 0.02, 0.98))

        return MatchupResult(
            team_a=team_a.name,
            team_b=team_b.name,
            win_prob_a=round(win_prob_a, 4),
            historical_component=round(hist_prob, 4),
            efficiency_component=round(eff_prob, 4),
            profile_component=round(prof_prob, 4),
            matchup_component=round(match_prob, 4),
        )

    def win_probability(self, team_a: Team, team_b: Team, round_name: str = "Round of 64") -> float:
        return self.evaluate(team_a, team_b, round_name).win_prob_a

    def _historical_component(self, team_a: Team, team_b: Team) -> float:
        return self.historical.seed_matchup_win_rate(team_a.seed, team_b.seed)

    def _efficiency_component(self, team_a: Team, team_b: Team) -> float:
        sr_diff = team_a.stats.sos_adjusted_net - team_b.stats.sos_adjusted_net
        if team_a.stats.bpi != 0 and team_b.stats.bpi != 0:
            bpi_diff = team_a.stats.bpi - team_b.stats.bpi
            blended_diff = sr_diff * 0.6 + bpi_diff * 0.4
        else:
            blended_diff = sr_diff
        return float(expit(blended_diff / config.LOGISTIC_SCALE))

    def _profile_component(self, team_a: Team, team_b: Team) -> float:
        prof_a = self.profiles.get(team_a.name)
        prof_b = self.profiles.get(team_b.name)
        if not prof_a or not prof_b:
            return 0.5
        diff = prof_a.composite_score - prof_b.composite_score
        return float(expit(diff / 25.0))

    def _matchup_component(self, team_a: Team, team_b: Team) -> float:
        adjustments = []

        adjustments.append(self._perimeter_battle(team_a, team_b))
        adjustments.append(self._turnover_battle(team_a, team_b))
        adjustments.append(self._rebounding_battle(team_a, team_b))
        adjustments.append(self._pace_mismatch(team_a, team_b))
        adjustments.append(self._free_throw_edge(team_a, team_b))
        adjustments.append(self._head_to_head(team_a, team_b))
        adjustments.append(self._common_opponents(team_a, team_b))
        adjustments.append(self._momentum(team_a, team_b))

        total = sum(adjustments)
        return float(expit(total / 3.0))

    def _perimeter_battle(self, team_a: Team, team_b: Team) -> float:
        sa, sb = team_a.stats, team_b.stats
        if sa.three_pt_pct == 0 or sb.three_pt_pct == 0:
            return 0.0
        a_expected = (sa.three_pt_pct + sb.three_pt_def_pct) / 2 if sb.three_pt_def_pct > 0 else sa.three_pt_pct
        b_expected = (sb.three_pt_pct + sa.three_pt_def_pct) / 2 if sa.three_pt_def_pct > 0 else sb.three_pt_pct
        diff = a_expected - b_expected
        return np.clip(diff * 0.3, -2.0, 2.0)

    def _turnover_battle(self, team_a: Team, team_b: Team) -> float:
        sa, sb = team_a.stats, team_b.stats
        if sa.turnover_rate_off == 0 and sb.turnover_rate_off == 0:
            return 0.0
        a_exposure = sa.turnover_rate_off + sb.turnover_rate_def
        b_exposure = sb.turnover_rate_off + sa.turnover_rate_def
        return np.clip((b_exposure - a_exposure) * 0.1, -2.0, 2.0)

    def _rebounding_battle(self, team_a: Team, team_b: Team) -> float:
        sa, sb = team_a.stats, team_b.stats
        a_edge = sa.off_rebound_rate - sb.def_rebound_rate
        b_edge = sb.off_rebound_rate - sa.def_rebound_rate
        return np.clip((a_edge - b_edge) * 0.08, -1.5, 1.5)

    def _pace_mismatch(self, team_a: Team, team_b: Team) -> float:
        sa, sb = team_a.stats, team_b.stats
        if sa.tempo == 0 or sb.tempo == 0:
            return 0.0
        tempo_diff = abs(sa.tempo - sb.tempo)
        if tempo_diff < 3:
            return 0.0
        a_sos = sa.strength_of_schedule
        b_sos = sb.strength_of_schedule
        if a_sos > b_sos:
            return np.clip(tempo_diff * 0.03, 0, 1.0)
        elif b_sos > a_sos:
            return np.clip(-tempo_diff * 0.03, -1.0, 0)
        return 0.0

    def _free_throw_edge(self, team_a: Team, team_b: Team) -> float:
        sa, sb = team_a.stats, team_b.stats
        a_ft_value = sa.ft_rate * sa.ft_pct / 100 if sa.ft_pct > 0 else 0
        b_ft_value = sb.ft_rate * sb.ft_pct / 100 if sb.ft_pct > 0 else 0
        diff = a_ft_value - b_ft_value
        return np.clip(diff * 3.0, -1.0, 1.0)

    def _head_to_head(self, team_a: Team, team_b: Team) -> float:
        margins = []
        for game in team_a.schedule:
            opp = _normalize_opponent(game.opponent)
            if opp == team_b.name and game.team_score > 0:
                margins.append(game.team_score - game.opp_score)
        for game in team_b.schedule:
            opp = _normalize_opponent(game.opponent)
            if opp == team_a.name and game.team_score > 0:
                margins.append(-(game.team_score - game.opp_score))
        if not margins:
            return 0.0
        avg_margin = sum(margins) / len(margins)
        return np.clip(avg_margin * 0.06, -2.5, 2.5)

    def _common_opponents(self, team_a: Team, team_b: Team) -> float:
        a_results: dict[str, list[int]] = {}
        for game in team_a.schedule:
            if game.team_score > 0:
                opp = _normalize_opponent(game.opponent)
                a_results.setdefault(opp, []).append(game.team_score - game.opp_score)
        b_results: dict[str, list[int]] = {}
        for game in team_b.schedule:
            if game.team_score > 0:
                opp = _normalize_opponent(game.opponent)
                b_results.setdefault(opp, []).append(game.team_score - game.opp_score)

        common = set(a_results.keys()) & set(b_results.keys())
        common.discard(team_a.name)
        common.discard(team_b.name)
        if not common:
            return 0.0

        diffs = []
        for opp in common:
            a_avg = sum(a_results[opp]) / len(a_results[opp])
            b_avg = sum(b_results[opp]) / len(b_results[opp])
            diffs.append(a_avg - b_avg)

        avg_diff = sum(diffs) / len(diffs)
        return np.clip(avg_diff * 0.03, -1.5, 1.5)

    def _momentum(self, team_a: Team, team_b: Team) -> float:
        a_mom = self._compute_momentum_score(team_a)
        b_mom = self._compute_momentum_score(team_b)
        diff = a_mom - b_mom
        return np.clip(diff * 0.4, -1.5, 1.5)

    def _compute_momentum_score(self, team: Team) -> float:
        recent = [g for g in team.schedule if g.team_score > 0][-10:]
        if not recent:
            return 0.0
        margins = [g.team_score - g.opp_score for g in recent]
        avg_margin = sum(margins) / len(margins)
        wins = sum(1 for m in margins if m > 0)
        win_rate = wins / len(margins)
        trend = 0.0
        if len(margins) >= 4:
            first_half = margins[:len(margins) // 2]
            second_half = margins[len(margins) // 2:]
            trend = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
        return avg_margin * 0.4 + win_rate * 5.0 + trend * 0.2

    def _coach_nudge(self, team_a: Team, team_b: Team) -> float:
        ca, cb = team_a.coach, team_b.coach
        if not ca.name or not cb.name:
            return 0.0
        tourney_diff = ca.tourney_appearances - cb.tourney_appearances
        ff_diff = ca.final_fours - cb.final_fours
        champ_diff = ca.championships - cb.championships
        nudge = tourney_diff * 0.002 + ff_diff * 0.005 + champ_diff * 0.008
        return float(np.clip(nudge, -0.05, 0.05))

    def get_matchup_breakdown(self, team_a: Team, team_b: Team) -> dict:
        result = self.evaluate(team_a, team_b)
        return {
            "team_a": team_a.name,
            "team_b": team_b.name,
            "win_prob_a": result.win_prob_a,
            "win_prob_b": round(1.0 - result.win_prob_a, 4),
            "components": {
                "historical": {
                    "weight": self.w_hist,
                    "team_a_prob": result.historical_component,
                },
                "efficiency": {
                    "weight": self.w_eff,
                    "team_a_net_eff": round(team_a.stats.sos_adjusted_net, 1),
                    "team_b_net_eff": round(team_b.stats.sos_adjusted_net, 1),
                    "team_a_raw_net": round(team_a.stats.net_efficiency, 1),
                    "team_b_raw_net": round(team_b.stats.net_efficiency, 1),
                    "team_a_sos": team_a.stats.strength_of_schedule,
                    "team_b_sos": team_b.stats.strength_of_schedule,
                    "team_a_bpi": team_a.stats.bpi,
                    "team_b_bpi": team_b.stats.bpi,
                    "team_a_bpi_rank": team_a.stats.bpi_rank,
                    "team_b_bpi_rank": team_b.stats.bpi_rank,
                    "team_a_sor_rank": team_a.stats.sor_rank,
                    "team_b_sor_rank": team_b.stats.sor_rank,
                    "team_a_quality_record": f"{team_a.stats.quality_wins}-{team_a.stats.quality_losses}",
                    "team_b_quality_record": f"{team_b.stats.quality_wins}-{team_b.stats.quality_losses}",
                    "team_a_prob": result.efficiency_component,
                },
                "profile": {
                    "weight": self.w_prof,
                    "team_a_composite": self.profiles.get(team_a.name, TeamProfile(team=team_a)).composite_score,
                    "team_b_composite": self.profiles.get(team_b.name, TeamProfile(team=team_b)).composite_score,
                    "team_a_prob": result.profile_component,
                },
                "matchup": {
                    "weight": self.w_match,
                    "team_a_prob": result.matchup_component,
                },
            },
            "team_a_strengths": self.profiles.get(team_a.name, TeamProfile(team=team_a)).strengths,
            "team_b_strengths": self.profiles.get(team_b.name, TeamProfile(team=team_b)).strengths,
        }
