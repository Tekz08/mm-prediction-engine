from __future__ import annotations
from pydantic import BaseModel, Field


class TeamStats(BaseModel):
    adj_offensive_efficiency: float = 0.0
    adj_defensive_efficiency: float = 0.0
    tempo: float = 0.0
    strength_of_schedule: float = 0.0
    record_vs_top25: str = "0-0"
    turnover_rate_off: float = 0.0
    turnover_rate_def: float = 0.0
    off_rebound_rate: float = 0.0
    def_rebound_rate: float = 0.0
    ft_rate: float = 0.0
    ft_pct: float = 0.0
    three_pt_pct: float = 0.0
    three_pt_def_pct: float = 0.0
    wins: int = 0
    losses: int = 0
    last10_wins: int = 0
    last10_losses: int = 0
    tournament_experience: float = 0.0

    @property
    def net_efficiency(self) -> float:
        return self.adj_offensive_efficiency - self.adj_defensive_efficiency


class PlayerStats(BaseModel):
    name: str
    position: str = ""
    year: str = ""
    height: str = ""
    games: int = 0
    games_started: int = 0
    mpg: float = 0.0
    ppg: float = 0.0
    rpg: float = 0.0
    apg: float = 0.0
    spg: float = 0.0
    bpg: float = 0.0
    topg: float = 0.0
    fg_pct: float = 0.0
    three_pct: float = 0.0
    ft_pct: float = 0.0
    efg_pct: float = 0.0
    orpg: float = 0.0
    drpg: float = 0.0


class ScheduleGame(BaseModel):
    date: str
    location: str = ""
    opponent: str = ""
    result: str = ""
    team_score: int = 0
    opp_score: int = 0
    is_ranked_opp: bool = False


class CoachInfo(BaseModel):
    name: str = ""
    career_wins: int = 0
    career_losses: int = 0
    career_win_pct: float = 0.0
    years_coaching: int = 0
    tourney_appearances: int = 0
    tourney_wins: int = 0
    tourney_losses: int = 0
    final_fours: int = 0
    championships: int = 0


class Team(BaseModel):
    name: str
    seed: int
    region: str
    conference: str = ""
    stats: TeamStats = Field(default_factory=TeamStats)
    coach: CoachInfo = Field(default_factory=CoachInfo)
    players: list[PlayerStats] = Field(default_factory=list)
    schedule: list[ScheduleGame] = Field(default_factory=list)


class TeamProfile(BaseModel):
    team: Team
    offensive_score: float = 0.0
    defensive_score: float = 0.0
    intangible_score: float = 0.0
    composite_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class MatchupResult(BaseModel):
    team_a: str
    team_b: str
    win_prob_a: float
    historical_component: float = 0.0
    efficiency_component: float = 0.0
    profile_component: float = 0.0
    matchup_component: float = 0.0


class GameResult(BaseModel):
    round_name: str
    region: str
    winner: str
    loser: str
    winner_seed: int
    loser_seed: int
    winner_score: int = 0
    loser_score: int = 0


class BracketEntry(BaseModel):
    seed: int
    team: str
    region: str
    play_in: str | None = None


class SimulationSummary(BaseModel):
    iterations: int
    team_round_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    championship_wins: dict[str, int] = Field(default_factory=dict)
    most_common_final_four: list[str] = Field(default_factory=list)
    most_common_champion: str = ""


class HistoricalMatchup(BaseModel):
    year: int
    round_name: str
    region: str
    seed_a: int
    team_a: str
    score_a: int
    seed_b: int
    team_b: str
    score_b: int
