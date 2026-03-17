import json
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress

import config
from src.data.team_stats import load_teams, load_teams_by_name
from src.analysis.historical import HistoricalAnalyzer
from src.analysis.profiler import build_all_profiles, build_profile
from src.analysis.matchup import MatchupEvaluator
from src.analysis.advisor import BracketAdvisor
from src.simulation.bracket import BracketSimulator, load_bracket
from src.simulation.engine import MonteCarloEngine
from src.simulation.results import SimulationResults

console = Console()


def _build_engine(year: int, iterations: int, seed: int | None = None, weights: dict | None = None):
    teams = load_teams(year)
    teams_by_name = {t.name: t for t in teams}
    bracket = load_bracket(year)
    historical = HistoricalAnalyzer()
    profiles = build_all_profiles(teams, historical)
    evaluator = MatchupEvaluator(historical, profiles, **(weights or {}))
    bracket_sim = BracketSimulator(bracket, teams_by_name)
    engine = MonteCarloEngine(bracket_sim, evaluator, iterations=iterations, seed=seed)
    return engine, teams, teams_by_name, profiles


@click.group()
def cli():
    """NCAA March Madness Bracket Prediction Engine"""
    pass


@cli.command()
@click.option("--year", default=config.CURRENT_YEAR, help="Tournament year")
@click.option("--iterations", "-n", default=config.DEFAULT_ITERATIONS, help="Number of simulations")
@click.option("--seed", "-s", default=None, type=int, help="Random seed for reproducibility")
def simulate(year, iterations, seed):
    """Run Monte Carlo bracket simulation."""
    console.print(f"\n[bold cyan]NCAA March Madness {year} - Monte Carlo Simulation[/bold cyan]")
    console.print(f"Running {iterations:,} tournament simulations...\n")

    engine, teams, teams_by_name, profiles = _build_engine(year, iterations, seed)

    with Progress(console=console) as progress:
        task = progress.add_task("Simulating tournaments...", total=iterations)

        def update_progress(current, total):
            progress.update(task, completed=current)

        raw = engine.run(progress_callback=update_progress)
        progress.update(task, completed=iterations)

    results = SimulationResults(raw)
    summary = results.to_summary_dict()

    champ = summary["predicted_champion"]
    console.print(Panel(
        f"[bold yellow]{champ['team']}[/bold yellow] — {champ['probability']}% of simulations",
        title="[bold]Predicted Champion[/bold]",
        border_style="green",
    ))

    table = Table(title="Championship Probabilities (Top 10)")
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right")
    for team_name, pct in summary["championship_probabilities"].items():
        table.add_row(team_name, f"{pct:.1f}%")
    console.print(table)

    ff_table = Table(title="Final Four Probabilities (Top 8)")
    ff_table.add_column("Team", style="bold")
    ff_table.add_column("Final Four %", justify="right")
    for team_name, pct in list(summary["final_four_probabilities"].items())[:8]:
        ff_table.add_row(team_name, f"{pct:.1f}%")
    console.print(ff_table)

    tiers = summary["tiers"]
    _print_tier(tiers, "locks", "Locks (90%+ to advance R64)", "green")
    _print_tier(tiers, "upsets_to_watch", "Upsets to Watch", "red")

    console.print(f"\n[dim]Full results available via the web UI: python -m src.web.app[/dim]\n")


def _print_tier(tiers, key, title, color):
    items = tiers.get(key, [])
    if not items:
        return
    table = Table(title=title)
    table.add_column("Team", style="bold")
    table.add_column("R64 Win %", justify="right", style=color)
    for entry in items[:12]:
        table.add_row(entry["team"], f"{entry['r64_prob']:.1f}%")
    console.print(table)


@cli.command()
@click.argument("team_name")
@click.option("--year", default=config.CURRENT_YEAR, help="Tournament year")
def profile(team_name, year):
    """View a team's profile and analytics."""
    teams = load_teams(year)
    teams_by_name = {t.name: t for t in teams}

    team = teams_by_name.get(team_name)
    if not team:
        close = [t.name for t in teams if team_name.lower() in t.name.lower()]
        if close:
            console.print(f"[yellow]Did you mean: {', '.join(close)}?[/yellow]")
        else:
            console.print(f"[red]Team '{team_name}' not found.[/red]")
        return

    prof = build_profile(team, teams)
    s = team.stats

    c = team.coach
    coach_line = ""
    if c.name:
        coach_line = f"\nCoach: {c.name} ({c.career_wins}-{c.career_losses}, .{int(c.career_win_pct*1000):03d}) | {c.tourney_appearances} tourney apps, {c.final_fours} FF, {c.championships} titles"
    console.print(Panel(
        f"[bold]{team.name}[/bold] ({team.conference})\n"
        f"Seed: {team.seed} | Region: {team.region} | Record: {s.wins}-{s.losses}"
        f"{coach_line}",
        title="Team Profile",
        border_style="cyan",
    ))

    stats_table = Table(title="Key Metrics")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Adj. Off. Efficiency", f"{s.adj_offensive_efficiency:.1f}")
    stats_table.add_row("Adj. Def. Efficiency", f"{s.adj_defensive_efficiency:.1f}")
    stats_table.add_row("Net Efficiency", f"{s.net_efficiency:.1f}")
    stats_table.add_row("Tempo", f"{s.tempo:.1f}")
    stats_table.add_row("SOS", f"{s.strength_of_schedule:.1f}")
    stats_table.add_row("3PT %", f"{s.three_pt_pct:.1f}%")
    stats_table.add_row("FT %", f"{s.ft_pct:.1f}%")
    stats_table.add_row("TO Rate (Off)", f"{s.turnover_rate_off:.1f}%")
    stats_table.add_row("Off Reb Rate", f"{s.off_rebound_rate:.1f}%")
    stats_table.add_row("Last 10", f"{s.last10_wins}-{s.last10_losses}")
    stats_table.add_row("vs Top 25", s.record_vs_top25)
    console.print(stats_table)

    scores_table = Table(title="Composite Scores")
    scores_table.add_column("Category", style="bold")
    scores_table.add_column("Score (0-100)", justify="right")
    scores_table.add_row("Offensive", f"{prof.offensive_score:.1f}")
    scores_table.add_row("Defensive", f"{prof.defensive_score:.1f}")
    scores_table.add_row("Intangible", f"{prof.intangible_score:.1f}")
    scores_table.add_row("[bold]Composite[/bold]", f"[bold]{prof.composite_score:.1f}[/bold]")
    console.print(scores_table)

    if prof.strengths:
        console.print(Panel(
            "\n".join(f"[green]+[/green] {s}" for s in prof.strengths),
            title="Strengths",
            border_style="green",
        ))
    if prof.weaknesses:
        console.print(Panel(
            "\n".join(f"[red]-[/red] {w}" for w in prof.weaknesses),
            title="Weaknesses",
            border_style="red",
        ))

    if team.players:
        p_table = Table(title="Roster — Per Game Stats")
        p_table.add_column("Player", style="bold")
        p_table.add_column("Yr", justify="center")
        p_table.add_column("Pos", justify="center")
        p_table.add_column("G", justify="right")
        p_table.add_column("MPG", justify="right")
        p_table.add_column("PPG", justify="right")
        p_table.add_column("RPG", justify="right")
        p_table.add_column("APG", justify="right")
        p_table.add_column("FG%", justify="right")
        p_table.add_column("3P%", justify="right")
        p_table.add_column("FT%", justify="right")
        for p in sorted(team.players, key=lambda x: -x.ppg):
            if p.mpg < 5:
                continue
            p_table.add_row(
                p.name, p.year, p.position,
                str(p.games), f"{p.mpg:.1f}", f"{p.ppg:.1f}",
                f"{p.rpg:.1f}", f"{p.apg:.1f}",
                f"{p.fg_pct:.1f}", f"{p.three_pct:.1f}", f"{p.ft_pct:.1f}",
            )
        console.print(p_table)

    if team.schedule:
        last_games = team.schedule[-10:]
        s_table = Table(title="Last 10 Games")
        s_table.add_column("Date", style="dim")
        s_table.add_column("", justify="center")
        s_table.add_column("Opponent")
        s_table.add_column("Result", justify="center")
        s_table.add_column("Score", justify="right")
        for g in last_games:
            color = "green" if g.result == "W" else "red"
            loc = "@" if g.location == "@" else "vs" if g.location == "H" else "N"
            s_table.add_row(
                g.date, loc, g.opponent,
                f"[{color}]{g.result}[/{color}]",
                f"{g.team_score}-{g.opp_score}",
            )
        console.print(s_table)


@cli.command()
@click.option("--seed", "-s", type=int, default=None, help="Show data for a specific seed")
def history(seed):
    """View historical tournament trends."""
    analyzer = HistoricalAnalyzer()

    if seed:
        console.print(f"\n[bold cyan]Historical Performance: {seed}-Seeds[/bold cyan]\n")
        rates = analyzer.seed_advancement_rates()
        if seed in rates:
            table = Table(title=f"{seed}-Seed Historical Advancement Rates")
            table.add_column("Round", style="bold")
            table.add_column("Win %", justify="right")
            for round_name, rate in rates[seed].items():
                table.add_row(round_name, f"{rate * 100:.1f}%")
            console.print(table)
    else:
        console.print("\n[bold cyan]Historical Seed Matchup Win Rates (Round of 64)[/bold cyan]\n")
        table = Table(title="Higher Seed Win Rate by Matchup")
        table.add_column("Matchup", style="bold")
        table.add_column("Higher Seed Wins", justify="right")
        for (h, l), rate in sorted(analyzer.get_seed_win_rate_lookup().items()):
            if l == 17 - h:
                table.add_row(f"{h} vs {l}", f"{rate * 100:.1f}%")
        console.print(table)

        upset_table = Table(title="Upset Rates by Round")
        upset_table.add_column("Round", style="bold")
        upset_table.add_column("Upset Rate", justify="right")
        for round_name, rate in analyzer.upset_rate_by_round().items():
            upset_table.add_row(round_name, f"{rate * 100:.1f}%")
        console.print(upset_table)


@cli.command()
@click.option("--year", default=config.CURRENT_YEAR, help="Tournament year")
@click.option("--iterations", "-n", default=config.DEFAULT_ITERATIONS, help="Number of simulations")
@click.option("--seed", "-s", default=None, type=int, help="Random seed")
def bracket(year, iterations, seed):
    """Output the predicted bracket based on simulation."""
    console.print(f"\n[bold cyan]Predicted Bracket - {year}[/bold cyan]")
    console.print(f"Based on {iterations:,} simulations...\n")

    engine, teams, teams_by_name, profiles = _build_engine(year, iterations, seed)

    with Progress(console=console) as progress:
        task = progress.add_task("Simulating...", total=iterations)

        def update_progress(current, total):
            progress.update(task, completed=current)

        raw = engine.run(progress_callback=update_progress)
        progress.update(task, completed=iterations)

    results = SimulationResults(raw)
    predicted = results.predicted_bracket()

    for round_name in config.ROUNDS:
        if round_name in predicted:
            table = Table(title=round_name)
            table.add_column("Team", style="bold")
            table.add_column("Advance %", justify="right")
            for entry in predicted[round_name]:
                color = "green" if entry["probability"] > 75 else "yellow" if entry["probability"] > 55 else "white"
                table.add_row(f"[{color}]{entry['team']}[/{color}]", f"{entry['probability']:.1f}%")
            console.print(table)


@cli.command()
@click.argument("team_a_name")
@click.argument("team_b_name")
@click.option("--year", default=config.CURRENT_YEAR, help="Tournament year")
def matchup(team_a_name, team_b_name, year):
    """Analyze a head-to-head matchup between two teams."""
    teams = load_teams(year)
    teams_by_name = {t.name: t for t in teams}
    historical = HistoricalAnalyzer()
    profiles = build_all_profiles(teams, historical)
    evaluator = MatchupEvaluator(historical, profiles)

    team_a = teams_by_name.get(team_a_name)
    team_b = teams_by_name.get(team_b_name)
    if not team_a or not team_b:
        console.print("[red]One or both teams not found.[/red]")
        return

    breakdown = evaluator.get_matchup_breakdown(team_a, team_b)

    console.print(Panel(
        f"[bold]{team_a.name}[/bold] ({team_a.seed}-seed) vs "
        f"[bold]{team_b.name}[/bold] ({team_b.seed}-seed)",
        title="Matchup Analysis",
        border_style="cyan",
    ))

    table = Table(title="Win Probability")
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right")
    table.add_row(team_a.name, f"{breakdown['win_prob_a'] * 100:.1f}%")
    table.add_row(team_b.name, f"{breakdown['win_prob_b'] * 100:.1f}%")
    console.print(table)

    comp_table = Table(title="Component Breakdown")
    comp_table.add_column("Factor", style="bold")
    comp_table.add_column("Weight", justify="right")
    comp_table.add_column(f"{team_a.name} Prob", justify="right")
    for name, data in breakdown["components"].items():
        comp_table.add_row(
            name.title(),
            f"{data['weight']:.0%}",
            f"{data['team_a_prob'] * 100:.1f}%",
        )
    console.print(comp_table)


@cli.command()
@click.option("--year", default=config.CURRENT_YEAR, help="Tournament year")
@click.option("--iterations", "-n", default=config.DEFAULT_ITERATIONS, help="Number of simulations")
@click.option("--seed", "-s", default=None, type=int, help="Random seed")
@click.option("--w-hist", default=None, type=float, help="Historical weight (0-100)")
@click.option("--w-eff", default=None, type=float, help="Efficiency weight (0-100)")
@click.option("--w-prof", default=None, type=float, help="Profile weight (0-100)")
@click.option("--w-match", default=None, type=float, help="Matchup weight (0-100)")
@click.option("--chaos", is_flag=True, help="Bias picks toward underdogs (chaos bracket)")
@click.option("--chaos-strength", default=50, type=int, help="Chaos strength 0-100 (default 50)")
def advisor(year, iterations, seed, w_hist, w_eff, w_prof, w_match, chaos, chaos_strength):
    """Generate a complete bracket recommendation with confidence levels and upset picks."""
    weights = {}
    if w_hist is not None:
        weights["weight_historical"] = w_hist / 100.0
    if w_eff is not None:
        weights["weight_efficiency"] = w_eff / 100.0
    if w_prof is not None:
        weights["weight_profile"] = w_prof / 100.0
    if w_match is not None:
        weights["weight_matchup"] = w_match / 100.0

    console.print(f"\n[bold cyan]NCAA March Madness {year} - Bracket Advisor[/bold cyan]")
    w = weights if weights else {
        "weight_historical": config.WEIGHT_HISTORICAL,
        "weight_efficiency": config.WEIGHT_EFFICIENCY,
        "weight_profile": config.WEIGHT_PROFILE,
        "weight_matchup": config.WEIGHT_MATCHUP,
    }
    console.print(f"Weights: Historical={w.get('weight_historical', config.WEIGHT_HISTORICAL):.0%}  "
                  f"Efficiency={w.get('weight_efficiency', config.WEIGHT_EFFICIENCY):.0%}  "
                  f"Profile={w.get('weight_profile', config.WEIGHT_PROFILE):.0%}  "
                  f"Matchup={w.get('weight_matchup', config.WEIGHT_MATCHUP):.0%}")
    console.print(f"Running {iterations:,} simulations to build your bracket...\n")

    engine, teams, teams_by_name, profiles = _build_engine(year, iterations, seed, weights or None)

    upset_bias = 0.0
    if chaos:
        upset_bias = max(0, min(100, chaos_strength)) / 100.0
        console.print(f"[yellow]Chaos mode: upset bias = {upset_bias:.0%}[/yellow]\n")

    with Progress(console=console) as progress:
        task = progress.add_task("Simulating tournaments...", total=iterations)

        def update_progress(current, total):
            progress.update(task, completed=current)

        raw = engine.run(progress_callback=update_progress)
        progress.update(task, completed=iterations)

    sim_results = SimulationResults(raw)
    consensus = engine.predict_consensus_bracket(upset_bias=upset_bias)
    adv = BracketAdvisor(engine.evaluator, sim_results, teams_by_name, profiles)
    picks = adv.generate_bracket(consensus)
    result = adv.to_dict(picks)

    champ = result["champion"]
    console.print(Panel(
        f"[bold yellow]{champ['team']}[/bold yellow] ({champ['seed']}-seed)\n"
        f"Confidence: {champ['confidence']}\n"
        f"{champ['reasoning']}",
        title="[bold]Predicted Champion[/bold]",
        border_style="green",
    ))

    ff = result["final_four"]
    console.print(Panel(
        "  |  ".join(f"[bold]{t}[/bold]" for t in ff),
        title="Final Four",
        border_style="cyan",
    ))

    if result["smart_upsets"]:
        upset_table = Table(title="Smart Upset Picks")
        upset_table.add_column("Round", style="dim")
        upset_table.add_column("Pick", style="bold red")
        upset_table.add_column("Over", style="bold")
        upset_table.add_column("Conf", justify="center")
        upset_table.add_column("Rate", justify="right")
        upset_table.add_column("Reasoning")
        for u in result["smart_upsets"]:
            upset_table.add_row(
                u["round"], f"({u['seed']}) {u['pick']}", u["over"],
                u["confidence"], f"{u['win_rate']:.0f}%", u["reasoning"][:60],
            )
        console.print(upset_table)

    round_order = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    for rname in round_order:
        games = result["rounds"].get(rname, [])
        if not games:
            continue
        table = Table(title=rname)
        table.add_column("Matchup", style="bold", min_width=30)
        table.add_column("Pick", style="bold green")
        table.add_column("Conf", justify="center")
        table.add_column("Rate", justify="right")
        table.add_column("Upset", justify="center")
        for g in games:
            matchup_str = f"({g['seed_a']}) {g['team_a']} vs ({g['seed_b']}) {g['team_b']}"
            upset_marker = "[red]YES[/red]" if g["is_upset"] else ""
            conf_color = {"Lock": "green", "Strong": "cyan", "Lean": "yellow", "Toss-up": "red"}.get(g["confidence"], "white")
            table.add_row(
                matchup_str,
                f"({g['pick_seed']}) {g['pick']}",
                f"[{conf_color}]{g['confidence']}[/{conf_color}]",
                f"{g['win_rate']:.0f}%",
                upset_marker,
            )
        console.print(table)

    total_upsets = result["total_upsets"]
    console.print(f"\n[dim]Total picks: {result['total_picks']} | Upsets picked: {total_upsets}[/dim]\n")


if __name__ == "__main__":
    cli()
