# NCAA March Madness Bracket Prediction Engine

A data-driven March Madness bracket prediction system that combines historical tournament analytics, team profiling, and Monte Carlo simulation to predict tournament outcomes.

## Quick Start

```bash
pip install -r requirements.txt

# Run a full simulation (10,000 iterations)
python cli.py simulate

# View a team profile
python cli.py profile "Duke"

# Head-to-head matchup analysis
python cli.py matchup "Duke" "Arizona"

# Historical seed trends
python cli.py history --seed 12

# Predicted bracket
python cli.py bracket -n 5000

# Launch web UI
python -m src.web.app
```

## How It Works

**Three pillars feed into a unified win probability model:**

1. **Historical Analytics** — Decades of tournament data establish baseline probabilities for seed matchups (e.g., 1-seeds beat 16-seeds 99.3% of the time, but 5v12 upsets happen 35% of the time)

2. **Team Profiles** — Each team gets scored (0-100) on offense, defense, and intangibles based on efficiency metrics, shooting, rebounding, turnover rates, strength of schedule, and tournament experience

3. **Win Probability Model** — For any matchup, combines historical seed baseline (30%), efficiency margin differential via logistic function (50%), and profile composite comparison (20%). Weights are configurable in `config.py`

4. **Monte Carlo Engine** — Runs N complete tournament simulations, drawing random outcomes weighted by win probabilities. Aggregates results into championship odds, Final Four probabilities, upset alerts, and a predicted bracket

## CLI Commands

| Command | Description |
|---------|-------------|
| `simulate` | Run Monte Carlo simulation with configurable iterations |
| `profile <team>` | View detailed team profile with strengths/weaknesses |
| `matchup <team_a> <team_b>` | Head-to-head probability breakdown |
| `history` | Historical seed performance and upset rates |
| `bracket` | Output predicted bracket with confidence levels |

## Web UI

Run `python -m src.web.app` and visit `http://localhost:5000`:

- **Bracket view** — Full tournament bracket with clickable team links
- **Team profiles** — Radar charts, key metrics, strengths/weaknesses
- **Simulation dashboard** — Run simulations from the browser, see championship odds and upset alerts

## Configuration

Edit `config.py` to tune:

- `DEFAULT_ITERATIONS` — Number of simulations (default: 10,000)
- `WEIGHT_HISTORICAL` / `WEIGHT_EFFICIENCY` / `WEIGHT_PROFILE` — How much each factor contributes to win probability
- `LOGISTIC_SCALE` — Sensitivity of the efficiency differential curve
- `CURRENT_YEAR` — Target tournament year

## Project Structure

```
├── cli.py                    # CLI entry point
├── config.py                 # Global configuration
├── data/
│   ├── historical/           # Tournament results CSV (1985-present)
│   ├── teams/                # Team stats JSON per year
│   └── brackets/             # Bracket seedings per year
├── src/
│   ├── data/                 # Data loading and models
│   ├── analysis/             # Historical analyzer, profiler, matchup evaluator
│   ├── simulation/           # Bracket structure, Monte Carlo engine, results
│   └── web/                  # Flask app with templates
└── tests/                    # Test suite
```
