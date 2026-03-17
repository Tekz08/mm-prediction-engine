import json
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response

import config
from src.data.team_stats import load_teams, load_teams_by_name
from src.analysis.historical import HistoricalAnalyzer
from src.analysis.profiler import build_all_profiles, build_profile
from src.analysis.matchup import MatchupEvaluator
from src.analysis.advisor import BracketAdvisor
from src.simulation.bracket import BracketSimulator, load_bracket, get_first_four_matchups
from src.simulation.engine import MonteCarloEngine
from src.simulation.results import SimulationResults

app = Flask(__name__)

_cache = {}
_active_engine = {"engine": None, "lock": threading.Lock()}


def _get_data(year=None):
    year = year or config.CURRENT_YEAR
    if year not in _cache:
        teams = load_teams(year)
        teams_by_name = {t.name: t for t in teams}
        bracket = load_bracket(year)
        historical = HistoricalAnalyzer()
        profiles = build_all_profiles(teams, historical)
        evaluator = MatchupEvaluator(historical, profiles)
        bracket_sim = BracketSimulator(bracket, teams_by_name)
        _cache[year] = {
            "teams": teams,
            "teams_by_name": teams_by_name,
            "bracket": bracket,
            "historical": historical,
            "profiles": profiles,
            "evaluator": evaluator,
            "bracket_sim": bracket_sim,
        }
    return _cache[year]


@app.route("/")
def index():
    data = _get_data()
    ff_entries = get_first_four_matchups(data["bracket"])
    first_four = [
        {"seed": a.seed, "region": a.region, "team_a": a.team, "team_b": b.team}
        for a, b in ff_entries
    ]
    return render_template(
        "bracket.html", bracket=data["bracket"],
        year=config.CURRENT_YEAR, first_four=first_four,
    )


@app.route("/team/<team_name>")
def team_detail(team_name):
    data = _get_data()
    team = data["teams_by_name"].get(team_name)
    if not team:
        return render_template("404.html", message=f"Team '{team_name}' not found"), 404
    profile = data["profiles"].get(team_name)
    return render_template("team.html", team=team, profile=profile)


@app.route("/api/matchup", methods=["POST"])
def api_matchup():
    body = request.get_json(silent=True) or {}
    team_a_name = body.get("team_a", "")
    team_b_name = body.get("team_b", "")
    data = _get_data()
    team_a = data["teams_by_name"].get(team_a_name)
    team_b = data["teams_by_name"].get(team_b_name)
    if not team_a or not team_b:
        return jsonify({"error": "Team not found"}), 404
    breakdown = data["evaluator"].get_matchup_breakdown(team_a, team_b)
    return jsonify(breakdown)


@app.route("/api/teams")
def api_teams():
    data = _get_data()
    teams_list = []
    for t in data["teams"]:
        p = data["profiles"].get(t.name)
        teams_list.append({
            "name": t.name,
            "seed": t.seed,
            "region": t.region,
            "conference": t.conference,
            "composite_score": p.composite_score if p else 0,
            "net_efficiency": t.stats.net_efficiency,
            "record": f"{t.stats.wins}-{t.stats.losses}",
        })
    return jsonify(teams_list)


@app.route("/matchup")
def matchup_picker():
    data = _get_data()
    teams_list = sorted(
        [{"name": t.name, "seed": t.seed, "region": t.region} for t in data["teams"]],
        key=lambda x: (x["region"], x["seed"]),
    )
    return render_template("matchup.html", teams=teams_list, year=config.CURRENT_YEAR)


@app.route("/matchup/<team_a_name>/<team_b_name>")
def matchup_detail(team_a_name, team_b_name):
    data = _get_data()
    team_a = data["teams_by_name"].get(team_a_name)
    team_b = data["teams_by_name"].get(team_b_name)
    if not team_a or not team_b:
        return render_template("404.html", message="Team not found"), 404
    profile_a = data["profiles"].get(team_a_name)
    profile_b = data["profiles"].get(team_b_name)
    breakdown = data["evaluator"].get_matchup_breakdown(team_a, team_b)
    return render_template(
        "matchup_detail.html",
        team_a=team_a, team_b=team_b,
        profile_a=profile_a, profile_b=profile_b,
        breakdown=breakdown, year=config.CURRENT_YEAR,
    )


def _parse_weights(source):
    weights = {}
    for key, param in [("weight_historical", "w_hist"), ("weight_efficiency", "w_eff"),
                       ("weight_profile", "w_prof"), ("weight_matchup", "w_match")]:
        val = source.get(param)
        if val is not None:
            weights[key] = float(val) / 100.0
    return weights if weights else None


def _make_evaluator(data, weights):
    if not weights:
        return data["evaluator"]
    return MatchupEvaluator(data["historical"], data["profiles"], **weights)


@app.route("/advisor")
def advisor_page():
    return render_template("advisor.html", year=config.CURRENT_YEAR)


def _upset_bias(source, key="chaos"):
    if source.get(key) in (True, "1", 1):
        return 0.5
    return 0.0


@app.route("/api/advisor", methods=["POST"])
def api_advisor():
    body = request.get_json(silent=True) or {}
    iterations = min(int(body.get("iterations", 5000)), 50000)
    year = int(body.get("year", config.CURRENT_YEAR))
    seed = body.get("seed")

    data = _get_data(year)
    weights = _parse_weights(body)
    evaluator = _make_evaluator(data, weights)
    engine = MonteCarloEngine(
        data["bracket_sim"], evaluator,
        iterations=iterations, seed=seed,
    )
    raw = engine.run()
    sim_results = SimulationResults(raw)
    consensus = engine.predict_consensus_bracket(upset_bias=_upset_bias(body))

    adv = BracketAdvisor(
        evaluator, sim_results,
        data["teams_by_name"], data["profiles"],
    )
    picks = adv.generate_bracket(consensus)
    return jsonify(adv.to_dict(picks))


@app.route("/api/advisor-stream")
def api_advisor_stream():
    iterations = min(int(request.args.get("iterations", 5000)), 50000)
    year = int(request.args.get("year", config.CURRENT_YEAR))
    seed_val = request.args.get("seed")
    seed = int(seed_val) if seed_val else None

    data = _get_data(year)
    weights = _parse_weights(request.args)
    evaluator = _make_evaluator(data, weights)
    engine = MonteCarloEngine(
        data["bracket_sim"], evaluator,
        iterations=iterations, seed=seed,
    )
    upset_bias = _upset_bias(dict(request.args), "chaos")

    with _active_engine["lock"]:
        old = _active_engine["engine"]
        if old:
            old.cancel()
        _active_engine["engine"] = engine

    q = queue.Queue()

    def on_progress(current, total):
        q.put({"type": "progress", "current": current, "total": total})

    def run_sim():
        try:
            raw = engine.run(progress_callback=on_progress)
            if engine.cancelled:
                q.put({"type": "cancelled"})
                return
            sim_results = SimulationResults(raw)
            consensus = engine.predict_consensus_bracket(upset_bias=upset_bias)
            adv = BracketAdvisor(
                evaluator, sim_results,
                data["teams_by_name"], data["profiles"],
            )
            picks = adv.generate_bracket(consensus)
            q.put({"type": "result", "data": adv.to_dict(picks)})
        except Exception:
            q.put({"type": "cancelled"})
        finally:
            with _active_engine["lock"]:
                if _active_engine["engine"] is engine:
                    _active_engine["engine"] = None

    threading.Thread(target=run_sim, daemon=True).start()

    def generate():
        while True:
            msg = q.get()
            if msg["type"] == "progress":
                yield f"data: {json.dumps(msg)}\n\n"
            elif msg["type"] == "cancelled":
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                break
            elif msg["type"] == "result":
                yield f"data: {json.dumps(msg)}\n\n"
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/advisor-cancel", methods=["POST"])
def api_advisor_cancel():
    with _active_engine["lock"]:
        engine = _active_engine["engine"]
        if engine:
            engine.cancel()
            _active_engine["engine"] = None
            return jsonify({"status": "cancelled"})
    return jsonify({"status": "no_active_run"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
