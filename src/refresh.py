"""
The weekly refresh (notebook 29): pull the newest real results, rerun notebook 28's in-season update with its
frozen settings, run the real-world scenario, and save a dated forecast entry that a later notebook can grade.

Thin functions only. The model lives in inseason.py (the update), season.py (the simulator) and scoring.py
(stat lines); this file moves notebook 28's glue (its update26 and team bonus cells) into functions, adds the
checks, and saves the entries. Nothing here reruns a backtest.

    import refresh as R
    settings = R.load_settings(DATA / "forecast_log" / "settings.json")
    B_now = R.with_evidence(bundle, results, box_lines, evidence_date, start_date)
    players, Z = R.player_update(B_now, lineup, start_date, settings, war_file=..., with_skater_war=False)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import inseason as IS
import season as SE

ENTRY_FILES = ["players.csv", "teams.csv", "metadata.json"]


# ---------------------------------------------------------------------------
# 1. Frozen settings (saved once from notebook 28; every weekly run only reads them)
# ---------------------------------------------------------------------------
def save_settings(path, settings):
    """Write the frozen settings once. Refuses if the file already exists, so they can't drift."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; frozen settings are never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=1))


def load_settings(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: the frozen notebook 28 settings must exist before a refresh runs")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 2. Dates and the files Ryan controls
# ---------------------------------------------------------------------------
def day_range(first, last):
    """Every date from first to last (YYYY-MM-DD strings), inclusive."""
    d0, d1 = date.fromisoformat(first), date.fromisoformat(last)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def ensure_csv(path, columns, rows=()):
    """Create a small hand-kept CSV once (with starting rows) if it doesn't exist; never write to it again."""
    path = Path(path)
    if not path.exists():
        pd.DataFrame(list(rows), columns=columns).to_csv(path, index=False)
        print(f"Created {path} ({len(rows)} rows). It's yours to edit; this notebook never writes to it again.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 3. Checks
# ---------------------------------------------------------------------------
def missing_boxscores(results, box_lines, through):
    """Final games dated on or before `through` without a saved boxscore, and postponed games."""
    r = results[results["date"] <= through]
    have = set(box_lines["game_id"])
    missing = r[r["final"] & ~r["game_id"].isin(have)]
    postponed = r[~r["final"]]
    return missing, postponed


def goal_check(results, box_lines):
    """Team goals from the boxscores (skaters' goals, plus a goal credited to a goalie) vs. the daily scores'
    results, game by game. The shootout winner's extra goal in the score isn't a player's goal, so it's taken off."""
    r = results[results["final"] & results["game_id"].isin(box_lines["game_id"])]
    goals = box_lines.groupby(["game_id", "team"])["goals"].sum()
    so = r["last_period"] == "SO"
    rows = []
    for side, other in [("home", "away"), ("away", "home")]:
        won = r[f"{side}_goals"] > r[f"{other}_goals"]
        real = r[f"{side}_goals"] - (so & won)
        box = goals.reindex(pd.MultiIndex.from_arrays([r["game_id"], r[side]])).fillna(0).to_numpy()
        rows.append(pd.DataFrame({"game_id": r["game_id"].to_numpy(), "date": r["date"].to_numpy(), "team": r[side].to_numpy(),
                                  "boxscore": box, "result": real.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["boxscore"] != out["result"]]


def file_dates(paths, newest_boxscore, days=7):
    """Each hand-downloaded file's date (its modification time: MoneyPuck's files carry no date inside), and a
    warning flag if it's more than `days` older than the newest boxscore."""
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            rows.append({"file": p.name, "date": None, "stale": True, "note": "missing"})
            continue
        d = date.fromtimestamp(p.stat().st_mtime)
        stale = (date.fromisoformat(newest_boxscore) - d).days > days
        rows.append({"file": p.name, "date": d.isoformat(), "stale": stale,
                     "note": f"more than {days} days older than the newest boxscore ({newest_boxscore})" if stale else ""})
    return pd.DataFrame(rows)


def roster_drift(box_lines, rosters, trades, through, ack_ids=()):
    """
    Players whose most recent 2026-27 game (through `through`) was for a different team than their roster team,
    after every logged trade dated on or before `through`.
      blocking: roster team is an NHL team and he now plays for a different NHL team (a trade or waiver claim
                not in trades_2026-27.csv yet), minus the playerIds Ryan acknowledged.
      warning: he's on no roster (no row, or no NHL team): affiliate call-ups and the like.
    """
    b = box_lines[(box_lines["date"] <= through) & (box_lines["toi"] > 0)].sort_values(["date", "game_id"])
    last = b.groupby("playerId").agg(name=("name", "last"), team_now=("team", "last"))
    team = rosters.set_index("playerId")["team_opening_night"]
    t = trades[trades["date"] <= through].sort_values("date")
    team = team.copy()
    for pid, to in zip(t["playerId"], t["to_team"]):
        team.loc[pid] = to
    last["roster_team"] = last.index.map(team)
    first_game = b.merge(last[["team_now"]], left_on=["playerId", "team"], right_on=[last.index, "team_now"]) \
        .groupby("playerId")["date"].min()
    last["first_game_with_team_now"] = last.index.map(first_game)
    off_roster = last["roster_team"].isna()
    moved = ~off_roster & (last["roster_team"] != last["team_now"])
    blocking = last[moved & ~last.index.isin(list(ack_ids))].reset_index()
    acknowledged = last[moved & last.index.isin(list(ack_ids))].reset_index()
    warning = last[off_roster].reset_index()
    return blocking, warning, acknowledged


# ---------------------------------------------------------------------------
# 4. The bundle with this week's evidence (in memory only; the bundle files are never rewritten)
# ---------------------------------------------------------------------------
def with_evidence(bundle, results, box_lines, through, start):
    """
    A copy of the simulator bundle whose schedule and real games reflect every real game through `through`:
    played flags, scores and REG/OT/SO from the daily scores; real player-games from the boxscores.
    A game dated on or before `through` that wasn't played (postponed) is moved to its new date if the daily
    scores show one, or else to `start`, so run_scenario simulates it and every team still plays 84.
    The goalie start rule is recomputed for the moved schedule.
    """
    B = dict(bundle)
    s = bundle["schedule"].copy()
    r = results.set_index("game_id")
    done = r[r["final"] & (r["date"] <= through)]
    known = s["game_id"].isin(done.index)
    s["played"] = known
    for col in ["home_goals", "away_goals"]:
        s[col] = s["game_id"].map(done[col]).where(known)
    s["last_period"] = s["game_id"].map(done["last_period"]).where(known)
    s["date"] = s["game_id"].map(done["date"]).where(known, s["date"])    # a made-up game is on its real date
    moved = (~s["played"]) & (s["date"] <= through)
    new_date = s["game_id"].map(r["date"]).where(lambda d: d > through)
    s.loc[moved, "date"] = new_date[moved].fillna(start)
    s = s.sort_values(["date", "game_id"]).reset_index(drop=True)
    B["schedule"] = s
    B["goalie_starts"] = SE.goalie_starts(s, bundle["share"])
    cols = ["game_id", "date", "team", "playerId", "name", "position", "group", "goals", "assists", "toi",
            "goals_against", "decision"]
    B["real_games"] = box_lines[box_lines["date"] <= through][cols].reset_index(drop=True)
    ppd = r.index[~r["final"] & (r["date"] <= through)]                     # postponed in the daily scores
    B["postponed_moved"] = s.loc[s["game_id"].isin(ppd), ["game_id", "date", "home", "away"]].rename(columns={"date": "new_date"})
    return B


def games_check(B, start):
    """Every team's real games before `start` + games left on or after it. Each must be 84."""
    s = B["schedule"]
    real = pd.concat([s.loc[s["played"] & (s["date"] < start), "home"], s.loc[s["played"] & (s["date"] < start), "away"]]).value_counts()
    left = pd.concat([s.loc[s["date"] >= start, "home"], s.loc[s["date"] >= start, "away"]]).value_counts()
    t = pd.DataFrame({"real": real, "left": left}).fillna(0).astype(int)
    t["total"] = t["real"] + t["left"]
    return t


# ---------------------------------------------------------------------------
# 5. The update (notebook 28's update26 and team bonus cells, as functions)
# ---------------------------------------------------------------------------
def reality_lineup(B, trade_ids, sit_out):
    return SE.scenario_lineups(B, trade_ids, sit_out)


def player_table(B, lineup, start, war_file, goalie_file=None, goalie_repl_gap=None):
    """Every pool player with his evidence before `start`, his minutes from the reality lineup and his rates
    (notebook 28's P26). goalie_file / goalie_repl_gap: goalie WAR/60 so far, logged as evidence (the goalie
    update isn't adopted, so it doesn't move anything)."""
    from war_model import run_season
    from goalie_war import load_goalies
    from forecast import GOALS_PER_WIN
    pl = B["players"]
    s = B["schedule"]
    left = pd.concat([s.loc[s["date"] >= start, "home"], s.loc[s["date"] >= start, "away"]]).value_counts()
    P = pl[pl["in_pool"]].sort_values("playerId")[["playerId", "name", "team", "group", "proj_war_60", "pool_id"]].reset_index(drop=True)
    P = P.merge(lineup[["playerId", "team", "minutes_per_game", "dressed"]].rename(columns={"team": "team_now"}), on="playerId", how="left")
    rg = B["real_games"]
    rg = rg[(rg["date"] < start) & (rg["toi"] > 0)]
    ev = rg[rg["group"] != "G"].groupby("playerId").agg(G=("goals", "sum"), A=("assists", "sum"), toi=("toi", "sum"), gp=("game_id", "size"))
    P = P.join(ev, on="playerId")
    P[["G", "A", "toi", "gp"]] = P[["G", "A", "toi", "gp"]].fillna(0.0)
    P["g60"] = P["playerId"].map(B["scoring"]["g60"]).fillna(0.0)
    P["a60"] = P["playerId"].map(B["scoring"]["a60"]).fillna(0.0)
    mpg = P["minutes_per_game"].fillna(0.0)
    P["toi_season"] = np.maximum(mpg * SE.GAMES_PER_TEAM, P["toi"]).clip(lower=1.0)
    P["toi_rest"] = (mpg * P["team_now"].map(left).fillna(0)).clip(lower=1.0)
    war = run_season(war_file).set_index("playerId")
    P["war60_so_far"] = (P["playerId"].map(war["war"]) / P["playerId"].map(war["all_toi"]) * 60).where(P["group"] != "G")
    P["war_min"] = P["playerId"].map(war["all_toi"])
    P["gwar60_so_far"] = np.nan
    if goalie_file is not None and Path(goalie_file).exists():
        g = load_goalies(pd.read_csv(goalie_file)).groupby("playerId").agg(gsax=("gsax", "sum"), gtoi=("toi", "sum"))
        repl = g["gsax"].sum() / g["gtoi"].sum() * 60 + goalie_repl_gap
        P["gwar60_so_far"] = ((P["playerId"].map(g["gsax"]) / P["playerId"].map(g["gtoi"]) * 60 - repl) / GOALS_PER_WIN).where(P["group"] == "G")
    return P, left


def player_update(P, settings, fit, pools, with_skater_war=False):
    """Weights and rest-of-season z for every pool player with the frozen settings (notebook 28's update26)."""
    seed, version, k = settings["update_seed"], settings["version"], settings["k"]
    Z = IS.prior(P["pool_id"].to_numpy(), pools, settings["n_guess"], seed=seed)
    ze = IS.evidence_z(Z, version, fit, P["group"], P["toi_season"])
    f = IS.scoring_factor(ze, k)
    size = IS.draw_size(fit, P["group"], P["toi_season"])
    level_z = Z if version == "whole" else ze
    ll = np.zeros_like(Z)
    if settings["skaters"]:
        sk = ((P["group"] != "G") & (P["g60"] > 0)).to_numpy()[:, None]
        ll += np.where(sk, IS.log_lik_counts(P["G"], P["g60"], P["toi"], f) + IS.log_lik_counts(P["A"], P["a60"], P["toi"], f), 0.0)
    if settings["goalies"]:
        raise NotImplementedError("the goalie update wasn't adopted in notebook 28")
    if with_skater_war:
        ll += IS.log_lik_war(P["war60_so_far"], P["proj_war_60"], level_z, size,
                             IS.war_noise_var(version, fit, P["group"], P["war_min"].fillna(1.0), P["toi_season"]))
    w, ess = IS.weights(ll)
    n_rest = settings["n_rest"]
    zr = IS.rest_z(Z, w, version, fit, P["group"], P["toi_season"], P["toi_rest"], n_rest, seed=seed + 1)
    zr0 = IS.rest_z(Z, np.full_like(w, 1 / Z.shape[1]), version, fit, P["group"], P["toi_season"], P["toi_rest"], n_rest, seed=seed + 1)
    norm = np.exp(k * zr0).mean(axis=1)
    out = pd.DataFrame({"playerId": P["playerId"], "ess": ess,
                        "war60": P["proj_war_60"] + IS.posterior_mean(level_z, w) * size,
                        "factor": np.exp(k * zr).mean(axis=1) / norm})
    out["g60"], out["a60"] = P["g60"] * out["factor"], P["a60"] * out["factor"]
    return out, zr


def team_bonus_update(B, B_open, start, trade_ids, sit_out, P, Z, left, settings):
    """
    The updated team bonus (notebook 28's cell): real points so far vs. what the preseason model expected
    plus the change the player update makes per game, weighted by g / (g + g0) and turned into goals with beta.
    Returns (bonus Series, the runs it used).
    """
    n, seed = settings["n_sims"], settings["scenario_seed"]
    base = SE.run_scenario(B_open, n_sims=n, seed=seed)
    pre = SE.run_scenario(B, start_date=start, trade_ids=trade_ids, sit_out=sit_out, n_sims=n, seed=seed, stat_lines=True)
    upd_p = SE.run_scenario(B, start_date=start, trade_ids=trade_ids, sit_out=sit_out, n_sims=n, seed=seed,
                            updated={"playerId": P["playerId"].to_numpy(), "z": Z})
    real = pre["table"].set_index("team")
    gp_left = real.index.map(left).to_numpy()
    pre_rest = real["points"] - real["real_points"]
    upd_rest = upd_p["table"].set_index("team")["points"] - real["real_points"]
    base_ppg = base["table"].set_index("team")["points"] / SE.GAMES_PER_TEAM
    expect = (base_ppg + (upd_rest - pre_rest) / gp_left) * real["real_gp"]
    gap = (real["real_points"] - expect) / real["real_gp"]
    bonus0 = pd.Series(B["bonus"]).reindex(real.index)
    bonus = bonus0 + IS.team_weight(real["real_gp"], settings["g0"]) * gap / settings["beta"] if settings["teams"] else bonus0
    return bonus, {"base": base, "pre": pre, "upd_players": upd_p, "gap": gap, "bonus0": bonus0}


# ---------------------------------------------------------------------------
# 6. The entry: tables, metadata, an all-or-nothing save that never overwrites
# ---------------------------------------------------------------------------
PLAYER_COLS = ["playerId", "name", "team", "team_now", "group", "gp", "G", "A", "minutes_so_far", "war60_so_far",
               "gwar60_so_far", "pre_war60", "pre_g60", "pre_a60", "upd_war60", "upd_g60", "upd_a60", "upd_ess",
               "upd_war_war60", "upd_war_g60", "upd_war_a60", "upd_war_ess", "team_bonus_pre", "team_bonus_upd",
               "evidence_through", "version", "adopted"]
REST_COLS = ["rest_g", "rest_g_p10", "rest_g_p90", "rest_a", "rest_a_p10", "rest_a_p90", "rest_p", "rest_p_p10",
             "rest_p_p90", "season_p", "season_p_p10", "season_p_p90", "rest_p_war_version"]


def player_entry(P, U_def, U_war, bonus0, bonus, through, settings, lines_def, lines_war):
    """One row per pool player: notebook 28's log columns (same names, same meaning) + rest-of-season lines."""
    log = P[["playerId", "name", "team", "team_now", "group", "gp", "G", "A", "toi", "war60_so_far", "gwar60_so_far",
             "proj_war_60", "g60", "a60"]].copy()
    log = log.rename(columns={"proj_war_60": "pre_war60", "g60": "pre_g60", "a60": "pre_a60", "toi": "minutes_so_far"})
    for tag, U in [("upd", U_def), ("upd_war", U_war)]:
        u = U.set_index("playerId")
        for c in ["war60", "g60", "a60", "ess"]:
            log[f"{tag}_{c}"] = log["playerId"].map(u[c])
    log["team_bonus_pre"] = log["team_now"].map(bonus0)
    log["team_bonus_upd"] = log["team_now"].map(bonus)
    log["evidence_through"] = through
    log["version"] = settings["version"]
    log["adopted"] = json.dumps({"version": settings["version"], "skaters": settings["skaters"],
                                 "goalies": settings["goalies"], "teams": settings["teams"]})
    log = log[PLAYER_COLS]
    return add_rest_lines(log, lines_def, lines_war)


def add_rest_lines(log, lines_def, lines_war):
    """Rest-of-season G/A/P (average, 10th-90th) and full-season P from season.season_lines, default version;
    the skater-WAR version's rest-of-season P next to it."""
    d = lines_def.set_index("playerId")
    w = lines_war.set_index("playerId")
    out = log.copy()
    for c, src in [("rest_g", "sim_g"), ("rest_g_p10", "sim_g_p10"), ("rest_g_p90", "sim_g_p90"), ("rest_a", "sim_a"),
                   ("rest_a_p10", "sim_a_p10"), ("rest_a_p90", "sim_a_p90"), ("rest_p", "sim_p"), ("rest_p_p10", "sim_p_p10"),
                   ("rest_p_p90", "sim_p_p90"), ("season_p", "p"), ("season_p_p10", "p_p10"), ("season_p_p90", "p_p90")]:
        out[c] = out["playerId"].map(d[src])
    out["rest_p_war_version"] = out["playerId"].map(w["sim_p"])
    return out


def team_entry(run_def, run_war, bonus0, bonus):
    t = run_def["table"].set_index("team")
    w = run_war["table"].set_index("team")
    out = pd.DataFrame({"team_bonus_pre": bonus0, "team_bonus_upd": bonus, "real_gp": t["real_gp"], "real_points": t["real_points"],
                        "points": t["points"], "points_p10": t["p10"], "points_p90": t["p90"], "playoff_pct": t["playoff_pct"],
                        "division_pct": t["division_pct"], "presidents_pct": t["presidents_pct"],
                        "points_war_version": w["points"], "playoff_pct_war_version": w["playoff_pct"]})
    out.index.name = "team"
    return out.reset_index()


def code_fingerprint(repo):
    """The git commit, whether any src/ file has uncommitted changes, and a sha256 of every src/*.py module this
    Python session imported (so an entry pins the exact code even when the working tree isn't committed)."""
    repo = Path(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=repo, capture_output=True, text=True).stdout.strip()
    src = (repo / "src").resolve()
    hashes = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().parent == src and f.endswith(".py"):
            hashes[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    return {"git_commit": commit, "src_uncommitted_changes": bool(dirty), "src_sha256": dict(sorted(hashes.items()))}


def entry_exists(root, through):
    return (Path(root) / through).exists()


def save_entry(root, through, players, teams, metadata, evidence=None):
    """
    Save one entry folder root/<through>/ all or nothing: everything is written to a temporary folder first,
    then renamed in one step. Refuses (returns False, writes nothing) if the entry already exists.
    """
    root = Path(root)
    final = root / through
    if final.exists():
        print(f"An entry for {through} already exists ({final}); this run won't save. Nothing was written.")
        return False
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp_{through}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    players.to_csv(tmp / "players.csv", index=False)
    teams.to_csv(tmp / "teams.csv", index=False)
    if evidence:
        (tmp / "evidence").mkdir()
        for name, df in evidence.items():
            df.to_csv(tmp / "evidence" / f"{name}.csv", index=False)
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=1, default=str))
    os.rename(tmp, final)
    print(f"Saved the entry {final} ({len(players):,} players, {len(teams)} teams)")
    return True


def entries(root):
    """Saved entry folders (dates), oldest first."""
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.exists() else []


def load_entry(root, through):
    d = Path(root) / through
    return pd.read_csv(d / "players.csv"), pd.read_csv(d / "teams.csv"), json.loads((d / "metadata.json").read_text())
