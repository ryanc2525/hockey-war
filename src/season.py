"""
The 2026-27 season runner: plays the real schedule from opening night or from any later date, with trades and
sit-outs, and reports standings and playoff odds. Notebook 26 builds and checks it; the Simulator page uses it.

    import season as SE
    bundle = SE.load_bundle(Path("../data/simulator_2026-27"), Path("../data/trades_2026-27.csv"))
    result = SE.run_scenario(bundle, start_date="2026-11-10", sit_out=[8477946], n_sims=10000, seed=0)
    result["table"]          # one row per team: points, percentiles, playoff odds, ...

How this file splits the work with the others:
  - simulate.py is the game engine (one game, many seasons at once). Notebooks 24 and 25 tested it.
  - lineups.py picks each team's lineup and minutes (notebooks 23 and 24).
  - this file is everything specific to 2026-27: the real schedule, real results locked in before a start date,
    which goalie starts each game, the standings rules, the playoff format, and the saved "bundle" of inputs.

The bundle (data/simulator_2026-27/, written once by notebook 26) holds everything a run needs, so a run never
recomputes WAR and never calls the NHL: players.csv (every 2026-27 player, coach-scored), schedule.csv (all
1,344 games, with real results so far) and settings.json (game model, team bonuses, minutes, divisions, ...).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lineups as L
import simulate as S
import scoring as SC

SEASON = 2026
OPENING_NIGHT = "2026-09-29"
GAMES_PER_TEAM = 84
TEAM_LAMBDA = 0.4                        # notebook 25: share of last season's residual kept (fit on training seasons)
TIERED = True                            # notebook 24: call-ups only when the NHL level can't fill a slot
OWN_WEIGHT = {"F": 0.75, "D": 0.75}      # notebook 24: minutes = 75% own recent ice time, 25% slot
PERCENTILES = [10, 25, 50, 75, 90]
DIVISION_SPOTS, WILD_CARDS = 3, 2        # playoffs: top 3 in each division + 2 wild cards per conference
SKIPPED_TIEBREAKERS = ["total wins", "points in head-to-head games", "goal difference", "goals for"]


# ---------------------------------------------------------------------------
# 1. The bundle: saved once by notebook 26, read by every run
# ---------------------------------------------------------------------------
def _late_to_json(late):
    return {"extra_goals": late["extra_goals"],
            "rows": {str(k): [late[k][0].tolist(), late[k][1].tolist(), late[k][2].tolist()]
                     for k in range(S.MAX_MARGIN + 1)}}


def _late_from_json(d):
    out = {int(k): (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in d["rows"].items()}
    out["extra_goals"] = d["extra_goals"]
    return out


def save_bundle(out_dir, players, schedule, schedule_stamp, game, bonus, slots, share, divisions, so_share, inputs):
    """
    Saves everything a run needs (notebook 26 calls this once):
      players: 2026-27 players after lineups.coach_scores, with age, draft_overall and ovr added;
      schedule: edge_data.season_schedule's games; schedule_stamp: when the schedule was downloaded;
      game: simulate.game_settings(...) for 2026-27 (all-seasons coefficients, notebook 25);
      bonus: {team: normal goals per game} (notebook 25's residual x TEAM_LAMBDA / beta);
      slots, share: minutes by slot (scaled to 2026-27's totals) and the starter's share of goalie games;
      divisions: DataFrame with team, division, conference; so_share: share of OT/SO games that are shootouts;
      inputs: simulate.draw_inputs(...) (notebook 17's draw pools, centered).
    Each pool player gets a pool_id (which notebook 17 pool his random draw comes from).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(inputs["pools"])
    key_id = {k: i for i, k in enumerate(keys)}
    p = players.copy()
    p["pool_id"] = -1
    pool = p["in_pool"].to_numpy()
    buckets = S.draw_buckets(p.loc[pool, "group"], p.loc[pool, "age"], p.loc[pool, "draft_overall"], inputs)
    p.loc[pool, "pool_id"] = [key_id[b] for b in buckets]
    p.to_csv(out / "players.csv", index=False)
    schedule.to_csv(out / "schedule.csv", index=False)
    settings = {
        "season": SEASON, "schedule_downloaded": schedule_stamp,
        "game": {k: float(game[k]) for k in ["home", "a_s", "a_g", "total", "ot_b0", "ot_b1"]},
        "late": _late_to_json(game["late"]),
        "bonus": {t: float(v) for t, v in bonus.items()},
        "slots": slots.reset_index().to_dict(orient="records"),
        "share": float(share), "so_share": float(so_share),
        "divisions": divisions[["team", "division", "conference"]].to_dict(orient="records"),
        "draws": {"pools": [{"key": [str(k[0]), str(k[1])], "z": inputs["pools"][k][0].tolist(),
                             "p": inputs["pools"][k][1].tolist()} for k in keys],
                  "fit": inputs["fit"].reset_index().to_dict(orient="records")},
        "lambda": TEAM_LAMBDA, "tiered": TIERED, "own_weight": OWN_WEIGHT,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    return out


def load_bundle(bundle_dir, trades_path):
    """Reads the saved bundle and the hand-kept trades file into one dict (what run_scenario needs)."""
    d = Path(bundle_dir)
    st = json.loads((d / "settings.json").read_text())
    players = pd.read_csv(d / "players.csv")
    schedule = pd.read_csv(d / "schedule.csv")
    slots = pd.DataFrame(st["slots"]).set_index(["group", "rank"])
    game = dict(st["game"])
    game["late"] = _late_from_json(st["late"])
    divisions = pd.DataFrame(st["divisions"])
    pools = [(np.array(x["z"]), np.array(x["p"])) for x in st["draws"]["pools"]]
    fit = pd.DataFrame(st["draws"]["fit"]).set_index("group")[["a", "b"]]
    trades = pd.read_csv(trades_path)
    return {"players": players, "schedule": schedule, "slots": slots, "share": st["share"], "game": game,
            "bonus": st["bonus"], "divisions": divisions, "so_share": st["so_share"], "pools": pools, "fit": fit,
            "trades": trades, "schedule_downloaded": st["schedule_downloaded"],
            "goalie_starts": goalie_starts(schedule, st["share"]), **_load_stat_files(d)}


def _load_stat_files(d):
    """Notebook 27's files, when they're there: scoring rates (scoring.csv), stat-line settings (scoring.json) and
    real 2026-27 games so far, one row per player per game (real_games.csv)."""
    out = {}
    if (d / "scoring.csv").exists() and (d / "scoring.json").exists():
        out["scoring"] = pd.read_csv(d / "scoring.csv").set_index("playerId")
        out["stat_settings"] = json.loads((d / "scoring.json").read_text())
    if (d / "real_games.csv").exists():
        out["real_games"] = pd.read_csv(d / "real_games.csv")
    return out


# ---------------------------------------------------------------------------
# 2. Rules: default trades, who starts in goal, real results so far
# ---------------------------------------------------------------------------
def default_trades(trades, start_date=None):
    """Trades in effect at a start date: every trade dated on or before it (none at opening night).
    On or before, because the dates of T2-T6 are each player's first game with his new team."""
    start = start_date or OPENING_NIGHT
    return sorted(trades.loc[trades["date"] <= start, "trade_id"].unique())


def goalie_starts(schedule, share, games_per_team=GAMES_PER_TEAM):
    """
    Which games each team's backup goalie starts (the same every run, no randomness):
      1. the second night of every back-to-back (a game the day after the team's previous game);
      2. if that's fewer than round((1 - share) x games) starts (30 of 84), more games are added, spread evenly
         over the team's other games, so the starter's share stays near `share`.
    If a team has more back-to-backs than that, the backup gets just the back-to-backs.
    Returns a DataFrame per game (index = schedule rows): home_backup, away_backup (bool), plus a per-team summary.
    """
    target = int(round((1 - share) * games_per_team))
    home_b = pd.Series(False, index=schedule.index)
    away_b = pd.Series(False, index=schedule.index)
    rows = []
    dates = pd.to_datetime(schedule["date"])
    for team in sorted(set(schedule["home"]) | set(schedule["away"])):
        idx = schedule.index[(schedule["home"] == team) | (schedule["away"] == team)]
        d = dates[idx].sort_values(kind="stable")
        b2b = list(d.index[(d.diff().dt.days == 1).to_numpy()])
        backups = list(b2b)
        others = [i for i in d.index if i not in set(b2b)]
        extra = max(target - len(b2b), 0)
        if extra:
            pos = np.floor((np.arange(extra) + 0.5) * len(others) / extra).astype(int)
            backups += [others[k] for k in pos]
        for i in backups:
            if schedule.at[i, "home"] == team:
                home_b[i] = True
            else:
                away_b[i] = True
        rows.append({"team": team, "back_to_backs": len(b2b), "backup_starts": len(backups),
                     "starter_share": 1 - len(backups) / len(idx)})
    out = pd.DataFrame({"home_backup": home_b, "away_backup": away_b})
    out.attrs["summary"] = pd.DataFrame(rows)
    return out


def real_so_far(schedule, start_date):
    """Each team's real results in games played before start_date: gp, wins, OT losses, points, regulation
    wins (rw), regulation + overtime wins (row) and goal difference."""
    p = schedule[schedule["played"] & (schedule["date"] < start_date)]
    hw = p["home_goals"] > p["away_goals"]
    ot = p["last_period"] != "REG"
    so = p["last_period"] == "SO"
    parts = []
    for side, win, gd in [("home", hw, p["home_goals"] - p["away_goals"]), ("away", ~hw, p["away_goals"] - p["home_goals"])]:
        parts.append(pd.DataFrame({"team": p[side], "gp": 1, "wins": win * 1, "otl": (~win & ot) * 1,
                                   "points": np.where(win, 2, ot * 1), "rw": (win & ~ot) * 1,
                                   "row": (win & ~so) * 1, "gd": gd}))
    cols = ["gp", "wins", "otl", "points", "rw", "row", "gd"]
    if not len(p):
        return pd.DataFrame(columns=cols, dtype=float)
    return pd.concat(parts).groupby("team")[cols].sum()


# ---------------------------------------------------------------------------
# 3. Lineups and random draws
# ---------------------------------------------------------------------------
def scenario_lineups(bundle, trade_ids=(), sit_out=()):
    """All 32 lineups after trades and sit-outs (notebook 23's rules with notebook 24's settings)."""
    lu = L.build_lineups(bundle["players"], bundle["slots"], bundle["share"], trades=bundle["trades"],
                         trade_ids=list(trade_ids), sit_out=list(sit_out), tiered=TIERED, own_weight=OWN_WEIGHT)
    return lu


def league_z(bundle, n, rng):
    """
    One random draw (z, before scaling) for every pool player in the league, n simulated seasons.
    Players are in a fixed order (by playerId) and drawn pool by pool, so a player gets the same draw in every
    scenario with the same seed: comparing two scenarios shows the roster change, not new random numbers.
    """
    p = bundle["players"]
    pool = p[p["in_pool"]].sort_values("playerId")
    z = np.zeros((len(pool), n))
    ids = pool["pool_id"].to_numpy()
    for k in sorted(set(ids)):
        rows = np.flatnonzero(ids == k)
        zz, pp = bundle["pools"][k]
        z[rows] = rng.choice(zz, size=(len(rows), n), p=pp)
    return pool["playerId"].to_numpy(), z


def _expected_sd(fit, group, minutes):
    """Notebook 17: the expected size of a player's miss at his minutes, sqrt(a + b / minutes)."""
    g = pd.Series(group)
    return np.sqrt(g.map(fit["a"]).to_numpy() + g.map(fit["b"]).to_numpy() / np.maximum(np.asarray(minutes, float), 1.0))


# ---------------------------------------------------------------------------
# 4. Standings: ranks, division titles, wild cards
# ---------------------------------------------------------------------------
def rank_seasons(teams, divisions, points, rw, row, rng, seeds=False):
    """
    For every simulated season (rows): each team's league rank and whether it won its division, made the
    playoffs or won the Presidents' Trophy. Order: points, then regulation wins, then regulation + overtime wins,
    then a coin flip (the real tiebreakers after those are skipped: see SKIPPED_TIEBREAKERS).
    seeds=True (notebook 27) also returns each team's playoff seed: 1-3 = its place in its division,
    4 = first wild card, 5 = second wild card, 0 = out. The ranks are the same either way.
    """
    n, T = points.shape
    key = points * 1e6 + rw * 1e3 + row + rng.random((n, T))       # one number that sorts in that order
    league_rank = (-key).argsort(axis=1).argsort(axis=1) + 1
    div = divisions.set_index("team").loc[teams]
    div_win, playoff = np.zeros((n, T), bool), np.zeros((n, T), bool)
    seed = np.zeros((n, T), np.int8)
    for conf in sorted(div["conference"].unique()):
        in_conf = np.flatnonzero((div["conference"] == conf).to_numpy())
        top3 = np.zeros((n, T), bool)
        for d in sorted(div.loc[div["conference"] == conf, "division"].unique()):
            cols = np.flatnonzero((div["division"] == d).to_numpy())
            r = (-key[:, cols]).argsort(axis=1).argsort(axis=1) + 1
            div_win[:, cols] = r == 1
            top3[:, cols] = r <= DIVISION_SPOTS
            seed[:, cols] = np.where(r <= DIVISION_SPOTS, r, 0)
        rest = np.where(top3[:, in_conf], -np.inf, key[:, in_conf])  # wild cards: best of the others
        r = (-rest).argsort(axis=1).argsort(axis=1) + 1
        playoff[:, in_conf] = top3[:, in_conf] | (r <= WILD_CARDS)
        seed[:, in_conf] = np.where(top3[:, in_conf], seed[:, in_conf], np.where(r <= WILD_CARDS, DIVISION_SPOTS + r, 0))
    if seeds:
        return league_rank, div_win, playoff, league_rank == 1, seed
    return league_rank, div_win, playoff, league_rank == 1


# ---------------------------------------------------------------------------
# 5. One scenario
# ---------------------------------------------------------------------------
def run_scenario(bundle, start_date=None, trade_ids=None, sit_out=(), n_sims=10000, seed=0, use_bonus=True, chunk=1000,
                 stat_lines=False, keep_seasons=False, updated=None):
    """
    Simulates the rest of the 2026-27 season n_sims times.
      start_date: None = opening night. A later date (YYYY-MM-DD) locks in every real game before it; only games
        on or after it are simulated (even ones already played).
      trade_ids: None = default_trades for the start date; a list (even an empty one) overrides it.
      sit_out: playerIds who don't play (they leave their team's pool; the next player up dresses).
      use_bonus: the team bonus (notebook 25) on or off.
      stat_lines (notebook 27; needs its files in the bundle): also give each team's simulated goals and assists to
        its skaters and each game's decision and goals against to its starting goalie. They use their own random
        numbers, so team results are exactly the same with or without them. Adds "skaters" and "goalies" tables
        (simulated games only; season_lines adds real stats so far). keep_seasons: also keep every simulated
        season's goals and assists per skater ("season_goals", "season_assists": n_sims x skaters).
      updated (notebook 28): in-season learning. A dict with "playerId" (array), "z" (players x samples: each
        player's rest-of-season draws from his updated guesses, inseason.rest_z) and optionally "bonus"
        ({team: normal goals per game}, the updated team bonus). Those players use these draws instead of their
        pool; everyone else is unchanged. None (the default) is notebook 26-27's simulator.
    Lineups are built once, after trades and sit-outs, and stay fixed: full health, no in-season changes.
    The same inputs and seed always give the same results.
    Returns a dict: table (one row per team), points (n_sims x teams), lineups, and what was applied.
    """
    clock = time.time()
    start = start_date or OPENING_NIGHT
    trades = default_trades(bundle["trades"], start) if trade_ids is None else sorted(trade_ids)
    lu = scenario_lineups(bundle, trades, sit_out)
    sched = bundle["schedule"]
    sim = (sched["date"] >= start).to_numpy()
    games = sched[sim]
    starts = bundle["goalie_starts"][sim]
    teams = np.array(sorted(set(sched["home"]) | set(sched["away"])))
    T = len(teams)
    home = np.searchsorted(teams, games["home"].to_numpy())
    away = np.searchsorted(teams, games["away"].to_numpy())
    # Goalie columns: 0..T-1 = each team's starter, T..2T-1 = its backup
    gh = home + T * starts["home_backup"].to_numpy()
    ga = away + T * starts["away_backup"].to_numpy()
    team_bonus = bundle["bonus"] if updated is None or "bonus" not in updated else updated["bonus"]
    bonus = np.array([team_bonus.get(t, 0.0) if use_bonus else 0.0 for t in teams])

    # Each lineup row's place in the league pool (for its random draw) and the size of its draw
    rng_draws = np.random.default_rng([seed, 1])
    rng_games = np.random.default_rng([seed, 2])
    pool_ids, _ = league_z(bundle, 0, np.random.default_rng(0))
    pos = np.searchsorted(pool_ids, lu["playerId"].to_numpy())
    if updated is not None:                                   # where each updated player sits in the league pool
        up_ids = np.asarray(updated["playerId"])
        keep = np.isin(up_ids, pool_ids)
        up_rows = np.searchsorted(pool_ids, up_ids[keep])
        up_z = np.asarray(updated["z"])[keep]
    sd = _expected_sd(bundle["fit"], lu["group"], lu["minutes_per_game"] * GAMES_PER_TEAM)
    goalies = lu[lu["dressed"] & (lu["group"] == "G")]
    g_row = {(t, r): i for i, t, r in zip(goalies.index, goalies["team"], goalies["rank"])}
    starter_rows = np.array([g_row[(t, 1)] for t in teams])
    backup_rows = np.array([g_row.get((t, 2), g_row[(t, 1)]) for t in teams])   # no backup: the starter plays

    tallies = {k: np.zeros((n_sims, T)) for k in ["points", "wins", "otl", "rw", "row", "gd"]}
    if stat_lines:
        # Each dressed skater's team, his place on it (0-17), his rates, minutes and draw pool
        ss = bundle["stat_settings"]
        rng_stats = np.random.default_rng([seed, 3])
        norms = SC.multiplier_norms(bundle["pools"], ss["k"])
        sk_rows = lu.index[lu["dressed"] & (lu["group"] != "G")].to_numpy()
        team_of = np.searchsorted(teams, lu.loc[sk_rows, "team"].to_numpy())
        slot = lu.loc[sk_rows].groupby("team").cumcount().to_numpy()
        K = int(slot.max()) + 1
        rates = bundle["scoring"].reindex(lu.loc[sk_rows, "playerId"])
        mins = lu.loc[sk_rows, "minutes_per_game"].to_numpy()
        g_weight = rates["g60"].fillna(0).to_numpy() * mins          # goals/60 x minutes per game
        a_weight = rates["a60"].fillna(0).to_numpy() * mins          # assists/60 x minutes per game
        sk_pool = bundle["players"].set_index("playerId")["pool_id"].reindex(lu.loc[sk_rows, "playerId"]).to_numpy()
        season_g = np.zeros((n_sims, len(sk_rows)), np.int16)
        season_a = np.zeros((n_sims, len(sk_rows)), np.int16)
        goalie = {k: np.zeros((n_sims, 2 * T)) for k in ["wins", "losses", "otl", "ga"]}
    for c0 in range(0, n_sims, chunk):
        n = min(chunk, n_sims - c0)
        _, z = league_z(bundle, n, rng_draws)
        if updated is not None:                                   # updated players: their in-season draws instead
            z[up_rows] = up_z[:, (c0 + np.arange(n)) % up_z.shape[1]]
        draws = z[pos] * sd[:, None]                                     # WAR/60 change, every lineup row
        _, sk, _ = S.team_strength(lu, draws)
        war60 = lu["proj_war_60"].to_numpy()[:, None] + draws            # a goalie's WAR for a full game
        gk = np.concatenate([war60[starter_rows].T, war60[backup_rows].T], axis=1)
        out = S.simulate_games(home, away, sk, gk, bundle["game"], rng_games, bonus=bonus, goalie_idx=(gh, ga),
                               details=stat_lines)
        hg, ag, ot, hw = out[:4]
        so = ot & (rng_games.random(ot.shape) < bundle["so_share"])     # which OT/SO games were shootouts
        rows = np.repeat(np.arange(n)[:, None], len(home), axis=1)

        def tally(idx, vals):
            return np.bincount((rows * T + idx[None, :]).ravel(), weights=vals.ravel(), minlength=n * T).reshape(n, T)
        sl = slice(c0, c0 + n)
        aw = ~hw
        tallies["points"][sl] = tally(home, np.where(hw, 2, ot * 1)) + tally(away, np.where(aw, 2, ot * 1))
        tallies["wins"][sl] = tally(home, hw * 1) + tally(away, aw * 1)
        tallies["otl"][sl] = tally(home, ot & aw) + tally(away, ot & hw)
        tallies["rw"][sl] = tally(home, hw & ~ot) + tally(away, aw & ~ot)
        tallies["row"][sl] = tally(home, hw & ~so) + tally(away, aw & ~so)
        tallies["gd"][sl] = tally(home, hg - ag) + tally(away, ag - hg)
        if stat_lines:
            late = out[4]
            so_h, so_a = so & hw, so & aw                               # the shootout winner's extra "goal"
            team_goals = (tally(home, hg - so_h) + tally(away, ag - so_a)).astype(np.int64)
            m = SC.scoring_multiplier(z[pos[sk_rows]], sk_pool, norms, ss["k"])     # skaters x n
            wg, wa = np.zeros((n, T, K)), np.zeros((n, T, K))
            wg[:, team_of, slot] = g_weight[None, :] * m.T
            wa[:, team_of, slot] = a_weight[None, :] * m.T
            wg /= wg.sum(-1, keepdims=True)
            wa /= wa.sum(-1, keepdims=True)
            g_split, a_split = SC.split_season(team_goals, wg, wa, ss["mix"], rng_stats)
            assert (g_split.sum(-1) == team_goals).all()               # every team goal has a scorer
            season_g[sl] = g_split[:, team_of, slot]
            season_a[sl] = a_split[:, team_of, slot]
            # Goalies: the decision and goals against go to the starter of that game. Empty-net goals (the leading
            # team's late goals) and the shootout "goal" don't count against him.
            en_h = late["late_home"] * (late["margin"] > 0)
            en_a = late["late_away"] * (late["margin"] < 0)

            def gtally(idx, vals):
                return np.bincount((rows * 2 * T + idx[None, :]).ravel(), weights=np.asarray(vals, float).ravel(),
                                   minlength=n * 2 * T).reshape(n, 2 * T)
            goalie["wins"][sl] = gtally(gh, hw) + gtally(ga, aw)
            goalie["losses"][sl] = gtally(gh, aw & ~ot) + gtally(ga, hw & ~ot)
            goalie["otl"][sl] = gtally(gh, aw & ot) + gtally(ga, hw & ot)
            goalie["ga"][sl] = gtally(gh, ag - so_a - en_a) + gtally(ga, hg - so_h - en_h)

    real = real_so_far(sched, start).reindex(teams).fillna(0)
    for k in tallies:
        tallies[k] += real[k].to_numpy()[None, :]                        # real results before the start date
    rank, div_win, playoff, presidents, seeds = rank_seasons(teams, bundle["divisions"], tallies["points"], tallies["rw"],
                                                             tallies["row"], rng_games, seeds=True)
    pts = tallies["points"]
    div = bundle["divisions"].set_index("team").loc[teams]
    table = pd.DataFrame({"team": teams, "division": div["division"].to_numpy(), "conference": div["conference"].to_numpy(),
                          "points": pts.mean(0)})
    for q in PERCENTILES:
        table[f"p{q}"] = np.percentile(pts, q, axis=0)
    table["wins"] = tallies["wins"].mean(0)
    table["otl"] = tallies["otl"].mean(0)
    table["losses"] = GAMES_PER_TEAM - table["wins"] - table["otl"]
    table["goal_diff"] = tallies["gd"].mean(0)
    table["playoff_pct"] = playoff.mean(0) * 100
    table["division_pct"] = div_win.mean(0) * 100
    table["presidents_pct"] = presidents.mean(0) * 100
    table["avg_rank"] = rank.mean(0)
    table["real_gp"] = real["gp"].to_numpy()
    table["real_points"] = real["points"].to_numpy()
    result = {"table": table, "points": pts, "tallies": tallies, "lineups": lu, "teams": teams, "seeds": seeds,
              "start_date": start, "trades": trades, "sit_out": list(sit_out), "n_sims": n_sims, "seed": seed,
              "games_simulated": int(sim.sum()), "games_locked": int((sched["played"] & (sched["date"] < start)).sum())}
    if stat_lines:
        left = pd.concat([games["home"], games["away"]]).value_counts().reindex(teams).fillna(0)
        skl = lu.loc[sk_rows, ["playerId", "name", "team", "position", "group"]].reset_index(drop=True)
        skl["sim_gp"] = skl["team"].map(left).astype(int)          # fixed lineups: he plays every simulated game
        for col, arr in [("g", season_g), ("a", season_a), ("p", season_g.astype(np.int32) + season_a)]:
            skl[f"sim_{col}"] = arr.mean(0)
            skl[f"sim_{col}_p10"] = np.percentile(arr, 10, axis=0)
            skl[f"sim_{col}_p90"] = np.percentile(arr, 90, axis=0)
        starts = np.bincount(gh, minlength=2 * T) + np.bincount(ga, minlength=2 * T)
        g_rows = np.concatenate([starter_rows, backup_rows])
        gl = pd.DataFrame({"row": g_rows, "sim_gp": starts, **{f"sim_{k}": v.mean(0) for k, v in goalie.items()}})
        gl = gl.groupby("row", as_index=False).sum()               # a team without a backup: one goalie
        gl = lu.loc[gl["row"], ["playerId", "name", "team", "rank"]].reset_index(drop=True).join(gl.drop(columns="row"))
        gl["role"] = np.where(gl["rank"] == 1, "Starter", "Backup")
        result.update({"skaters": skl, "goalies": gl.drop(columns="rank")})
        if keep_seasons:
            result.update({"season_goals": season_g, "season_assists": season_a})
    result["seconds"] = time.time() - clock
    return result


# ---------------------------------------------------------------------------
# 6. Stat lines: real games so far + the simulated rest of the season
# ---------------------------------------------------------------------------
def real_lines(bundle, start_date):
    """
    Each player's real 2026-27 stats in games before start_date (a game counts if he got ice time; a traded
    player's teams are combined): skaters real_gp, real_g, real_a, real_p; goalies real_gp, real_w, real_l,
    real_otl, real_ga, real_min. last_team = the team of his most recent game.
    """
    rg = bundle.get("real_games")
    sk_cols = ["playerId", "name", "position", "last_team", "real_gp", "real_g", "real_a", "real_p"]
    g_cols = ["playerId", "name", "last_team", "real_gp", "real_w", "real_l", "real_otl", "real_ga", "real_min"]
    if rg is None:
        return pd.DataFrame(columns=sk_cols), pd.DataFrame(columns=g_cols)
    r = rg[(rg["date"] < (start_date or OPENING_NIGHT)) & (rg["toi"] > 0)].sort_values(["date", "game_id"])
    s = r[r["group"] != "G"].groupby("playerId").agg(name=("name", "last"), position=("position", "last"),
                                                    last_team=("team", "last"), real_gp=("game_id", "size"),
                                                    real_g=("goals", "sum"), real_a=("assists", "sum")).reset_index()
    s["real_p"] = s["real_g"] + s["real_a"]
    g = r[r["group"] == "G"].assign(w=r["decision"] == "W", l=r["decision"] == "L", o=r["decision"] == "O")
    g = g.groupby("playerId").agg(name=("name", "last"), last_team=("team", "last"), real_gp=("game_id", "size"),
                                  real_w=("w", "sum"), real_l=("l", "sum"), real_otl=("o", "sum"),
                                  real_ga=("goals_against", "sum"), real_min=("toi", "sum")).reset_index()
    return s[sk_cols], g[g_cols]


def season_lines(result, bundle):
    """
    Full-season stat lines: real stats before the start date + the simulated rest (run_scenario with stat_lines).
    At opening night there's nothing real, so it's the simulation alone. A player with real games who isn't in a
    lineup now (sitting out, sent down, traded to a team he doesn't dress for) keeps his real stats.
    For many simulated seasons, the 10th and 90th percentiles are real + the simulated percentiles (real numbers
    don't vary). Goalie GAA = (real + simulated goals against) / (real minutes + 60 x simulated starts) x 60.
    Returns (skaters, goalies), with the real_, sim_ and season columns side by side.
    """
    names = bundle["players"].set_index("playerId")["name"]
    real_s, real_g = real_lines(bundle, result["start_date"])
    s = result["skaters"].merge(real_s.drop(columns="name"), on="playerId", how="outer", suffixes=("", "_real"))
    s["name"] = s["name"].fillna(s["playerId"].map(names)).fillna(s["playerId"].map(real_s.set_index("playerId")["name"]))
    s["team"] = s["team"].fillna(s["last_team"])
    s["position"] = s["position"].fillna(s["position_real"])
    nums = [c for c in s.columns if c.startswith(("sim_", "real_"))]
    s[nums] = s[nums].fillna(0)
    for c in ["gp", "g", "a", "p"]:
        s[c] = s[f"real_{c}"] + s[f"sim_{c}"]
    for c in ["g", "a", "p"]:
        for q in ["p10", "p90"]:
            s[f"{c}_{q}"] = s[f"real_{c}"] + s[f"sim_{c}_{q}"]
    s = s.drop(columns=["position_real", "last_team"])
    g = result["goalies"].merge(real_g.drop(columns="name"), on="playerId", how="outer")
    g["name"] = g["name"].fillna(g["playerId"].map(names)).fillna(g["playerId"].map(real_g.set_index("playerId")["name"]))
    g["team"] = g["team"].fillna(g["last_team"])
    g["role"] = g["role"].fillna("Not dressed")
    nums = [c for c in g.columns if c.startswith(("sim_", "real_"))]
    g[nums] = g[nums].fillna(0)
    g["gp"] = g["real_gp"] + g["sim_gp"]
    for a, b in [("w", "wins"), ("l", "losses"), ("otl", "otl"), ("ga", "ga")]:
        g[a] = g[f"real_{a}"] + g[f"sim_{b}"]
    minutes = g["real_min"] + 60 * g["sim_gp"]
    g["gaa"] = np.where(minutes > 0, g["ga"] / minutes.where(minutes > 0, 1) * 60, np.nan)
    return s.sort_values("p", ascending=False).reset_index(drop=True), g.drop(columns="last_team")


SEED_LABELS = {1: "1", 2: "2", 3: "3", 4: "WC1", 5: "WC2", 0: ""}
