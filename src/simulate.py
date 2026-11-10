"""
The season simulator: a game model, player randomness, and a fast season engine. Notebook 24 fits and tests it;
notebook 25 tests coefficient fits and a team-level ingredient; later steps run 2026-27 on it.

    import simulate as S
    table = S.simulate_season(lineups, schedule, settings, n_sims=1000, seed=24)

How one game is simulated (all vectorized: thousands of seasons at once):
  1. "Normal" goals (everything except late, pulled-goalie goals) for each team are Poisson draws. Their total
     is the league's scoring level; the expected difference is
        home ice + a_s x (skater WAR per game, home - away) + a_g x (goalie WAR per game, home - away).
  2. Late game: with the goalie pulled, the trailing team sometimes ties it (a 6-on-5 goal) and the leading team
     often scores into the empty net. These come from a table of what really happened, by the normal-goal margin.
  3. Still tied: overtime or a shootout. The home team wins it with a probability that tilts a little with
     strength; the winner gets one more goal, as the standings count it.
  Points: win 2, overtime/shootout loss 1.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skellam

from development import age_bucket, draft_bucket, luck_share, expected_sd

MAX_MARGIN = 4                    # late table: normal-goal margins of 4 or more share one row
PERCENTILES = [10, 25, 50, 75, 90]
DRAW_PARTS = ["none", "full", "non-luck"]


# ---------------------------------------------------------------------------
# 1. Real games and standings
# ---------------------------------------------------------------------------
def games_from_shots(shots):
    """
    Regular-season games from MoneyPuck's shot data, one row per game, in the same format as
    edge_data.schedule_games: game_id, season, home, away, home_goals, away_goals (a shootout winner gets +1, as
    the standings count it), last_period (REG, OT, SO), plus normal_home / normal_away (regulation goals except
    empty-net goals and goals with the scoring team's own goalie pulled) and reg_home / reg_away (all regulation goals).
    """
    s = shots[shots["isPlayoffGame"] == 0]
    info = s.groupby(["season", "game_id"]).agg(home=("homeTeamCode", "first"), away=("awayTeamCode", "first"),
                                                home_won=("homeTeamWon", "max"), last=("period", "max"))
    g = s[s["goal"] == 1].copy()
    g["home_goal"] = g["isHomeTeam"] == 1
    own_empty = np.where(g["home_goal"], g["homeEmptyNet"], g["awayEmptyNet"]) == 1
    g["normal"] = (g["shotOnEmptyNet"] != 1) & ~own_empty
    reg = g[g["period"] <= 3]
    count = lambda d: d.groupby(["season", "game_id", "home_goal"]).size().unstack(fill_value=0)
    out = info.copy()
    for name, d in [("reg", reg), ("normal", reg[reg["normal"]]), ("all", g)]:
        c = count(d).reindex(info.index, fill_value=0)
        out[f"{name}_home"] = c.get(True, 0)
        out[f"{name}_away"] = c.get(False, 0)
    tied = out["all_home"] == out["all_away"]                       # tied after overtime: a shootout
    out["last_period"] = np.where(out["last"] <= 3, "REG", np.where(tied, "SO", "OT"))
    out["home_goals"] = out["all_home"] + (tied & (out["home_won"] == 1))
    out["away_goals"] = out["all_away"] + (tied & (out["home_won"] == 0))
    out = out.reset_index()
    cols = ["game_id", "season", "home", "away", "home_goals", "away_goals", "last_period",
            "normal_home", "normal_away", "reg_home", "reg_away"]
    return out[cols]


def add_results(games):
    """Adds home_win, ot (went past regulation), reg_margin (home - away after regulation; 0 if it went to OT/SO),
    and each team's points."""
    g = games.copy()
    g["home_win"] = g["home_goals"] > g["away_goals"]
    g["ot"] = g["last_period"] != "REG"
    g["reg_margin"] = np.where(g["ot"], 0, g["home_goals"] - g["away_goals"])
    g["home_points"] = np.where(g["home_win"], 2, np.where(g["ot"], 1, 0))
    g["away_points"] = np.where(~g["home_win"], 2, np.where(g["ot"], 1, 0))
    return g


def standings(games):
    """Real standings from game results: per season and team, games, wins, OT losses, points, goals for/against."""
    g = add_results(games)
    home = pd.DataFrame({"season": g["season"], "team": g["home"], "win": g["home_win"], "points": g["home_points"],
                         "gf": g["home_goals"], "ga": g["away_goals"], "ot": g["ot"]})
    away = pd.DataFrame({"season": g["season"], "team": g["away"], "win": ~g["home_win"], "points": g["away_points"],
                         "gf": g["away_goals"], "ga": g["home_goals"], "ot": g["ot"]})
    t = pd.concat([home, away])
    t["otl"] = ~t["win"] & t["ot"]
    out = t.groupby(["season", "team"]).agg(gp=("points", "size"), wins=("win", "sum"), otl=("otl", "sum"),
                                            points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    out["gd"] = out["gf"] - out["ga"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# 2. The game model
# ---------------------------------------------------------------------------
def late_table(games):
    """
    What happens late in regulation, by the normal-goal margin k = |home - away| (4+ pooled):
    the joint chance of the leading team adding L goals (mostly empty-net) and the trailing team adding T goals
    (mostly 6-on-5 with its goalie pulled). For tied games (k = 0) it's home and away, made symmetric.
    Returns {k: (L values, T values, probabilities)} plus "extra_goals" (late goals per game).
    """
    m = games["normal_home"] - games["normal_away"]
    add_h, add_a = games["reg_home"] - games["normal_home"], games["reg_away"] - games["normal_away"]
    lead = np.where(m >= 0, add_h, add_a)
    trail = np.where(m >= 0, add_a, add_h)
    k = np.minimum(m.abs(), MAX_MARGIN)
    out = {}
    for kk in range(MAX_MARGIN + 1):
        sel = (k == kk).to_numpy()
        a, b = lead[sel], trail[sel]
        if kk == 0:                                                     # tied: no leader, so count both ways
            a, b = np.concatenate([a, b]), np.concatenate([b, a])
        pairs = pd.Series(list(zip(a, b))).value_counts(normalize=True).sort_index()
        out[kk] = (np.array([p[0] for p in pairs.index]), np.array([p[1] for p in pairs.index]), pairs.to_numpy())
    out["extra_goals"] = float((add_h + add_a).mean())
    return out


def _late_mean(table, kk):
    L, T, p = table[kk]
    return float(np.sum((L - T) * p))


def stretch(table, total, d=0.25):
    """
    How much the late game stretches the expected margin: E[regulation margin] / E[normal-goal margin], computed
    exactly (Skellam distribution of the normal-goal margin) around a small expected difference d.
    A regression on regulation margins gives coefficients x this stretch; dividing by it gives normal-goal ones.
    """
    def mean_final(diff):
        mu_h, mu_a = (total + diff) / 2, (total - diff) / 2
        ms = np.arange(-15, 16)
        pm = skellam.pmf(ms, mu_h, mu_a)
        late = np.array([np.sign(m) * _late_mean(table, min(abs(m), MAX_MARGIN)) for m in ms])
        return float(np.sum(pm * (ms + late)))
    return (mean_final(d) - mean_final(-d)) / (2 * d)


def fit_game_model(games, table, total, weights=None):
    """
    games: one row per game with ds = skater WAR per game (home - away), dg = goalie WAR per game (home - away),
    reg_margin, ot, home_win. total = normal goals per game.
    1. Least squares: reg_margin = c0 + cs x ds + cg x dg (c0 = home ice), with standard errors.
    2. Divide by the late-game stretch to get normal-goal coefficients (what the simulator uses).
    3. Overtime: logistic fit of the home team winning OT/SO games on its strength edge in goals.
    weights (notebook 25): one weight per game, for example to count recent seasons more. They apply to both
    fits; None (the default) is notebook 24's equal weights.
    """
    X = np.column_stack([np.ones(len(games)), games["ds"], games["dg"]])
    y = games["reg_margin"].to_numpy(dtype=float)
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.mean()                                              # only relative weights matter
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        n_eff = w.sum() ** 2 / (w ** 2).sum()                         # effective number of games
        sigma2 = np.sum(w * resid ** 2) / (w.sum() * (1 - 3 / n_eff))
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * w[:, None]))) * sigma2)
    k = stretch(table, total)
    fit = {"c0": beta[0], "cs": beta[1], "cg": beta[2], "se_c0": se[0], "se_cs": se[1], "se_cg": se[2],
           "stretch": k, "home": beta[0] / k, "a_s": beta[1] / k, "a_g": beta[2] / k, "n_games": len(games)}
    is_ot = games["ot"].to_numpy(dtype=bool)
    ot = games[is_ot]
    edge = fit["a_s"] * ot["ds"].to_numpy() + fit["a_g"] * ot["dg"].to_numpy()
    yw = ot["home_win"].to_numpy(dtype=float)
    w_ot = np.ones(len(ot)) if weights is None else np.asarray(weights, dtype=float)[is_ot]

    def nll(b):
        z = b[0] + b[1] * edge
        return np.sum(w_ot * (np.logaddexp(0, z) - yw * z))
    res = minimize(nll, x0=[0.0, 0.0], method="BFGS")
    fit.update({"ot_b0": res.x[0], "ot_b1": res.x[1], "ot_games": len(ot), "ot_home_rate": float(yw.mean() if weights is None else np.average(yw, weights=w_ot))})
    return fit


def game_settings(fit, table, total):
    """Everything simulate_games needs, in one dict."""
    return {"home": fit["home"], "a_s": fit["a_s"], "a_g": fit["a_g"], "total": total,
            "ot_b0": fit["ot_b0"], "ot_b1": fit["ot_b1"], "late": table}


# ---------------------------------------------------------------------------
# 3. Player randomness (notebook 17's draws, vectorized)
# ---------------------------------------------------------------------------
def draw_inputs(dev_residuals, dev_ranges, center_pools=True):
    """
    From notebook 17's saved files: the draw pools {(pool_group, sim_bucket): (z values, chances by minutes)},
    the map from (group, age bucket, draft bucket) to a pool, the luck fit (a, b) and carry.
    center_pools=True: each pool's minutes-weighted mean is removed, so a draw adds spread only. Notebook 17's
    pools were misses against notebook 15's projection, so they lean (young high picks up, players 33+ down);
    notebook 18's prior already moved the projection that way, and drawing the lean again would count it twice.
    """
    r = dev_residuals
    pools = {}
    for k, d in r.groupby(["pool_group", "sim_bucket"]):
        p = (d["toi"] / d["toi"].sum()).to_numpy()
        z = d["z_c"].to_numpy()
        pools[k] = (z - np.sum(z * p) if center_pools else z, p)
    cells = r[r["pool_group"] == "F+D"].drop_duplicates(["age_bucket", "draft_bucket"])
    cell_map = {(a, d): b for a, d, b in zip(cells["age_bucket"], cells["draft_bucket"], cells["sim_bucket"])}
    plain = r[r["pool_group"] != "F+D"].drop_duplicates(["group", "age_bucket"])
    age_map = {(g, a): b for g, a, b in zip(plain["group"], plain["age_bucket"], plain["sim_bucket"])}
    fit = dev_ranges[dev_ranges["kind"] == "luck_fit"].set_index("group")[["a", "b"]]
    carry = dev_ranges[(dev_ranges["kind"] == "persistence") & dev_ranges["carry"].notna()].set_index("group")["carry"]
    return {"pools": pools, "cell_map": cell_map, "age_map": age_map, "fit": fit, "carry": carry.to_dict()}


def draw_buckets(group, age, draft_overall, inputs, unknown_age=28):
    """Each player's draw pool, by notebook 17's rule: skaters 23 and under in a flagged age x draft cell use that
    cell (forwards and defensemen pooled); everyone else his group's age bucket. No birth date: age 28."""
    age = pd.Series(age, dtype=float).fillna(unknown_age).to_numpy()
    ab = age_bucket(age)
    db = draft_bucket(pd.Series(draft_overall, dtype=float).to_numpy())
    out = []
    for g, a, ag, d in zip(group, ab, age, db):
        if g != "G" and ag <= 23 and (a, d) in inputs["cell_map"]:
            out.append(("F+D", inputs["cell_map"][(a, d)]))
        else:
            out.append((g, inputs["age_map"][(g, a)]))
    return out


def player_draws(group, buckets, minutes, inputs, parts, n, seed):
    """
    n draws of each player's season, as a change in WAR/60 (array players x n), the way notebook 17 draws:
    a centered miss z from his pool (chance in proportion to minutes), scaled to his minutes,
    miss = z x sqrt(a + b / minutes). parts:
      "none": no draw (zeros);  "full": the whole miss;
      "non-luck": the miss minus its sampling-luck share (one-year change + persistent share only), because the
      game model already adds game-to-game luck.
    """
    group, minutes = np.asarray(group), np.maximum(np.asarray(minutes, dtype=float), 1.0)
    out = np.zeros((len(group), n))
    if parts == "none":
        return out
    rng = np.random.default_rng(seed)
    for key in sorted(set(buckets)):                       # one pool at a time, always in the same order
        idx = np.array([i for i, b in enumerate(buckets) if b == key])
        z, p = inputs["pools"][key]
        out[idx] = rng.choice(z, size=(len(idx), n), p=p)
    out *= expected_sd(inputs["fit"], group, minutes)[:, None]
    if parts == "non-luck":
        share = np.array([luck_share(inputs["fit"], g, m) for g, m in zip(group, minutes)])
        out *= (1 - share)[:, None]
    return out


# ---------------------------------------------------------------------------
# 4. Team strength
# ---------------------------------------------------------------------------
def team_strength(lineups, draws=None):
    """
    Skater and goalie WAR per game for every team (dressed players only):
    sum of (projected WAR/60 + draw) x minutes per game / 60. draws: array (rows of lineups x n_sims) or None.
    Returns (teams, skater array n_sims x teams, goalie array n_sims x teams).
    """
    d = lineups["dressed"].to_numpy()
    teams = np.array(sorted(lineups["team"].unique()))
    t_idx = np.searchsorted(teams, lineups["team"].to_numpy())
    war60 = lineups["proj_war_60"].to_numpy()[:, None] + (0 if draws is None else draws)
    per_game = war60 * (lineups["minutes_per_game"].to_numpy() / 60)[:, None] * d[:, None]
    n = per_game.shape[1]
    is_g = (lineups["group"] == "G").to_numpy()
    sk, gk = np.zeros((n, len(teams))), np.zeros((n, len(teams)))
    for arr, m in [(sk, ~is_g), (gk, is_g)]:
        np.add.at(arr.T, t_idx[m], per_game[m])
    return teams, sk, gk


# ---------------------------------------------------------------------------
# 5. Simulating games and seasons
# ---------------------------------------------------------------------------
def simulate_games(home, away, skater, goalie, settings, rng, bonus=None):
    """
    Every game in a schedule for every simulated season at once.
    home, away: team positions (arrays of length games). skater, goalie: arrays n_sims x teams (WAR per game).
    bonus (notebook 25): a team-level boost in normal goals per game (array of length teams), added to the
    strength edge; None (the default) is notebook 24's model.
    Returns home goals, away goals, went-to-OT and home-win arrays (n_sims x games).
    """
    s = settings
    edge = s["a_s"] * (skater[:, home] - skater[:, away]) + s["a_g"] * (goalie[:, home] - goalie[:, away])
    if bonus is not None:
        edge = edge + (bonus[home] - bonus[away])[None, :]
    diff = s["home"] + edge
    mu_h = np.clip((s["total"] + diff) / 2, 0.05, None)
    mu_a = np.clip((s["total"] - diff) / 2, 0.05, None)
    hg, ag = rng.poisson(mu_h), rng.poisson(mu_a)
    m = hg - ag
    k = np.minimum(np.abs(m), MAX_MARGIN)
    u = rng.random(m.shape)
    lead_add, trail_add = np.zeros_like(m), np.zeros_like(m)
    for kk in range(MAX_MARGIN + 1):
        L, T, p = s["late"][kk]
        sel = k == kk
        j = np.minimum(np.searchsorted(np.cumsum(p), u[sel], side="right"), len(p) - 1)
        lead_add[sel], trail_add[sel] = L[j], T[j]
    home_leads = m >= 0                                   # tied games: "leader" = home (the table is symmetric)
    hg = hg + np.where(home_leads, lead_add, trail_add)
    ag = ag + np.where(home_leads, trail_add, lead_add)
    ot = hg == ag
    p_home = 1 / (1 + np.exp(-(s["ot_b0"] + s["ot_b1"] * edge)))
    ot_home = rng.random(m.shape) < p_home
    hg = hg + (ot & ot_home)
    ag = ag + (ot & ~ot_home)
    return hg, ag, ot, hg > ag


def simulate_game(home_skater, away_skater, home_goalie, away_goalie, settings, n=1, seed=None):
    """One matchup simulated n times: home goals, away goals, went to OT/SO, home won (arrays of length n)."""
    rng = np.random.default_rng(seed)
    sk = np.tile([[home_skater, away_skater]], (n, 1))
    gk = np.tile([[home_goalie, away_goalie]], (n, 1))
    hg, ag, ot, hw = simulate_games(np.array([0]), np.array([1]), sk, gk, settings, rng)
    return hg[:, 0], ag[:, 0], ot[:, 0], hw[:, 0]


def simulate_season(lineups, schedule, settings, n_sims=1000, seed=0, draws=None, chunk=1000, bonus=None):
    """
    n_sims seasons of a schedule (columns home, away). lineups: one row per pool player with team, group, dressed,
    proj_war_60, minutes_per_game. draws: player randomness (rows of lineups x n_sims) or None.
    bonus (notebook 25): {team: normal goals per game} added to a team's strength (missing teams get 0); None
    (the default) is notebook 24's model.
    The same seed always gives the same results.
    Returns (table, points): table = per team the average points, wins, OT losses, goal difference and points
    percentiles; points = array n_sims x teams.
    """
    rng = np.random.default_rng(seed)
    teams = np.array(sorted(lineups["team"].unique()))
    home = np.searchsorted(teams, schedule["home"].to_numpy())
    away = np.searchsorted(teams, schedule["away"].to_numpy())
    assert (teams[home] == schedule["home"].to_numpy()).all() and (teams[away] == schedule["away"].to_numpy()).all()
    T = len(teams)
    b = None if bonus is None else np.array([float(bonus.get(t, 0.0)) for t in teams])
    pts, wins, otl, gd = (np.zeros((n_sims, T)) for _ in range(4))
    for start in range(0, n_sims, chunk):
        n = min(chunk, n_sims - start)
        _, sk, gk = team_strength(lineups, None if draws is None else draws[:, start:start + n])
        if sk.shape[0] == 1:
            sk, gk = np.repeat(sk, n, axis=0), np.repeat(gk, n, axis=0)
        hg, ag, ot, hw = simulate_games(home, away, sk, gk, settings, rng, bonus=b)
        rows = np.repeat(np.arange(n)[:, None], len(home), axis=1)
        tally = lambda team_idx, vals: np.bincount((rows * T + team_idx[None, :]).ravel(), weights=vals.ravel(),
                                                   minlength=n * T).reshape(n, T)
        sl = slice(start, start + n)
        pts[sl] = tally(home, np.where(hw, 2, ot * 1)) + tally(away, np.where(~hw, 2, ot * 1))
        wins[sl] = tally(home, hw * 1) + tally(away, ~hw * 1)
        otl[sl] = tally(home, ot & ~hw) + tally(away, ot & hw)
        gd[sl] = tally(home, hg - ag) + tally(away, ag - hg)
    table = pd.DataFrame({"team": teams, "points": pts.mean(0), "wins": wins.mean(0), "ot_losses": otl.mean(0),
                          "goal_diff": gd.mean(0)})
    for q in PERCENTILES:
        table[f"p{q}"] = np.percentile(pts, q, axis=0)
    return table, pts


def balanced_schedule(teams, games_per_team, seed=0):
    """A simple schedule for tests: rounds of random pairings, each team plays games_per_team games, home and away
    as even as possible. (Step 3 uses the real 2026-27 schedule.)"""
    rng = np.random.default_rng(seed)
    teams = list(teams)
    homes = dict.fromkeys(teams, 0)
    rows = []
    for _ in range(games_per_team):
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            h, v = (a, b) if homes[a] < homes[b] or (homes[a] == homes[b] and rng.random() < 0.5) else (b, a)
            homes[h] += 1
            rows.append({"home": h, "away": v})
    return pd.DataFrame(rows)
