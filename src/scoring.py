"""
Player stat lines for the season simulator (notebook 27): scoring rates, the link to each player's random season,
and how a team's simulated goals and assists are given to its players.

    import scoring as SC
    raw = SC.scoring_history([DATA / "skaters_2008_to_2024.csv", DATA / "skaters25-26.csv"], skater_war)
    goals = SC.rate_table(raw, "goals")          # the shape forecast.project and forecast.backtest read
    proj = SC.project_rates(goals, bios, 2026, reg_minutes=500)

The idea, in four steps:
  1. Rates. Each skater's goals per 60 and assists per 60 (all situations), projected the way notebook 15 projects
     WAR: his last 3 seasons, weighted equally, pulled toward his position's average with reg_minutes of weight,
     and adjusted for age. forecast.py does the math: a rate table looks like a WAR history table, with gar = the
     count (goals or assists) and war = gar / GOALS_PER_WIN, so project() gives (count per 60) / GOALS_PER_WIN.
     Rookies and prospects: NHLe points per game turned into points per 60 at rookie minutes, shrunk toward the
     rookie average, and split into goals and assists by the position's goal share.
  2. His random season. A skater's rates are multiplied by m = exp(k x z) (divided by its average over his draw
     pool, so the average m is 1), where z is the same notebook 17 draw that moves his WAR that season.
  3. Goals. A team's simulated goals go to its dressed skaters in proportion to goals/60 x m x minutes per game.
  4. Assists. Each goal gets 0, 1 or 2 assists at the real mix; the team's assists go to its dressed skaters in
     proportion to assists/60 x m x minutes per game.

Why the whole season can be split at once (split_season) instead of goal by goal (split_by_goal): lineups are
fixed for the season, so every goal a team scores has the same chances. Handing out N goals one at a time with
the same chances is exactly a multinomial draw of N, so the goals are the same either way.
Assists and "never the scorer": a player's assists/60 is measured on real goals he didn't score, so it already
has that rule in it. Weighting assists by assists/60 gives every player exactly his projected share on average.
Played out goal by goal (the scorer can't assist, nobody gets both assists on one goal), the averages barely move
but each player's season total spreads a little less. The season-level split counts goals and assists the way
the link strength k was fit on real seasons (section 6 of notebook 27), so its spread is the one calibrated to
real player-seasons. Notebook 27 shows the comparison.
"""
import numpy as np
import pandas as pd

from forecast import project, aging_curve, GOALS_PER_WIN

GROUP_AGING = "adjust_then_blend"          # notebook 18's age adjustment for skaters
WEIGHTS = (1, 1, 1)                        # notebook 15: the last 3 seasons, weighted equally


# ---------------------------------------------------------------------------
# 1. Scoring history and rates
# ---------------------------------------------------------------------------
def scoring_history(skater_files, skater_war):
    """
    One row per skater-season (all situations) from MoneyPuck's skater files: playerId, season, name, team,
    position, group (F or D, from the WAR table when it has him), gp, toi (minutes), goals, a1, a2, assists, points.
    """
    cols = ["playerId", "season", "name", "team", "position", "situation", "games_played", "icetime",
            "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists"]
    raw = pd.concat([pd.read_csv(f, usecols=cols) for f in skater_files], ignore_index=True)
    raw = raw[raw["situation"] == "all"].drop(columns="situation")
    grp = skater_war.drop_duplicates(["playerId", "season"]).set_index(["playerId", "season"])["pos_group"]
    key = pd.MultiIndex.from_arrays([raw["playerId"], raw["season"]])
    raw["group"] = grp.reindex(key).to_numpy()
    raw["group"] = raw["group"].fillna(pd.Series(np.where(raw["position"] == "D", "D", "F"), index=raw.index))
    out = raw.rename(columns={"games_played": "gp", "I_F_goals": "goals", "I_F_primaryAssists": "a1",
                              "I_F_secondaryAssists": "a2"})
    out["toi"] = out["icetime"] / 60
    out["assists"] = out["a1"] + out["a2"]
    out["points"] = out["goals"] + out["assists"]
    return out.drop(columns="icetime").reset_index(drop=True)


def rate_table(raw, stat):
    """A scoring history in the shape forecast.py reads: gar = the count of `stat`, war = gar / GOALS_PER_WIN."""
    t = raw[["playerId", "season", "name", "team", "position", "group", "toi"]].copy()
    t["gar"] = raw[stat].to_numpy(dtype=float)
    t["war"] = t["gar"] / GOALS_PER_WIN
    return t


def project_rates(table, bios, target, reg_minutes, aging=True):
    """Projected count per 60 (goals or assists) going into `target`, for every skater with a season in the 3
    before it. aging=True: notebook 18's adjust-then-blend with this stat's own aging curve."""
    version = GROUP_AGING if aging else "none"
    curve = aging_curve(table, bios)[1] if aging else None
    p = project(table, bios, target, WEIGHTS, reg_minutes, {"F": version, "D": version, "G": "none"}, curve)
    return p["proj_war_60"] * GOALS_PER_WIN


def assist_mix(raw, seasons):
    """Share of goals with 0, 1 and 2 assists in `seasons`: every goal with a secondary assist also has a primary."""
    r = raw[raw["season"].isin(seasons)]
    g, a1, a2 = r["goals"].sum(), r["a1"].sum(), r["a2"].sum()
    return {"p0": float(1 - a1 / g), "p1": float((a1 - a2) / g), "p2": float(a2 / g)}


def rookie_rates(nhle_ppg, group, minutes, avg_p60, goal_share, shrink):
    """
    Goals and assists per 60 for a rookie or prospect from his NHLe (NHL points per game):
    points/60 = NHLe x 60 / rookie minutes per game, shrunk toward the rookie average
    (avg + shrink x (NHLe rate - avg)), then split by his position's goal share. No NHLe: the rookie average.
    group, minutes, avg_p60 and goal_share line up with nhle_ppg (minutes etc. already mapped by group).
    """
    nh = pd.Series(nhle_ppg, dtype=float).to_numpy()
    avg = np.asarray(avg_p60, dtype=float)
    p60 = np.where(np.isnan(nh), avg, avg + shrink * (nh * 60 / np.asarray(minutes, dtype=float) - avg))
    p60 = np.maximum(p60, 0.0)
    share = np.asarray(goal_share, dtype=float)
    return p60 * share, p60 * (1 - share)


# ---------------------------------------------------------------------------
# 2. The link to each player's random season
# ---------------------------------------------------------------------------
def multiplier_norms(pools, k):
    """For each draw pool: the average of exp(k x z) over the pool, so dividing by it keeps the average m at 1."""
    return np.array([float(np.sum(p * np.exp(k * z))) for z, p in pools])


def scoring_multiplier(z, pool_id, norms, k):
    """m = exp(k x z) / (its pool average): his scoring rates in that simulated season are rate x m.
    z: one value per player, or players x simulated seasons."""
    z = np.asarray(z, dtype=float)
    d = norms[np.asarray(pool_id)]
    return np.exp(k * z) / (d[:, None] if z.ndim == 2 else d)


# ---------------------------------------------------------------------------
# 3-4. Giving a team's goals and assists to its players
# ---------------------------------------------------------------------------
def split_season(goals, s, a, mix, rng):
    """
    The whole season at once. goals: team goals (any shape, e.g. sims x teams); s, a: scoring and assist weights
    (same shape + skaters, each summing to 1). mix: {"p0", "p1", "p2"}.
    Returns (goals, assists) per skater (same shape as s). Team goals add up exactly; team assists = the number
    of assisted goals + the number with a second assist.
    """
    g = rng.multinomial(goals, s)
    n1 = rng.binomial(goals, mix["p1"] + mix["p2"])                       # goals with at least one assist
    n2 = rng.binomial(n1, mix["p2"] / (mix["p1"] + mix["p2"]))            # ... and a second one
    ast = rng.multinomial(n1 + n2, a)
    return g, ast


def goal_by_goal_weights(s, a, mix, iters=500):
    """
    Assister weights for split_by_goal that give every player his share a of the assists on average, even though
    the scorer can't assist and nobody gets both assists on one goal (found by repeated adjustment).
    """
    b = a.copy()
    q1, q2 = mix["p1"] + mix["p2"], mix["p2"]
    for _ in range(iters):
        first = b * ((s / (1 - b)).sum() - s / (1 - b))                   # chance of the first assist (scorer excluded)
        second = np.zeros_like(b)
        for j in range(len(s)):                                           # second assist: scorer j and first k excluded
            pk = b / (1 - b[j])
            pk[j] = 0
            rest = 1 - b[j] - b
            w = np.where(rest > 0, pk / np.where(rest > 0, rest, 1), 0)
            second += s[j] * b * (w.sum() - w)
            second[j] -= s[j] * b[j] * (w.sum() - w)[j]
        total = q1 * first + q2 * second
        b = b * (a / (total / total.sum()))
        b = b / b.sum()
    return b


def split_by_goal(goals, s, a, mix, rng):
    """
    The slow, goal-by-goal version for one team, for the check in notebook 27. goals: team goals per season (array
    of seasons); s, a: weights (1-D, the team's skaters; a = sampling weights, see goal_by_goal_weights). Each goal:
    a scorer by s; 0, 1 or 2 assists by the mix; assisters by a, never the scorer, and the second never the first.
    Returns (goals, assists) per season x skater.
    """
    K = len(s)
    out_g = np.zeros((len(goals), K), int)
    out_a = np.zeros((len(goals), K), int)
    probs = [mix["p0"], mix["p1"], mix["p2"]]
    for i, n in enumerate(goals):
        scorers = rng.choice(K, size=n, p=s)
        n_ast = rng.choice(3, size=n, p=probs)
        np.add.at(out_g[i], scorers, 1)
        for j, k in zip(scorers, n_ast):
            taken = [j]
            for _ in range(k):
                w = a.copy()
                w[taken] = 0
                pick = rng.choice(K, p=w / w.sum())
                out_a[i, pick] += 1
                taken.append(pick)
    return out_g, out_a
