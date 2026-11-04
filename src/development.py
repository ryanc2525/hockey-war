"""
Development ranges: how far real seasons land from the projection, so a simulator can draw breakouts,
busts and collapses that match history.

    from development import residual_table, luck_fit, add_z, persistence, bucket_stats, calibrate, draw
    res = residual_table(history, bios, profiles)    # notebook 15's backtest misses, one row per player-season
    fit = luck_fit(res)                              # how much of a miss is sampling luck, by group
    res = add_z(res, fit)                            # each miss divided by its expected size

A miss (residual) = actual minus projected WAR per 60 in the target season. The projection is notebook 15's
chosen setting. Seasons are start years (2025 = 2025-26); age = age on February 1 of the season.
"""

import numpy as np
import pandas as pd

from forecast import backtest, add_age, GOALS_PER_WIN, SHORT_SEASONS
from nhle import predict as nhle_predict

# Notebook 15's chosen settings
SK_WEIGHTS, SK_REG = (1, 1, 1), 1500
G_WEIGHTS, G_REG = (1, 1, 1), 12000
AGING = {"F": "adjust_then_blend", "D": "adjust_then_blend", "G": "one_year"}
MIN_TARGET_TOI = {"skaters": 300, "goalies": 1000}
TARGETS = [t for t in range(2009, 2026) if t not in SHORT_SEASONS]
GROUPS = ["F", "D", "G"]

# Games per team in short seasons; minute cutoffs scale by games / 82.
# 2019-20: teams played 68 to 71 games before the pause, so 70 is used.
FULL_GAMES = 82
SEASON_GAMES = {1994: 48, 2012: 48, 2019: 70, 2020: 56}

EXP_MIN = {"F": 200, "D": 200, "G": 500}          # minutes for a season to count as NHL experience
EXP_GP_PRE_2008 = {"F": 15, "D": 15, "G": 10}     # before MoneyPuck (no minutes): games instead
DROP_MIN_T = 500                                  # minutes in season t to be a regular for the dropout table

AGE_ORDER = ["<=21", "22-23", "24-26", "27-29", "30-32", "33+"]
EXP_ORDER = ["0-1", "2", "3", "4-6", "7+"]
DRAFT_ORDER = ["1-5", "6-15", "16-32", "33-64", "65+", "undrafted"]
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def cutoff(season, minutes):
    """A minutes (or games) threshold scaled to the length of the season."""
    games = pd.Series(season).map(SEASON_GAMES).fillna(FULL_GAMES).to_numpy()
    return minutes * games / FULL_GAMES


def age_bucket(age):
    return pd.cut(pd.Series(age, dtype=float), [-np.inf, 21, 23, 26, 29, 32, np.inf], labels=AGE_ORDER).astype(object)


def exp_bucket(exp):
    return pd.cut(pd.Series(exp, dtype=float), [-np.inf, 1, 2, 3, 6, np.inf], labels=EXP_ORDER).astype(object)


def draft_bucket(pick):
    b = pd.cut(pd.Series(pick, dtype=float), [0, 5, 15, 32, 64, np.inf], labels=DRAFT_ORDER[:-1]).astype(object)
    return b.fillna("undrafted")


# ---------------------------------------------------------------------------
# Small weighted-statistics helpers (weights = minutes)
# ---------------------------------------------------------------------------
def wmean(x, w):
    return float(np.average(x, weights=w))


def wsd(x, w):
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def wskew(x, w):
    m, s = np.average(x, weights=w), wsd(x, w)
    return float(np.average(((x - m) / s) ** 3, weights=w))


def wquantile(x, w, qs):
    """Weighted percentiles: each value counts in proportion to its minutes."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = (np.cumsum(w) - w / 2) / w.sum()
    return np.interp(qs, cum, x)


def wcorr(x, y, w):
    c = np.cov(x, y, aweights=w)
    return float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])), float(c[0, 1] / c[0, 0])     # correlation, slope


# ---------------------------------------------------------------------------
# 1. Residual table
# ---------------------------------------------------------------------------
def experience(history, profiles, pairs):
    """
    For each (playerId, season) in pairs: NHL seasons before it that count as experience.
    2008-09 on: 200+ minutes (skaters) or 500+ (goalies), from MoneyPuck. Earlier seasons have no minutes,
    so the profiles' games are used: 15+ GP (skaters) or 10+ (goalies). Short seasons are prorated.
    """
    h = history[["playerId", "season", "group", "toi"]]
    mp = h[h["toi"] >= cutoff(h["season"], h["group"].map(EXP_MIN))][["playerId", "season"]]
    p = profiles[(profiles["league"] == "NHL") & (profiles["gameTypeId"] == 2) & (profiles["season"] < 2008)]
    p = p.groupby(["playerId", "season"], as_index=False).agg(gp=("gp", "sum"), position=("position", "first"))
    need = np.where(p["position"] == "G", EXP_GP_PRE_2008["G"], EXP_GP_PRE_2008["F"])
    early = p[p["gp"] >= cutoff(p["season"], need)][["playerId", "season"]]
    qualified = pd.concat([early, mp]).drop_duplicates()
    m = pairs[["playerId", "season"]].merge(qualified, on="playerId", how="left", suffixes=("", "_q"))
    m["counts"] = m["season_q"] < m["season"]
    return m.groupby(["playerId", "season"])["counts"].sum().rename("experience").reset_index()


def residual_table(history, bios, profiles, targets=TARGETS):
    """Notebook 15's backtest misses, one row per tested player-season, with age, experience and draft pick."""
    sk = backtest(history[history["group"] != "G"], targets, SK_WEIGHTS, SK_REG, MIN_TARGET_TOI["skaters"],
                  bios, aging=AGING)
    gk = backtest(history[history["group"] == "G"], targets, G_WEIGHTS, G_REG, MIN_TARGET_TOI["goalies"],
                  bios, aging=AGING)
    bt = pd.concat([sk, gk], ignore_index=True)
    bt["proj_war_60"] = bt["pred"] / GOALS_PER_WIN
    bt["actual_war_60"] = bt["actual"] / GOALS_PER_WIN
    bt["residual"] = bt["actual_war_60"] - bt["proj_war_60"]           # positive = beat the projection
    bt["residual_war"] = bt["residual"] * bt["toi"] / 60               # the same miss in WAR at his minutes

    info = add_age(history, bios)[["playerId", "season", "name", "age"]]
    bt = bt.merge(info, on=["playerId", "season"], how="left")
    bt = bt.merge(bios[["playerId", "draft_overall"]], on="playerId", how="left")
    bt = bt.merge(experience(history, profiles, bt), on=["playerId", "season"], how="left")
    bt["age_bucket"] = age_bucket(bt["age"]).to_numpy()
    bt["exp_bucket"] = exp_bucket(bt["experience"]).to_numpy()
    bt["draft_bucket"] = draft_bucket(bt["draft_overall"]).to_numpy()
    cols = ["playerId", "name", "season", "group", "age", "experience", "draft_overall", "age_bucket",
            "exp_bucket", "draft_bucket", "toi", "proj_war_60", "actual_war_60", "residual", "residual_war"]
    return bt[cols].sort_values(["season", "group", "playerId"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Luck vs. real change, centering, scaled misses
# ---------------------------------------------------------------------------
def luck_fit(res, col="residual"):
    """
    Per group: variance of a miss = a + b / minutes (minutes-weighted least squares of miss^2 on 1/minutes).
    a = spread that more ice time doesn't remove; b / minutes = sampling luck, which shrinks with minutes.
    """
    rows = {}
    for g, d in res.groupby("group"):
        w = d["toi"].to_numpy()
        X = np.column_stack([np.ones(len(d)), 1 / w])
        y = d[col].to_numpy() ** 2
        a, b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        rows[g] = {"a": a, "b": b}
    return pd.DataFrame(rows).T.reindex([g for g in GROUPS if g in rows])


def luck_share(fit, group, minutes):
    """Share of a miss's variance that is sampling luck, at a given number of minutes."""
    a, b = fit.loc[group, "a"], fit.loc[group, "b"]
    return (b / minutes) / (a + b / minutes)


def expected_sd(fit, group, minutes):
    g = pd.Series(group)
    return np.sqrt(g.map(fit["a"]).to_numpy() + g.map(fit["b"]).to_numpy() / np.asarray(minutes, dtype=float))


def add_z(res, fit, means=None):
    """
    Adds z = miss / its expected size at his minutes, then centered versions of both (each group's
    minutes-weighted mean set to 0). means: group means to subtract (default: from res itself).
    """
    r = res.copy()
    r["z"] = r["residual"] / expected_sd(fit, r["group"], r["toi"])
    if means is None:
        means = group_means(r)
    r["residual_c"] = r["residual"] - r["group"].map(means["residual"])
    r["z_c"] = r["z"] - r["group"].map(means["z"])
    return r


def group_means(res):
    """Minutes-weighted mean miss per group (the projection's overall bias), in WAR/60 and in z."""
    return pd.DataFrame({c: res.groupby("group").apply(lambda d: wmean(d[c], d["toi"]), include_groups=False)
                         for c in ["residual", "z"] if c in res})


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------
def season_pairs(res, col="residual_c"):
    """Players tested in back-to-back target seasons: miss in season t next to miss in t+1."""
    a = res[["playerId", "season", "group", "age_bucket", "toi", col]]
    b = a[["playerId", "season", "toi", col]].assign(season=a["season"] - 1)
    p = a.merge(b, on=["playerId", "season"], suffixes=("", "_next"))
    p["weight"] = np.minimum(p["toi"], p["toi_next"])     # a pair is only as informative as its shorter season
    return p


def persistence(res, col="residual_c", by_age=True, min_pairs=50):
    """How much of a miss shows up again next season: minutes-weighted correlation and slope."""
    p = season_pairs(res, col)
    rows = []
    for g, d in p.groupby("group"):
        r, s = wcorr(d[col], d[f"{col}_next"], d["weight"])
        rows.append({"group": g, "age_bucket": "all", "pairs": len(d), "corr": r, "slope": s})
        if by_age:
            for ab in AGE_ORDER:
                e = d[d["age_bucket"] == ab]
                if len(e) >= min_pairs:
                    r, s = wcorr(e[col], e[f"{col}_next"], e["weight"])
                    rows.append({"group": g, "age_bucket": ab, "pairs": len(e), "corr": r, "slope": s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4-5. Buckets and what moves the range
# ---------------------------------------------------------------------------
def merge_map(counts, order, merge_below=50):
    """
    Merge buckets with fewer than merge_below player-seasons into their smaller neighbour, until none is small.
    Returns {old label: new label} and a list of plain-language notes.
    """
    groups = [[lab] for lab in order if counts.get(lab, 0) > 0]
    size = lambda grp: sum(counts.get(lab, 0) for lab in grp)
    while len(groups) > 1 and min(size(g) for g in groups) < merge_below:
        i = min(range(len(groups)), key=lambda k: size(groups[k]))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if size(groups[i - 1]) <= size(groups[i + 1]) else i + 1
        lo, hi = sorted([i, j])
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    mapping, notes = {}, []
    for grp in groups:
        label = grp[0] if len(grp) == 1 else " / ".join(grp)
        if len(grp) > 1:
            notes.append(f"merged {', '.join(f'{lab} ({counts.get(lab, 0)})' for lab in grp)}")
        for lab in grp:
            mapping[lab] = label
    return mapping, notes


def bucket_stats(res, bucket_col, order, groups=("F", "D"), merge_below=50, n_boot=1000, seed=17):
    """
    For each group and bucket: player-seasons, raw and centered mean miss (WAR/60), SD, percentiles and skew
    of the centered miss, centered z percentiles, and two noise checks:
    - mean_flag: the bucket's centered mean is more than 2 standard errors from 0 (its group's mean)
    - sd_flag: its z SD is outside the 90% range of z SDs from random draws of the same size from its group
    Returns (stats, mapping per group, notes).
    """
    rng = np.random.default_rng(seed)
    rows, maps, notes = [], {}, []
    for g in groups:
        d = res[res["group"] == g]
        mapping, n = merge_map(d[bucket_col].value_counts().to_dict(), order, merge_below)
        maps[g] = mapping
        notes += [f"{g}: {x}" for x in n]
        lab = d[bucket_col].map(mapping)
        zg, wg = d["z_c"].to_numpy(), d["toi"].to_numpy()
        for b in dict.fromkeys(mapping.values()):
            e = d[lab == b]
            w = e["toi"].to_numpy()
            x = e["residual_c"].to_numpy()
            n_eff = w.sum() ** 2 / (w ** 2).sum()
            se = wsd(x, w) / np.sqrt(n_eff)
            boot = [wsd(zg[i], wg[i]) for i in (rng.integers(0, len(zg), len(e)) for _ in range(n_boot))]
            lo, hi = np.percentile(boot, [5, 95])
            zsd = wsd(e["z_c"].to_numpy(), w)
            row = {"group": g, "bucket": b, "n": len(e), "mean_raw": wmean(e["residual"], w), "mean": wmean(x, w),
                   "se": se, "mean_flag": abs(wmean(x, w)) > 2 * se, "sd": wsd(x, w), "skew": wskew(x, w),
                   "z_sd": zsd, "z_sd_lo": lo, "z_sd_hi": hi, "sd_flag": not (lo <= zsd <= hi)}
            row.update({f"p{int(q * 100)}": v for q, v in zip(QS, wquantile(x, w, QS))})
            row.update({f"z_p{int(q * 100)}": v for q, v in zip(QS, wquantile(e["z_c"], w, QS))})
            rows.append(row)
    return pd.DataFrame(rows), maps, notes


# ---------------------------------------------------------------------------
# 6. NHLe for young players
# ---------------------------------------------------------------------------
def season_nhle(careers, factors, multipliers, min_gp=20):
    """
    NHLe (NHL points per game) of every pre-NHL season: non-NHL seasons with min_gp+ GP before the player's
    first NHL season with 20+ GP, x the chosen factor (forward/defense factor where one exists) x the age
    multiplier. Leagues without a factor are skipped.
    """
    first = careers[(careers["league"] == "NHL") & (careers["gp"] >= 20)].groupby("playerId")["season"].min()
    pre = careers[(careers["league"] != "NHL") & (careers["gp"] >= min_gp)].copy()
    pre = pre[pre["season"] < pre["playerId"].map(first).fillna(np.inf)]
    chosen = factors[factors["chosen"]]
    allf = chosen[chosen["group"] == "all"].set_index("league")["factor"]
    pos = chosen[chosen["group"].isin(["F", "D"])][["league", "group", "factor"]]
    mult = multipliers.set_index("age_bucket")["multiplier"]
    rows = pd.DataFrame({"prior_league": pre["league"].to_numpy(), "group": pre["group"].to_numpy(),
                         "prior_ppg": (pre["points"] / pre["gp"]).to_numpy(), "prior_age": pre["age"].to_numpy()},
                        index=pre.index)
    pre["nhle_ppg"] = nhle_predict(rows, allf, mult, by_group=pos).to_numpy()
    return pre.dropna(subset=["nhle_ppg"])


NHLE_COLS = {"season": "nhle_season", "league": "nhle_league", "gp": "nhle_gp", "points": "nhle_points",
             "age": "nhle_age"}


def best_nhle(careers, factors, multipliers, rule="best20"):
    """
    One pre-NHL NHLe per skater (NHL points per game), by one of two rules:
    - "best20" (default, notebooks 17-18): his best season with 20+ GP.
    - "last2" (notebook 19): games-weighted NHLe of his last two pre-NHL seasons, counting seasons with 10+ GP.
      Best-season picks lucky years and over-predicts; the last two seasons are the better guide.
    For "last2", nhle_season/league/age describe his latest season and nhle_gp/points are totals.
    """
    if rule == "best20":
        pre = season_nhle(careers, factors, multipliers, 20).sort_values("nhle_ppg", ascending=False)
        best = pre.groupby("playerId").head(1)
        return best[["playerId", "season", "league", "gp", "points", "age", "nhle_ppg"]] \
            .rename(columns=NHLE_COLS).reset_index(drop=True)
    if rule == "last2":
        pre = season_nhle(careers, factors, multipliers, 10)
        latest_two = pre.groupby("playerId")["season"].transform(lambda s: s >= sorted(s.unique())[-2:][0])
        pre = pre[latest_two]

        def summary(d):
            last = d[d["season"] == d["season"].max()].sort_values("gp").iloc[-1]
            return pd.Series({"season": last["season"], "league": last["league"], "gp": d["gp"].sum(),
                              "points": d["points"].sum(), "age": last["age"],
                              "nhle_ppg": np.average(d["nhle_ppg"], weights=d["gp"])})
        out = pre.groupby("playerId").apply(summary, include_groups=False).reset_index()
        return out.rename(columns=NHLE_COLS)
    raise ValueError(f"unknown rule {rule}")


def nhle_rules(careers, factors, multipliers, league_avg, shrink_games=20):
    """
    The four pre-NHL summaries compared in notebook 19, one row per skater (NHL points per game):
    (i) best season with 20+ GP; (ii) best season with 10+ GP after shrinking it toward the games-weighted
    NHLe of his other pre-NHL seasons (or league_avg[league] if he has none) with shrink_games games of weight;
    (iii) most recent season with 20+ GP (most games if two leagues); (iv) games-weighted NHLe of his last two
    pre-NHL seasons (10+ GP).
    """
    pre = season_nhle(careers, factors, multipliers, 10)

    def rules(d):
        out = {}
        d20 = d[d["gp"] >= 20]
        out["i"] = d20["nhle_ppg"].max() if len(d20) else np.nan
        gp, x = d["gp"].to_numpy(dtype=float), d["nhle_ppg"].to_numpy()
        shrunk = []
        for k in range(len(d)):
            others = np.arange(len(d)) != k
            ref = (np.average(x[others], weights=gp[others]) if others.any()
                   else league_avg.get(d["league"].iloc[k], np.nan))
            shrunk.append((gp[k] * x[k] + shrink_games * ref) / (gp[k] + shrink_games))
        out["ii"] = np.nanmax(shrunk) if np.isfinite(shrunk).any() else np.nan
        out["iii"] = (d20[d20["season"] == d20["season"].max()].sort_values("gp")["nhle_ppg"].iloc[-1]
                      if len(d20) else np.nan)
        last2 = d[d["season"] >= sorted(d["season"].unique())[-2:][0]]
        out["iv"] = np.average(last2["nhle_ppg"], weights=last2["gp"])
        return pd.Series(out)
    return pre.groupby("playerId").apply(rules, include_groups=False)


def league_average_nhle(careers, factors, multipliers, last_season, min_gp=10):
    """Games-weighted NHLe of every pre-NHL player-season in each league, seasons through last_season."""
    pre = season_nhle(careers, factors, multipliers, min_gp)
    pre = pre[pre["season"] <= last_season]
    return pre.groupby("league").apply(lambda d: np.average(d["nhle_ppg"], weights=d["gp"]), include_groups=False)


def wls(y, X, w):
    """Weighted least squares: coefficients, standard errors and weighted R^2 (weights scaled to mean 1)."""
    w = np.asarray(w, dtype=float) / np.mean(w)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    e = y - X @ beta
    sigma2 = (w * e ** 2).sum() / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtW @ X)))
    r2 = 1 - (w * e ** 2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()
    return beta, se, r2


# ---------------------------------------------------------------------------
# 7. Dropouts
# ---------------------------------------------------------------------------
def dropouts(history, bios):
    """
    Regulars (500+ minutes in season t, prorated) and whether they nearly vanish in t+1: under 200 minutes
    (skaters) or 500 (goalies), prorated, or no NHL games at all.
    """
    h = add_age(history, bios)
    reg = h[h["toi"] >= cutoff(h["season"], DROP_MIN_T)]
    reg = reg[reg["season"] < h["season"].max()]                      # need a season t+1
    nxt = h[["playerId", "season", "toi"]].assign(season=h["season"] - 1).rename(columns={"toi": "toi_next"})
    d = reg.merge(nxt, on=["playerId", "season"], how="left")
    d["toi_next"] = d["toi_next"].fillna(0)
    need = cutoff(d["season"] + 1, d["group"].map(EXP_MIN))
    d["dropout"] = d["toi_next"] < need
    d["age_bucket"] = age_bucket(d["age"]).to_numpy()
    return d[["playerId", "name", "season", "group", "age", "age_bucket", "toi", "toi_next", "dropout"]]


# ---------------------------------------------------------------------------
# 8. Calibration
# ---------------------------------------------------------------------------
RANGES = {"5-95": (0.05, 0.95, 0.90), "10-90": (0.10, 0.90, 0.80), "25-75": (0.25, 0.75, 0.50)}


def calibrate(res, train_end=2017, merge_below=50):
    """
    Fit ranges on target seasons through train_end, test on later seasons. Four versions:
    raw or scaled (z) misses, each uncentered or centered (training group mean moved to 0).
    Returns one row per test player-season with, for each version and range, whether the actual miss fell inside.
    Everything used to build the ranges (luck fit, group means, bucket merges) comes from training seasons only.
    """
    train, test = res[res["season"] <= train_end].copy(), res[res["season"] > train_end].copy()
    fit = luck_fit(train)
    means = group_means(train.assign(z=train["residual"] / expected_sd(fit, train["group"], train["toi"])))
    train = add_z(train, fit, means)
    test = add_z(test, fit, means)
    out = test[["playerId", "season", "group", "age_bucket", "toi", "residual"]].copy()
    for g in res["group"].unique():
        tr_g = train[train["group"] == g]
        mapping, _ = merge_map(tr_g["age_bucket"].value_counts().to_dict(), AGE_ORDER, merge_below)
        tr_lab = tr_g["age_bucket"].map(mapping)
        te = test["group"] == g
        te_lab = test.loc[te, "age_bucket"].map(mapping)
        sd_te = expected_sd(fit, test.loc[te, "group"], test.loc[te, "toi"])
        for b in dict.fromkeys(mapping.values()):
            src, idx = tr_g[tr_lab == b], te_lab.index[te_lab == b]
            sdb = pd.Series(sd_te, index=test.index[te]).loc[idx].to_numpy()
            actual = test.loc[idx, "residual"].to_numpy()
            for version, col, scaled, shift in [("raw", "residual", False, 0.0),
                                                ("raw, centered", "residual", False, means.loc[g, "residual"]),
                                                ("scaled", "z", True, 0.0),
                                                ("scaled, centered", "z", True, means.loc[g, "z"])]:
                for name, (lo_q, hi_q, _) in RANGES.items():
                    lo, hi = wquantile(src[col] - shift, src["toi"], [lo_q, hi_q])
                    if scaled:
                        lo, hi = lo * sdb, hi * sdb
                    out.loc[idx, f"{version}|{name}"] = (actual >= lo) & (actual <= hi)
    return out


def coverage(cal, by):
    """Share of test outcomes inside each range, per version, grouped by `by`."""
    cols = [c for c in cal.columns if "|" in c]
    t = cal.groupby(by)[cols].mean()
    t.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in cols], names=["version", "range"])
    t.insert(0, ("", "n"), cal.groupby(by).size())
    return t


def calibration_score(cal):
    """Average distance from the target share over the three ranges and three groups (lower is better)."""
    out = {}
    for version in dict.fromkeys(c.split("|")[0] for c in cal.columns if "|" in c):
        miss = [abs(cal.loc[cal["group"] == g, f"{version}|{name}"].mean() - target)
                for g in cal["group"].unique() for name, (_, _, target) in RANGES.items()]
        out[version] = float(np.mean(miss))
    return pd.Series(out)


# ---------------------------------------------------------------------------
# 9. Drawing a season
# ---------------------------------------------------------------------------
def draw_pools(res, key_col, group_col="group"):
    """
    Centered z values and minutes for every (pool group, bucket) key: what a draw samples from.
    group_col lets some buckets pool forwards and defensemen together (z has no units, so they mix fairly).
    """
    return {k: (d["z_c"].to_numpy(), d["toi"].to_numpy()) for k, d in res.groupby([group_col, key_col])}


def draw(players, pools, fit, carry, n=1000, seed=2026):
    """
    n simulated seasons for each player (rows: playerId, group, pool_group, pool_key, minutes, proj_war_60).
    A draw picks a centered z from the player's pool (chance in proportion to minutes, so it matches the
    minutes-weighted percentiles), converts it to WAR/60 at his minutes, and splits it into three parts,
    each its average share of the miss:
    - luck: sampling luck (share b/minutes of the variance); this season only
    - persistent: carry[group] x miss; carries into next year's true talent
    - one_year: the rest (role, linemates, health); this season only
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in players.itertuples(index=False):
        z, w = pools[(p.pool_group, p.pool_key)]
        zs = rng.choice(z, size=n, p=w / w.sum())
        miss = zs * expected_sd(fit, [p.group], [p.minutes])[0]
        luck = luck_share(fit, p.group, p.minutes) * miss
        persistent = carry.get(p.group, 0.0) * miss
        out.append(pd.DataFrame({
            "playerId": p.playerId, "draw": np.arange(n), "miss_war_60": miss, "luck": luck,
            "one_year": miss - luck - persistent, "persistent": persistent,
            "war_60": p.proj_war_60 + miss, "war": (p.proj_war_60 + miss) * p.minutes / 60}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 10. Player-specific prior (notebook 18)
# ---------------------------------------------------------------------------
# What a skater is regressed toward, instead of his position's average: an expected GAR/60 from his best
# pre-NHL scoring (NHLe), draft position, age and NHL experience, fit on what such players did in the NHL.
PRIOR_INPUTS = ["nhle_ppg", "log_pick", "undrafted", "age", "experience"]
DIAG_INPUTS = ["age", "experience"]          # diagnostic version (d): no NHLe, no draft
EXP_CAP = 7                                  # experience beyond 7 seasons counts as 7
PRIOR_MIN_TOI = 300                          # skater-seasons used to fit the prior
VERSIONS = {"a": "everyone", "b": "3 or fewer NHL seasons", "c": "25 or under",
            "d": "everyone, age + experience only"}


def training_nhle(careers, end):
    """
    Notebook 16's chosen NHLe (method A x age multiplier, forward/defense factors), refit with data through
    `end` only, in the format best_nhle reads: (factors table, age multiplier table).
    """
    from nhle import next_season_pairs, method_a, position_factors, rookie_rows, age_multipliers
    pairs = next_season_pairs(careers, max_to_season=end)
    a = method_a(pairs).set_index("league")
    pos = position_factors(pairs, a.index[a["path_type"] == "direct"])
    rk = rookie_rows(careers)
    rk = rk[rk["season"] <= end]
    mult = age_multipliers(rk, nhle_predict(rk, a["factor"], by_group=pos))
    factors = pd.concat([a.reset_index()[["league", "factor", "pairs", "path_type"]].assign(group="all"),
                         pos[["league", "group", "factor", "pairs"]].assign(path_type="direct")],
                        ignore_index=True).assign(chosen=True)
    multipliers = mult.rename("multiplier").rename_axis("age_bucket").reset_index()
    return factors, multipliers


def skater_features(pairs, history, bios, profiles, nhle_best):
    """
    Prior inputs for each (playerId, season, group) in pairs: age (Feb 1), NHL experience before that season
    (capped at EXP_CAP), log of overall draft pick (0 if undrafted), undrafted flag, and best pre-NHL NHLe.
    """
    f = add_age(pairs[["playerId", "season", "group"]].drop_duplicates(["playerId", "season"]), bios)
    f = f.merge(experience(history, profiles, f), on=["playerId", "season"], how="left")
    f["experience"] = f["experience"].fillna(0).clip(upper=EXP_CAP)
    f = f.merge(bios[["playerId", "draft_overall"]], on="playerId", how="left")
    f["undrafted"] = f["draft_overall"].isna().astype(float)
    f["log_pick"] = np.log(f["draft_overall"].fillna(1.0))        # undrafted: 0 here, the flag carries it
    f = f.merge(nhle_best[["playerId", "nhle_ppg"]], on="playerId", how="left")
    return f


def _design(rows, inputs, fill):
    X = rows[inputs].astype(float).copy()
    if "nhle_ppg" in inputs:
        X["nhle_ppg"] = X["nhle_ppg"].fillna(fill)                  # no pre-NHL season: position average
    X.insert(0, "constant", 1.0)
    return X


def fit_prior(rows, inputs=PRIOR_INPUTS):
    """
    Per position (F, D): minutes-weighted least squares of actual GAR/60 on the inputs.
    rows: skater-seasons with gar_60, toi, group and the inputs. Returns {group: model dict}.
    """
    models = {}
    for g, d in rows.groupby("group"):
        has = d["nhle_ppg"].notna()
        fill = float(np.average(d.loc[has, "nhle_ppg"], weights=d.loc[has, "toi"]))
        X = _design(d, inputs, fill)
        beta, se, r2 = wls(d["gar_60"], X, d["toi"])
        models[g] = {"inputs": list(inputs), "coef": pd.Series(beta, index=X.columns),
                     "se": pd.Series(se, index=X.columns), "r2": r2, "n": len(d), "fill": fill}
    return models


def apply_prior(rows, models):
    """Expected GAR/60 from a fitted prior model, for every row whose group has a model (NaN otherwise)."""
    out = pd.Series(np.nan, index=rows.index)
    for g, m in models.items():
        sel = rows["group"] == g
        if sel.any():
            out[sel] = _design(rows[sel], m["inputs"], m["fill"]).to_numpy() @ m["coef"].to_numpy()
    return out


def model_table(models):
    """Coefficients and standard errors side by side, one column pair per position."""
    return pd.concat({g: pd.DataFrame({"coef": m["coef"], "se": m["se"]}) for g, m in models.items()}, axis=1)


def eligible(rows, version):
    """Who gets the prior under each version (see VERSIONS)."""
    if version in ("a", "d"):
        return pd.Series(True, index=rows.index)
    if version == "b":
        return rows["experience"] <= 3
    if version == "c":
        return rows["age"] <= 25
    raise ValueError(version)


def candidates(history, targets):
    """Every skater the projection covers for each target season (a season in target-3 to target-1)."""
    sk = history[history["group"] != "G"].sort_values("season")
    out = []
    for t in targets:
        p = sk[sk["season"].between(t - 3, t - 1)].groupby("playerId")["group"].last().reset_index()
        out.append(p.assign(season=t))
    return pd.concat(out, ignore_index=True)


def fit_rows(history, features, targets, min_toi=PRIOR_MIN_TOI):
    """Skater-seasons used to fit the prior: min_toi+ minutes in a target season (rookies included)."""
    sk = history[(history["group"] != "G") & history["season"].isin(targets) & (history["toi"] >= min_toi)]
    sk = sk.assign(gar_60=sk["gar"] / sk["toi"] * 60)[["playerId", "season", "name", "toi", "gar_60"]]
    return sk.merge(features, on=["playerId", "season"], how="inner")


def first_seasons(history):
    """Each player's first season in our NHL data (2008-09 on)."""
    return history.groupby("playerId")["season"].min()


# ---------------------------------------------------------------------------
# 11. Prospects (notebook 19)
# ---------------------------------------------------------------------------
LEAGUE_GROUPS = {
    "CHL": {"OHL", "WHL", "QMJHL"},
    "NCAA": {"NCAA"},
    "KHL": {"KHL", "VHL", "MHL"},
    "AHL": {"AHL"},
    "Europe": {"SHL", "Liiga", "NL", "DEL", "HockeyAllsvenskan", "Czechia", "ICEHL", "Mestis", "SL", "Slovakia",
               "J20 Nationell", "U20 SM-liiga", "Czechia2", "DEL2", "Norway", "Denmark", "EIHL", "Ligue Magnus"},
}


def league_group(league):
    """CHL, NCAA, KHL (with VHL and MHL), AHL, Europe, or other (USHL, BCHL, ECHL, ...)."""
    for name, leagues in LEAGUE_GROUPS.items():
        if league in leagues:
            return name
    return "other" if isinstance(league, str) else np.nan


def prospect_table(drafts, profiles, projected_ids, season=2025, max_age=24, excluded=()):
    """
    Who is a prospect for season+1: drafted, not already in the NHL projection, played a regular-season game
    in `season` in some league (tournaments in `excluded` don't count), and max_age or younger on Feb 1 of
    season+2 (2027 for 2026-27). Returns (prospects, exclusion counts). Each prospect gets his `season` league
    with the most games, plus his games and points there.
    """
    from forecast import season_age
    steps = {"drafted picks with a playerId": len(drafts)}
    have = set(profiles["playerId"])
    d = drafts[drafts["playerId"].isin(have)]
    steps["  no profile (download missing or failed)"] = len(drafts) - len(d)
    reg = d["playerId"].isin(projected_ids)
    steps["  already in the NHL projection (played 2023-24 to 2025-26)"] = int(reg.sum())
    d = d[~reg]
    played = profiles[(profiles["season"] == season) & (profiles["gameTypeId"] == 2) & (profiles["gp"] > 0)
                      & ~profiles["league"].isin(excluded)]
    main = (played.groupby(["playerId", "league"], as_index=False)[["gp", "points"]].sum()
            .sort_values("gp", ascending=False).groupby("playerId").head(1).set_index("playerId"))
    ok = d["playerId"].isin(main.index)
    steps[f"  didn't play a regular-season game in {season}-{str(season + 1)[-2:]}"] = int((~ok).sum())
    d = d[ok].copy()
    d["age"] = season_age(pd.to_datetime(d["birth_date"]), np.full(len(d), season + 1))
    young = d["age"] <= max_age
    steps[f"  older than {max_age} on Feb 1, {season + 2}"] = int((~young).sum())
    d = d[young].copy()
    d = d.drop_duplicates("playerId", keep="last")             # re-drafted players: keep the latest pick
    d = d.join(main.rename(columns={"league": "league_last", "gp": "gp_last", "points": "points_last"}), on="playerId")
    steps["prospects"] = len(d)
    return d.reset_index(drop=True), pd.Series(steps, name="players")


def prospect_range(proj_war_60, group, rookie_rows, luck, minutes, qs=(10, 50, 90)):
    """10th/50th/90th WAR/60 for a projection, from the rookie z percentiles at a given number of minutes."""
    r = rookie_rows.set_index("group")
    sd = np.sqrt(luck.loc[group, "a"] + luck.loc[group, "b"] / minutes)
    return {f"war_60_p{q}": proj_war_60 + r.loc[group, f"z_p{q}"] * sd for q in qs}
