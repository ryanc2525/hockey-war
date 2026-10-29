"""
Forecast building blocks: true-talent WAR and aging curves, for skaters and goalies.

    from forecast import history_table, true_talent, aging_curve, project, backtest, rmse
    history = history_table(skater_war, goalie_war)          # notebook 14's two tables
    tt = true_talent(history, 2026, weights=(1, 1, 1), reg_minutes=1000)
    raw, curve = aging_curve(history, bios)                  # bios = edge_data.build_bio_table(...)
    proj = project(history, bios, 2026, (1, 1, 1), 1000, aging="adjust_then_blend", curve=curve)

True talent = a player's GAR per 60 over up to 3 prior seasons (weight x minutes), blended with reg_minutes
of his group's average (F, D or G) from the most recent prior season. Rates are per 60 minutes;
GAR is in goals and WAR = GAR / GOALS_PER_WIN.
season = MoneyPuck start year (2025 = 2025-26). Age in a season = age on February 1 of its end year.
"""

import numpy as np
import pandas as pd

from war_model import V04_PARAMS

GOALS_PER_WIN = V04_PARAMS["goals_per_win"]
KEY = ["playerId", "season"]
GROUPS = ["F", "D", "G"]
MIN_PAIR_TOI = {"F": 200, "D": 200, "G": 500}      # minutes in both seasons for an aging pair
AGE_MIN, AGE_MAX = 20, 37                          # ages outside this range use the edge value
SHORT_SEASONS = {2012, 2019, 2020}                 # 48 games, stopped in March, 56 games
AGING_VERSIONS = ["none", "one_year", "adjust_then_blend"]


# ---------------------------------------------------------------------------
# 1. History and ages
# ---------------------------------------------------------------------------
def history_table(skater_war, goalie_war):
    """One row per player-season, skaters and goalies: playerId, season, name, team, position, group, toi, gar, war."""
    cols = ["playerId", "season", "name", "team", "position", "group", "toi", "gar", "war"]
    sk = skater_war.rename(columns={"all_toi": "toi", "pos_group": "group"})
    gk = goalie_war.assign(position="G", group="G")
    return pd.concat([sk[cols], gk[cols]], ignore_index=True)


def season_age(birth, season):
    """Age on February 1 of the season's end year (2025 -> Feb 1, 2026). birth = datetimes, season = start years."""
    birth = pd.to_datetime(pd.Series(birth).reset_index(drop=True), errors="coerce")
    season = pd.Series(season).reset_index(drop=True)
    not_yet = (birth.dt.month > 2) | ((birth.dt.month == 2) & (birth.dt.day > 1))   # birthday after Feb 1
    return (season + 1 - birth.dt.year - not_yet.astype(int)).to_numpy(dtype=float)


def _birth_dates(bios, player_ids):
    """Birth date for each id in player_ids (NaT when unknown or when bios is None)."""
    if bios is None:
        return pd.Series(pd.NaT, index=range(len(player_ids)), dtype="datetime64[ns]")
    birth = pd.to_datetime(bios.set_index("playerId")["birth_date"], errors="coerce")
    return birth.reindex(player_ids).reset_index(drop=True)


def add_age(history, bios):
    """history plus age (NaN when the birth date is unknown)."""
    h = history.copy()
    h["age"] = season_age(_birth_dates(bios, h["playerId"]), h["season"])
    return h


# ---------------------------------------------------------------------------
# 2. True talent
# ---------------------------------------------------------------------------
def group_average(history, season):
    """TOI-weighted GAR/60 of each group (F, D, G) in one season."""
    s = history[history["season"] == season].groupby("group")
    return s["gar"].sum() / s["toi"].sum() * 60


def _blend(prior, history, target_season, weights, reg_minutes):
    """Weighted, regressed GAR/60 for every player in prior (rows from target-1 to target-3)."""
    t = target_season
    w = prior["season"].map({t - 1: weights[0], t - 2: weights[1], t - 3: weights[2]})
    g = prior.assign(wg=w * prior["gar"], wt=w * prior["toi"]).groupby("playerId")
    out = g.agg(wg=("wg", "sum"), wt=("wt", "sum"), seasons_used=("season", "size"), minutes_used=("toi", "sum"),
                last_season=("season", "max"))
    latest = prior.sort_values("season").groupby("playerId")[["name", "team", "position", "group"]].last()
    out = latest.join(out)
    out["group_gar_60"] = out["group"].map(group_average(history, t - 1))
    out["tt_gar_60"] = (out["wg"] + reg_minutes * out["group_gar_60"] / 60) / (out["wt"] + reg_minutes) * 60
    out["tt_war_60"] = out["tt_gar_60"] / GOALS_PER_WIN
    return out.drop(columns=["wg", "wt"])


def true_talent(history, target_season, weights=(1, 1, 1), reg_minutes=1000):
    """True-talent WAR/60 going into target_season, for every player with a season in target-1 to target-3.
    weights[0] is the most recent season. With reg_minutes=0, a player whose seasons all have weight 0 gets NaN."""
    prior = history[history["season"].between(target_season - 3, target_season - 1)]
    return _blend(prior, history, target_season, weights, reg_minutes)


# ---------------------------------------------------------------------------
# 3. Aging curves (delta method)
# ---------------------------------------------------------------------------
def aging_pairs(history, bios, exclude_season=None):
    """Same player, same group, seasons s and s+1, MIN_PAIR_TOI+ minutes both. delta = WAR/60 change,
    credited to his age in season s (clipped to AGE_MIN..AGE_MAX); weight = harmonic mean of the two TOIs.
    exclude_season drops every pair that includes that season (so a backtest never sees its target)."""
    h = add_age(history, bios)
    h["rate"] = h["war"] / h["toi"] * 60
    p = h.merge(h[KEY + ["group", "toi", "rate"]].assign(season=h["season"] - 1), on=KEY, suffixes=("_1", "_2"))
    p = p[(p["group_1"] == p["group_2"]) & p["age"].notna()]
    p = p[(p["toi_1"] >= p["group_1"].map(MIN_PAIR_TOI)) & (p["toi_2"] >= p["group_1"].map(MIN_PAIR_TOI))]
    if exclude_season is not None:
        p = p[(p["season"] != exclude_season) & (p["season"] + 1 != exclude_season)]
    return pd.DataFrame({
        "playerId": p["playerId"], "season": p["season"], "group": p["group_1"],
        "age": p["age"].clip(AGE_MIN, AGE_MAX).astype(int),
        "delta": p["rate_2"] - p["rate_1"],
        "weight": 2 * p["toi_1"] * p["toi_2"] / (p["toi_1"] + p["toi_2"]),
    }).reset_index(drop=True)


def age_mix(history, bios, exclude_season=None):
    """Total minutes by group and age (clipped to AGE_MIN..AGE_MAX), over every player-season with a birth date."""
    h = add_age(history, bios)
    h = h[h["age"].notna()]
    if exclude_season is not None:
        h = h[h["season"] != exclude_season]
    return h.groupby(["group", h["age"].clip(AGE_MIN, AGE_MAX).astype(int).rename("age")])["toi"].sum()


def aging_curve(history, bios, exclude_season=None, center=True):
    """(raw, smoothed). raw: group, age, delta (weighted mean WAR/60 change from this age to the next), pairs, weight.
    smoothed: group, age (AGE_MIN..AGE_MAX), delta_uncentered (weighted quadratic fit of the raw points, per group),
    shift, and delta. With center=True, delta = delta_uncentered - shift, where shift is the curve's average over the
    league's actual age mix (minutes at each age), so the league as a whole neither gains nor loses from aging.
    The raw delta method curves sit below zero because players who keep their jobs tend to regress after a lucky
    season; true talent already handles that regression, so without centering it would be counted twice."""
    p = aging_pairs(history, bios, exclude_season)
    raw = (p.assign(wd=p["delta"] * p["weight"])
            .groupby(["group", "age"])
            .agg(wd=("wd", "sum"), weight=("weight", "sum"), pairs=("delta", "size"))
            .reset_index())
    raw["delta"] = raw["wd"] / raw["weight"]
    raw = raw[["group", "age", "delta", "pairs", "weight"]]

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    mix = age_mix(history, bios, exclude_season)
    smoothed = []
    for group, r in raw.groupby("group"):
        coef = np.polyfit(r["age"], r["delta"], 2, w=np.sqrt(r["weight"]))   # polyfit's w multiplies residuals
        fit = np.polyval(coef, ages)
        shift = np.average(fit, weights=mix[group].reindex(ages, fill_value=0)) if center else 0.0
        smoothed.append(pd.DataFrame({"group": group, "age": ages, "delta_uncentered": fit,
                                      "shift": shift, "delta": fit - shift}))
    return raw, pd.concat(smoothed, ignore_index=True)


def _curve_lookup(curve, group, age):
    """Smoothed delta for each (group, age); ages clipped to AGE_MIN..AGE_MAX. NaN where age or group is unknown."""
    group, age = np.asarray(group), np.asarray(age, dtype=float)
    table = curve.pivot(index="age", columns="group", values="delta")
    out = np.full(len(age), np.nan)
    for g in table.columns:
        m = (group == g) & ~np.isnan(age)
        out[m] = table[g].to_numpy()[(np.clip(age[m], AGE_MIN, AGE_MAX) - AGE_MIN).astype(int)]
    return out


def cumulative_aging(curve, group, from_age, to_age):
    """Total smoothed WAR/60 change from from_age to to_age: delta(from_age) + ... + delta(to_age - 1).
    0 where the age is unknown (no adjustment)."""
    from_age, to_age = np.asarray(from_age, dtype=float), np.asarray(to_age, dtype=float)
    steps = np.nan_to_num(to_age - from_age)
    total = np.zeros(len(from_age))
    for k in range(int(steps.max(initial=0))):
        d = _curve_lookup(curve, group, from_age + k)
        total += np.where((k < steps) & ~np.isnan(d), d, 0)
    return total


# ---------------------------------------------------------------------------
# 4. Projection and backtest
# ---------------------------------------------------------------------------
def project(history, bios, target_season, weights=(1, 1, 1), reg_minutes=1000, aging="none", curve=None):
    """True talent going into target_season, with an optional aging adjustment:
    "none": true talent as is.
    "one_year": true talent + the smoothed change at his age in target-1.
    "adjust_then_blend": each prior season's GAR/60 is moved to his target-season age (cumulative smoothed
        changes), then the seasons are weighted and regressed as usual. The group average isn't aged.
    aging can also be a dict {group: version}. aging_change = projected minus unadjusted true talent (WAR/60)."""
    if not isinstance(aging, dict):
        aging = {g: aging for g in GROUPS}
    h = add_age(history, bios)
    prior = h[h["season"].between(target_season - 3, target_season - 1)].copy()
    out = _blend(prior, h, target_season, weights, reg_minutes)
    out["age_next"] = season_age(_birth_dates(bios, out.index), np.full(len(out), target_season))
    out["has_age"] = out["age_next"].notna()
    out["played_last_season"] = out["last_season"] == target_season - 1     # team is then his target-1 team
    out["aging"] = out["group"].map(aging)

    one_year = _curve_lookup(curve, out["group"], out["age_next"] - 1) if curve is not None else np.zeros(len(out))
    proj = np.where(out["aging"] == "one_year", out["tt_war_60"] + np.nan_to_num(one_year), out["tt_war_60"])

    adjust = out.index[out["aging"] == "adjust_then_blend"]
    if len(adjust):
        a = prior[prior["playerId"].isin(adjust)].copy()
        target_age = a["playerId"].map(out["age_next"])
        a["gar"] = a["gar"] + cumulative_aging(curve, a["group"], a["age"], target_age) * GOALS_PER_WIN * a["toi"] / 60
        adjusted = _blend(a, h, target_season, weights, reg_minutes)["tt_war_60"]
        proj = pd.Series(proj, index=out.index)
        proj.loc[adjust] = adjusted.reindex(adjust)
    out["proj_war_60"] = np.asarray(proj, dtype=float)
    out["aging_change"] = out["proj_war_60"] - out["tt_war_60"]
    return out


def backtest(history, targets, weights, reg_minutes, min_target_toi, bios=None, aging="none", average_only=False,
             center=True):
    """Predicted vs. actual GAR/60 for every target season. Players: min_target_toi+ minutes in the target season
    and a season in target-1 (the same players for every setting). With aging, the curve is refit for each target
    without any pair that includes the target season (center: see aging_curve).
    average_only=True predicts his group's average instead."""
    needs_curve = (aging if isinstance(aging, dict) else {"x": aging}).values()
    rows = []
    for t in targets:
        tgt = history[(history["season"] == t) & (history["toi"] >= min_target_toi)
                      & history["playerId"].isin(history.loc[history["season"] == t - 1, "playerId"])]
        curve = aging_curve(history, bios, exclude_season=t, center=center)[1] if any(v != "none" for v in needs_curve) else None
        proj = project(history, bios, t, weights, reg_minutes, aging, curve)
        pred = proj["group_gar_60"] if average_only else proj["proj_war_60"] * GOALS_PER_WIN
        rows.append(pd.DataFrame({"season": t, "playerId": tgt["playerId"].to_numpy(), "group": tgt["group"].to_numpy(),
                                  "pred": pred.reindex(tgt["playerId"]).to_numpy(),
                                  "actual": (tgt["gar"] / tgt["toi"] * 60).to_numpy(), "toi": tgt["toi"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def rmse(bt):
    """TOI-weighted root mean squared error of GAR/60 (players with no prediction are an error, not skipped)."""
    if bt["pred"].isna().any():
        raise ValueError(f"{int(bt['pred'].isna().sum())} players have no prediction")
    return float(np.sqrt(np.average((bt["pred"] - bt["actual"]) ** 2, weights=bt["toi"])))


def grid(history, targets, weight_sets, reg_list, min_target_toi):
    """RMSE for every weights x reg_minutes setting (rows = weights, columns = reg_minutes)."""
    return pd.DataFrame({reg: {str(w): rmse(backtest(history, targets, w, reg, min_target_toi)) for w in weight_sets}
                         for reg in reg_list})
