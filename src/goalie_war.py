"""
Goalie WAR v0: goals saved above expected (GSAx) above a replacement-level backup, in wins.

    from goalie_war import load_goalies, replacement_rates, goalie_war, team_check, repeatability
    goalies = load_goalies(raw)                  # raw = MoneyPuck goalie files, all seasons stacked
    rates, gap = replacement_rates(goalies)
    gw = goalie_war(goalies, rates)

GSAx = xGoals - goals, all situations, with MoneyPuck's regular xGoals (not flurry-adjusted, same as the skater model).
Replacement = goalies outside each team's top 2 by ice time (same idea as the skaters' top 13 F / 7 D).
Their gap below the league average is pooled over all seasons, because one season's pool (23 to 55 backups)
jumps around a lot; each season's replacement rate is that season's league average plus the pooled gap,
so league-wide drift in expected goals cancels out.
season = MoneyPuck start year (2025 = 2025-26). Data: MoneyPuck.com.
"""

import pandas as pd

from war_model import V04_PARAMS

GOALS_PER_WIN = V04_PARAMS["goals_per_win"]      # the same conversion the skater model uses
KEY = ["playerId", "season"]
STARTERS_PER_TEAM = 2                            # top 2 goalies per team by ice time; everyone else is the replacement pool
MIN_REPEAT_TOI = 1000                            # minutes in both seasons for the repeatability table
MIN_PK_TOI = 150                                 # 4-on-5 minutes in both seasons for the penalty kill row

# Old team code -> current franchise code. Only used to tell whether a goalie changed teams between seasons,
# so a renamed code or a relocated franchise doesn't count as a team change. franchise_check proves it's complete.
FRANCHISE = {
    "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL",    # MoneyPuck's codes before 2021-22
    "ATL": "WPG",                                                # Atlanta moved to Winnipeg for 2011-12
    "ARI": "UTA",                                                # Arizona moved to Utah for 2024-25
}
CURRENT_TEAMS = {"ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN",
                 "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA",
                 "VAN", "VGK", "WPG", "WSH"}

# Repeatability rows: column, label, (more-is-better count, subtracted count), minutes column.
# Every rate is per 60 minutes, oriented so higher = better goalie. Save % is not per 60.
REPEAT_STATS = [
    ("gsax_60", "GSAx",                    ("xGoals", "goals"),                     "toi"),
    ("hd_60",   "High-danger GSAx",        ("highDangerxGoals", "highDangerGoals"), "toi"),
    ("md_60",   "Medium-danger GSAx",      ("mediumDangerxGoals", "mediumDangerGoals"), "toi"),
    ("ld_60",   "Low-danger GSAx",         ("lowDangerxGoals", "lowDangerGoals"),   "toi"),
    ("reb_60",  "Rebounds vs. expected",   ("xRebounds", "rebounds"),               "toi"),
    ("frz_60",  "Freezes vs. expected",    ("freeze", "xFreeze"),                   "toi"),
    ("sv_pct",  "Save %",                  None,                                    "toi"),
    ("ev_60",   "5v5 GSAx",                ("ev_xGoals", "ev_goals"),               "ev_toi"),
    ("pk_60",   "4on5 GSAx (150+ PK min)", ("pk_xGoals", "pk_goals"),               "pk_toi"),
]
ALL_COUNTS = ["goals", "xGoals", "ongoal", "rebounds", "xRebounds", "freeze", "xFreeze",
              "highDangerGoals", "highDangerxGoals", "mediumDangerGoals", "mediumDangerxGoals",
              "lowDangerGoals", "lowDangerxGoals"]


def season_label(season):
    """2025 -> '2025-26'."""
    return f"{season}-{str(season + 1)[-2:]}"


# ---------------------------------------------------------------------------
# 1. One row per goalie-season
# ---------------------------------------------------------------------------
def load_goalies(raw):
    """MoneyPuck goalie rows (any number of seasons) -> one row per goalie-season.
    toi in minutes (icetime is in seconds); all-situation counts, plus 5v5 (ev_) and 4-on-5 (pk_) goals and xGoals."""
    def sit(name):
        return raw[raw["situation"] == name].set_index(KEY)

    a, ev, pk = sit("all"), sit("5on5"), sit("4on5")
    g = a[["name", "team", "games_played"] + ALL_COUNTS].copy()
    g["toi"] = a["icetime"] / 60
    g["gsax"] = g["xGoals"] - g["goals"]
    for prefix, s in [("ev", ev), ("pk", pk)]:
        g[f"{prefix}_toi"] = s["icetime"] / 60
        g[f"{prefix}_goals"] = s["goals"]
        g[f"{prefix}_xGoals"] = s["xGoals"]
    g = g.reset_index()
    g["season_label"] = g["season"].map(season_label)
    g["toi_rank"] = g.groupby(["season", "team"])["toi"].rank(ascending=False, method="first")
    g["is_starter"] = g["toi_rank"] <= STARTERS_PER_TEAM
    return g


# ---------------------------------------------------------------------------
# 2. Replacement level and WAR
# ---------------------------------------------------------------------------
def replacement_rates(goalies):
    """One row per season: league GSAx/60 (all goalies, TOI-weighted), the backup pool's GSAx/60 and size,
    rate_per_season (the pool's own rate, for comparison) and rate (league + pooled gap, used for WAR).
    Returns (rates, gap), gap = the pool's TOI-weighted GSAx/60 below the league average, over all seasons."""
    by = goalies.groupby("season")
    pool = goalies[~goalies["is_starter"]].groupby("season")
    rates = pd.DataFrame({
        "league_gsax_60": by["gsax"].sum() / by["toi"].sum() * 60,
        "pool_gsax_60": pool["gsax"].sum() / pool["toi"].sum() * 60,
        "pool_n": pool.size(),
        "pool_toi": pool["toi"].sum(),
    })
    gap = (((rates["pool_gsax_60"] - rates["league_gsax_60"]) * rates["pool_toi"]).sum()
           / rates["pool_toi"].sum())
    rates["rate_per_season"] = rates["pool_gsax_60"]
    rates["rate"] = rates["league_gsax_60"] + gap
    return rates, gap


def goalie_war(goalies, rates, rate_col="rate"):
    """Add gsax_60, repl_rate, gar, war and war_rank (within season).
    GAR = GSAx - replacement GSAx/60 x minutes / 60; WAR = GAR / GOALS_PER_WIN."""
    g = goalies.copy()
    g["gsax_60"] = g["gsax"] / g["toi"] * 60
    g["repl_rate"] = g["season"].map(rates[rate_col])
    g["gar"] = g["gsax"] - g["repl_rate"] * g["toi"] / 60
    g["war"] = g["gar"] / GOALS_PER_WIN
    g["war_rank"] = g.groupby("season")["war"].rank(ascending=False, method="min").astype(int)
    return g


# ---------------------------------------------------------------------------
# 3. Validation: team WAR vs. team goal differential (notebook 06's method)
# ---------------------------------------------------------------------------
def team_check(skater_war, goalie_war_table, skaters_raw):
    """Per season: correlation of team WAR with team 5v5 goal differential, skaters only and skaters + goalies.
    Team 5v5 goals = sum of every skater's on-ice goals / 5 (five skaters on the ice)."""
    ev = skaters_raw.loc[skaters_raw["situation"] == "5on5", KEY + ["OnIce_F_goals", "OnIce_A_goals"]]
    teams = (skater_war.merge(ev, on=KEY)
                       .groupby(["season", "team"])
                       .agg(skater_war=("war", "sum"), gf=("OnIce_F_goals", "sum"), ga=("OnIce_A_goals", "sum")))
    teams["gd_5v5"] = (teams["gf"] - teams["ga"]) / 5
    gw = goalie_war_table.groupby(["season", "team"])["war"].sum()
    missing = set(gw.index) - set(teams.index)
    if missing:
        raise ValueError(f"goalie teams with no skaters: {sorted(missing)}")
    teams["goalie_war"] = gw.reindex(teams.index).fillna(0)
    teams["total_war"] = teams["skater_war"] + teams["goalie_war"]
    return teams.groupby("season").apply(lambda t: pd.Series({
        "skaters": t["skater_war"].corr(t["gd_5v5"]),
        "skaters + goalies": t["total_war"].corr(t["gd_5v5"]),
        "teams": len(t),
    }), include_groups=False)


# ---------------------------------------------------------------------------
# 4. Team codes and repeatability
# ---------------------------------------------------------------------------
def franchise(team):
    """Current franchise code for any team code (Series)."""
    return team.map(lambda t: FRANCHISE.get(t, t))


def team_codes(raw):
    """Which team codes appear in which season (1 = present)."""
    return pd.crosstab(raw["team"], raw["season"]).gt(0).astype(int)


def franchise_check(skaters_raw, goalies_raw):
    """Check that FRANCHISE covers every renamed or relocated team code. Raises if anything is off;
    returns the code-by-season table."""
    codes = team_codes(skaters_raw)
    if not codes.equals(team_codes(goalies_raw)):
        raise ValueError("the skater and goalie files use different team codes")
    unknown = set(codes.index) - CURRENT_TEAMS - set(FRANCHISE)
    if unknown:
        raise ValueError(f"team codes not in CURRENT_TEAMS or FRANCHISE: {sorted(unknown)}")
    for old, new in FRANCHISE.items():
        if old not in codes.index:
            raise ValueError(f"{old} is in FRANCHISE but never appears in the data")
        last_old = codes.columns[codes.loc[old] == 1].max()
        first_new = codes.columns[codes.loc[new] == 1].min()
        if last_old + 1 != first_new:
            raise ValueError(f"{old} ends in {last_old} but {new} starts in {first_new}")
    return codes


def goalie_rates(goalies, adjust=False):
    """The REPEAT_STATS rates per goalie-season. adjust=True subtracts that season's league value
    (TOI-weighted), which removes league-wide drift in MoneyPuck's expected goals."""
    r = goalies[KEY + ["name", "team", "toi", "pk_toi"]].copy()
    for col, _, counts, toi in REPEAT_STATS:
        if counts is None:                                          # save %
            r[col] = 1 - goalies["goals"] / goalies["ongoal"]
            league = 1 - goalies.groupby("season")["goals"].sum() / goalies.groupby("season")["ongoal"].sum()
        else:
            num = goalies[counts[0]] - goalies[counts[1]]
            r[col] = num / goalies[toi] * 60
            by = goalies.assign(num=num).groupby("season")
            league = by["num"].sum() / by[toi].sum() * 60
        if adjust:
            r[col] = r[col] - goalies["season"].map(league)
    return r


def season_pairs(goalies, adjust=False):
    """Back-to-back seasons for the same goalie, MIN_REPEAT_TOI+ minutes in both.
    changed_team compares franchises; code_change_only = different code, same franchise (not a team change)."""
    r = goalie_rates(goalies, adjust)
    r["franchise"] = franchise(r["team"])
    p = r.merge(r.assign(season=r["season"] - 1), on=KEY, suffixes=("_1", "_2"))
    p = p[(p["toi_1"] >= MIN_REPEAT_TOI) & (p["toi_2"] >= MIN_REPEAT_TOI)].copy()
    p["changed_team"] = p["franchise_1"] != p["franchise_2"]
    p["code_change_only"] = (p["team_1"] != p["team_2"]) & ~p["changed_team"]
    return p


def repeatability(goalies, adjust=False):
    """Year-to-year correlation of each REPEAT_STATS rate: all pairs and team changers, with pair counts."""
    p = season_pairs(goalies, adjust)
    rows = []
    for col, label, _, toi in REPEAT_STATS:
        q = p
        if toi == "pk_toi":
            q = p[(p["pk_toi_1"] >= MIN_PK_TOI) & (p["pk_toi_2"] >= MIN_PK_TOI)]
        ch = q[q["changed_team"]]
        rows.append({"stat": label,
                     "all r": q[f"{col}_1"].corr(q[f"{col}_2"]), "all n": len(q),
                     "changed team r": ch[f"{col}_1"].corr(ch[f"{col}_2"]), "changed team n": len(ch)})
    return pd.DataFrame(rows).set_index("stat")
