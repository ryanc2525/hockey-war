"""
Hockey WAR model, packaged as reusable functions.

DEFAULT_PARAMS reproduces v0.3 exactly (kept so older notebooks still match).
V04_PARAMS is the current single-season model (v0.4).

Same math as notebooks 01 and 04, but written once so it can run on any
MoneyPuck season-level skaters file:

    from war_model import run_season
    war_2025 = run_season("../data/skaters25-26.csv")

Data: MoneyPuck.com (season-level skaters CSV).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model assumptions (v0.3). Change these here, not inside the functions.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "goals_per_win": 6.0,         # ~6 goals of goal differential = 1 win
    "penalty_goal_value": 0.17,   # value of drawing (or taking) one penalty, in goals
    "a1_weight": 0.5,             # credit for a primary assist
    "finish_weight": 0.3,         # credit for finishing (goals minus expected goals)
    "def_weight": 0.5,            # weight on on-ice defense (noisy, so half weight)
    "fo_goal_value": 0.015,       # value of one faceoff win, in goals
    "regular_forwards": 13,       # top forwards per team by ice time = regulars
    "regular_defense": 7,         # top defensemen per team by ice time = regulars
    "ev_pool_min_toi": 100,       # minutes needed to count in the EV replacement pool
    "pp_regulars": 10,            # top power play players per team
    "pp_pool_min_toi": 10,
    "pk_pool_min_toi": 10,
    "fo_pool_min": 100,           # faceoffs needed to count in the faceoff pool
}

STAT_COLS = [
    "icetime",
    "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_xGoals",
    "OnIce_F_xGoals", "OnIce_A_xGoals",
    "onIce_xGoalsPercentage", "offIce_xGoalsPercentage",
    "penalties", "penaltiesDrawn",
    "faceoffsWon", "faceoffsLost",
]

SIT_MAP = {"all": "all", "5on5": "ev", "5on4": "pp", "4on5": "pk"}

COMPONENTS = ["ev_off_gar", "pp_off_gar", "ev_def_gar", "pk_def_gar", "pen_gar", "fo_gar"]


V03_PARAMS = DEFAULT_PARAMS

# v0.4: single-season WAR measures what happened, so goals above expected count fully
V04_PARAMS = {**DEFAULT_PARAMS, "finish_weight": 1.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rate60(count, toi_minutes):
    """Rate per 60 minutes; NaN when there is no ice time."""
    return np.where(toi_minutes > 0, count / toi_minutes * 60, np.nan)


def pooled(group, stat, toi_col):
    """Total stat / total minutes for a group, per 60. Avoids small-sample noise."""
    return group[stat].sum() / group[toi_col].sum() * 60


def toi_weighted(group, col, toi_col):
    """Ice-time-weighted average of a per-60 rate."""
    return (group[col] * group[toi_col]).sum() / group[toi_col].sum()


# ---------------------------------------------------------------------------
# Step 1: load and reshape (notebook 01)
# ---------------------------------------------------------------------------
def load_players(path, params=DEFAULT_PARAMS):
    """Read a MoneyPuck skaters file and return one row per player."""
    raw = pd.read_csv(path).copy()
    raw["pos_group"] = np.where(raw["position"] == "D", "D", "F")

    sub = raw[raw["situation"].isin(SIT_MAP.keys())].copy()
    sub["sit"] = sub["situation"].map(SIT_MAP)
    wide = sub.pivot_table(index="playerId", columns="sit", values=STAT_COLS, aggfunc="first")
    wide.columns = [f"{sit}_{stat}" for stat, sit in wide.columns]

    info = (raw.loc[raw["situation"] == "all",
                    ["playerId", "season", "name", "team", "position", "pos_group", "games_played"]]
               .set_index("playerId"))
    players = info.join(wide).reset_index()

    for sit in SIT_MAP.values():
        players[f"{sit}_toi"] = players[f"{sit}_icetime"] / 60

    # Even-strength bench time and team xG against while on the bench (for defense)
    bench = (raw.loc[raw["situation"] == "5on5", ["playerId", "timeOnBench", "OffIce_A_xGoals"]]
                .rename(columns={"timeOnBench": "ev_bench_sec", "OffIce_A_xGoals": "ev_OffIce_A_xGoals"}))
    players = players.merge(bench, on="playerId", how="left")
    players["ev_bench_toi"] = players["ev_bench_sec"] / 60

    players["pen_diff"] = players["all_penaltiesDrawn"] - players["all_penalties"]
    players["fo_total"] = players["all_faceoffsWon"] + players["all_faceoffsLost"]
    players["fo_pct"] = np.where(players["fo_total"] >= 100,
                                 players["all_faceoffsWon"] / players["fo_total"], np.nan)

    # Regulars: top 13 F / 7 D per team by total ice time
    players["toi_rank"] = players.groupby(["team", "pos_group"])["all_toi"].rank(ascending=False, method="first")
    lineup = players["pos_group"].map({"F": params["regular_forwards"], "D": params["regular_defense"]})
    players["is_regular"] = players["toi_rank"] <= lineup
    return players.copy()


# ---------------------------------------------------------------------------
# Step 2: replacement levels
# ---------------------------------------------------------------------------
def replacement_levels(players, params=DEFAULT_PARAMS):
    """Return the replacement (or baseline) tables used by each component."""
    p = players
    p["ev_rel_xga_60"] = (rate60(p["ev_OffIce_A_xGoals"], p["ev_bench_toi"])
                          - rate60(p["ev_OnIce_A_xGoals"], p["ev_toi"]))
    p["ev_rel_xga_60"] = p["ev_rel_xga_60"].fillna(0)

    # Even strength: non-regulars with enough minutes
    ev_pool = p[(~p["is_regular"]) & (p["ev_toi"] >= params["ev_pool_min_toi"])]
    ev = ev_pool.groupby("pos_group").apply(lambda g: pd.Series({
        "goals_60": pooled(g, "ev_I_F_goals", "ev_toi"),
        "a1_60": pooled(g, "ev_I_F_primaryAssists", "ev_toi"),
        "ixg_60": pooled(g, "ev_I_F_xGoals", "ev_toi"),
        "rel_xga_60": toi_weighted(g, "ev_rel_xga_60", "ev_toi"),
        "pen_diff_60": g["pen_diff"].sum() / g["all_toi"].sum() * 60,
    }), include_groups=False)
    ev["fin_60"] = ev["goals_60"] - ev["ixg_60"]

    # Power play: outside each team's top PP players
    p["pp_rank"] = p.groupby("team")["pp_toi"].rank(ascending=False, method="first")
    pp_pool = p[(p["pp_rank"] > params["pp_regulars"]) & (p["pp_toi"] >= params["pp_pool_min_toi"])]
    pp = pp_pool.groupby("pos_group").apply(lambda g: pd.Series({
        "ixg_60": pooled(g, "pp_I_F_xGoals", "pp_toi"),
        "fin_60": pooled(g, "pp_I_F_goals", "pp_toi") - pooled(g, "pp_I_F_xGoals", "pp_toi"),
        "a1_60": pooled(g, "pp_I_F_primaryAssists", "pp_toi"),
    }), include_groups=False)

    # Penalty kill: compared to own team's PK, league-average killer as baseline
    team_pk = p.groupby("team").apply(
        lambda g: g["pk_OnIce_A_xGoals"].sum() / g["pk_toi"].sum() * 60, include_groups=False)
    p["pk_rel_xga_60"] = (p["team"].map(team_pk) - rate60(p["pk_OnIce_A_xGoals"], p["pk_toi"]))
    p["pk_rel_xga_60"] = p["pk_rel_xga_60"].fillna(0)
    pk_pool = p[p["pk_toi"] >= params["pk_pool_min_toi"]]
    pk = pk_pool.groupby("pos_group").apply(lambda g: pd.Series({
        "rel_xga_60": toi_weighted(g, "pk_rel_xga_60", "pk_toi"),
    }), include_groups=False)

    # Faceoffs: non-regular centers
    fo_pool = p[(p["position"] == "C") & (~p["is_regular"]) & (p["fo_total"] >= params["fo_pool_min"])]
    fo_pct = fo_pool["all_faceoffsWon"].sum() / fo_pool["fo_total"].sum()

    return {"ev": ev, "pp": pp, "pk": pk, "fo_pct": fo_pct}


# ---------------------------------------------------------------------------
# Step 3: components and WAR (notebook 04, v0.3)
# ---------------------------------------------------------------------------
def compute_war(players, params=DEFAULT_PARAMS):
    """Add goals-above-replacement components and WAR to the players table."""
    p = players.copy()
    repl = replacement_levels(p, params)

    def r(table, col):
        return p["pos_group"].map(table[col])

    def offense(prefix, table):
        hours = p[f"{prefix}_toi"] / 60
        xg, goals, a1 = (p[f"{prefix}_I_F_xGoals"], p[f"{prefix}_I_F_goals"],
                         p[f"{prefix}_I_F_primaryAssists"])
        return ((xg - r(table, "ixg_60") * hours)
                + params["finish_weight"] * ((goals - xg) - r(table, "fin_60") * hours)
                + params["a1_weight"] * (a1 - r(table, "a1_60") * hours))

    p["ev_off_gar"] = offense("ev", repl["ev"])
    p["pp_off_gar"] = offense("pp", repl["pp"])
    p["ev_def_gar"] = params["def_weight"] * (p["ev_rel_xga_60"] - r(repl["ev"], "rel_xga_60")) * p["ev_toi"] / 60
    p["pk_def_gar"] = params["def_weight"] * (p["pk_rel_xga_60"] - r(repl["pk"], "rel_xga_60")) * p["pk_toi"] / 60
    p["pen_gar"] = params["penalty_goal_value"] * (p["pen_diff"] - r(repl["ev"], "pen_diff_60") * p["all_toi"] / 60)
    p["fo_gar"] = np.where(p["position"] == "C",
                           params["fo_goal_value"] * (p["all_faceoffsWon"] - repl["fo_pct"] * p["fo_total"]),
                           0.0)

    p["gar"] = p[COMPONENTS].sum(axis=1)
    p["war"] = p["gar"] / params["goals_per_win"]

    # Split for the player card: offense + defense + other = total WAR
    gpw = params["goals_per_win"]
    p["off_war"] = (p["ev_off_gar"] + p["pp_off_gar"]) / gpw
    p["def_war"] = (p["ev_def_gar"] + p["pk_def_gar"]) / gpw
    p["other_war"] = (p["pen_gar"] + p["fo_gar"]) / gpw
    p["war_rank"] = p["war"].rank(ascending=False, method="min").astype(int)
    return p


def run_season(path, params=DEFAULT_PARAMS):
    """Load one season file and return every skater's WAR."""
    return compute_war(load_players(path, params), params)
