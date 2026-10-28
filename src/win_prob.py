"""
Win probability (WP) and win probability added (WPA) from MoneyPuck shot data.

    from win_prob import game_states, fit_wp, wp_home, ot_home_rate, shot_wpa, player_wpa
    states = game_states(shots)                      # shots = data/shots_2021-2025.parquet
    coef = fit_wp(states[states["season"] <= 2023])
    p_ot, n_ot = ot_home_rate(shots, [2021, 2022, 2023])
    players = player_wpa(shot_wpa(shots, coef, p_ot))

The score always comes from MoneyPuck's score-before-shot columns (homeTeamGoals, awayTeamGoals), not from
counting goal rows: about 49 goals are missing from the shot file, and those columns still include them.
A game is (season, game_id); season = MoneyPuck start year (2025 = 2025-26).

Data: MoneyPuck.com.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

GAME = ["season", "game_id"]
KEY = ["playerId", "season"]
REG_SECS = 3600                      # regulation: three 20-minute periods
STATE_STEP = 10                      # one game state every 10 seconds, so the final minute is well covered
# Chosen so the model matches how often teams down one goal late actually come back (goalie pulls).
# With +1, a team down 1 with under a minute left was predicted at 0.6% vs 4.5% actual; with +240 it's about 5.6%.
LATE_GAME_OFFSET = 240
HIGH_LEVERAGE = 0.15                 # a goal worth at least 15 percentage points of win probability
FINISH_PRIOR = 10                    # finishing shrunk toward 1.0: (goals + 10) / (xG + 10)
COEF_NAMES = ["intercept (home ice)", "score diff", "secs remaining / 3600", f"diff / sqrt(secs + {LATE_GAME_OFFSET})"]


# ---------------------------------------------------------------------------
# 1. Game states and the model
# ---------------------------------------------------------------------------
def _prep(shots):
    """Integer keys, shots in game order, and the score right after each shot (score before + this goal)."""
    s = shots.copy()
    for col in ["season", "game_id", "time", "id", "period"]:
        s[col] = s[col].astype("int64")
    s = s.sort_values(GAME + ["time", "id"])
    home_goal = (s["goal"] == 1) & (s["isHomeTeam"] == 1)
    away_goal = (s["goal"] == 1) & (s["isHomeTeam"] == 0)
    s["home_after"] = s["homeTeamGoals"].astype("int64") + home_goal
    s["away_after"] = s["awayTeamGoals"].astype("int64") + away_goal
    return s


def game_states(shots):
    """One row per game every STATE_STEP seconds of regulation: season, game_id, t (seconds elapsed),
    secs (remaining), diff (home minus away, latest known score), home_won."""
    s = _prep(shots)
    games = s.groupby(GAME).agg(home_won=("homeTeamWon", "first")).reset_index()
    reg = s.loc[s["period"] <= 3, GAME + ["time", "home_after", "away_after"]].rename(columns={"time": "t"})
    steps = pd.DataFrame({"t": np.arange(0, REG_SECS, STATE_STEP, dtype="int64")})
    states = games.merge(steps, how="cross").sort_values("t")
    states = pd.merge_asof(states, reg.sort_values("t"), on="t", by=GAME, direction="backward")   # last shot at or before t
    score = states[["home_after", "away_after"]].fillna(0)                                         # before the first shot: 0-0
    states["diff"] = (score["home_after"] - score["away_after"]).astype("int64")
    states["secs"] = REG_SECS - states["t"]
    states["home_won"] = states["home_won"].astype("int64")
    return states[GAME + ["t", "secs", "diff", "home_won"]].sort_values(GAME + ["t"]).reset_index(drop=True)


def _features(diff, secs):
    diff = np.atleast_1d(np.asarray(diff, dtype=float))
    secs = np.atleast_1d(np.asarray(secs, dtype=float))
    return np.column_stack([np.ones_like(diff), diff, secs / REG_SECS, diff / np.sqrt(secs + LATE_GAME_OFFSET)])


def fit_wp(states):
    """Logistic regression of home_won on the four terms in COEF_NAMES (maximum likelihood, scipy BFGS)."""
    X = _features(states["diff"], states["secs"])
    y = states["home_won"].to_numpy(dtype=float)
    n = len(y)

    def loss(b):                     # average negative log-likelihood
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) / n

    def grad(b):
        return X.T @ (expit(X @ b) - y) / n

    result = minimize(loss, np.zeros(X.shape[1]), jac=grad, method="BFGS")
    if not result.success:
        raise RuntimeError(f"Win probability fit did not converge: {result.message}")
    return result.x


def wp_home(diff, secs, coef):
    """Home win probability in regulation, for one state or many (diff = home minus away, secs remaining)."""
    p = expit(_features(diff, secs) @ coef)
    return float(p[0]) if np.ndim(diff) == 0 and np.ndim(secs) == 0 else p


def ot_home_rate(shots, seasons):
    """Home win rate in games tied after regulation (overtime or shootout), and how many games that is."""
    s = _prep(shots)
    s = s[s["season"].isin(seasons)]
    games = s.groupby(GAME).agg(home_won=("homeTeamWon", "first"), had_ot=("period", lambda p: (p >= 4).any()))
    last_reg = s[s["period"] <= 3].groupby(GAME)[["home_after", "away_after"]].last().reindex(games.index)
    tied = games["had_ot"] | (last_reg["home_after"] == last_reg["away_after"])
    return float(games.loc[tied, "home_won"].mean()), int(tied.sum())


# ---------------------------------------------------------------------------
# 2. WPA per shot and per player-season
# ---------------------------------------------------------------------------
def shot_wpa(shots, coef, p_ot):
    """Add wpa_if_goal to every shot: the shooting team's win probability after a goal minus before, at that moment.
    Regulation: from the model at the same time, score diff +1 from the shooter's side.
    Overtime (period 4): a goal ends the game, so home gains 1 - p_ot and away gains p_ot."""
    s = shots.copy()
    home = s["isHomeTeam"].to_numpy() == 1
    diff = (s["homeTeamGoals"].astype("int64") - s["awayTeamGoals"].astype("int64")).to_numpy()
    secs = np.clip(REG_SECS - s["time"].astype("int64").to_numpy(), 0, None)
    ot = s["period"].to_numpy() >= 4

    before = wp_home(diff, secs, coef)
    home_gain = wp_home(diff + 1, secs, coef) - before          # home scores
    away_gain = before - wp_home(diff - 1, secs, coef)          # away scores (home WP drops by this much)
    s["wp_home_before"] = np.where(ot, p_ot, before)
    s["wpa_if_goal"] = np.where(ot, np.where(home, 1 - p_ot, p_ot), np.where(home, home_gain, away_gain))
    return s


def _most_common(s, col):
    """Most frequent value of col for each player-season."""
    counts = s.groupby(KEY + [col], observed=True).size().reset_index(name="n")
    return counts.sort_values("n").drop_duplicates(KEY, keep="last").set_index(KEY)[col].astype(str)


def player_wpa(shots):
    """One row per player-season: shots, goals, xG, WPA, xWPA, finishing, Clutch, high-leverage and OT goals.
    shots = output of shot_wpa. Shots with no shooter ID are left out."""
    s = shots[shots["shooterPlayerId"].notna() & (shots["shooterPlayerId"] != 0)].copy()
    s["playerId"] = s["shooterPlayerId"].astype("int64")
    s["season"] = s["season"].astype("int64")
    s["goal_wpa"] = s["wpa_if_goal"] * s["goal"]
    s["shot_xwpa"] = s["xGoal"].astype(float) * s["wpa_if_goal"]
    s["hl_goal"] = (s["goal"] == 1) & (s["wpa_if_goal"] >= HIGH_LEVERAGE)
    s["ot_goal"] = (s["goal"] == 1) & (s["period"].astype("int64") >= 4)

    out = s.groupby(KEY).agg(
        unblocked_shots=("goal", "size"), goals=("goal", "sum"), xg=("xGoal", "sum"),
        wpa=("goal_wpa", "sum"), xwpa=("shot_xwpa", "sum"), hl_goals=("hl_goal", "sum"), ot_goals=("ot_goal", "sum"))
    out["xg"] = out["xg"].astype(float)
    out["name"] = _most_common(s, "shooterName")
    out["position"] = _most_common(s, "playerPositionThatDidEvent")
    out["team"] = _most_common(s, "teamCode")
    out = out.reset_index()
    out["season_label"] = out["season"].map(lambda y: f"{y}-{str(y + 1)[-2:]}")
    out["pos_group"] = np.where(out["position"] == "D", "D", "F")
    out["finishing"] = (out["goals"] + FINISH_PRIOR) / (out["xg"] + FINISH_PRIOR)
    out["clutch"] = out["wpa"] - out["finishing"] * out["xwpa"]
    cols = ["playerId", "season", "season_label", "name", "position", "pos_group", "team", "unblocked_shots",
            "goals", "xg", "finishing", "wpa", "xwpa", "clutch", "hl_goals", "ot_goals"]
    return out[cols]
