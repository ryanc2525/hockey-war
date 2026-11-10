"""
Lineups and minutes: which 12 forwards, 6 defensemen and 2 goalies a coach dresses, and how much each one plays.
Notebook 23 builds and tests this; the season simulator imports it.

    import lineups as L
    players = L.load_players(DATA, usage, current_projections(DATA), LOOKBACK)   # 2026-27 opening night
    scored = L.coach_scores(players[players["in_pool"]], SETTINGS, DEFAULTS)
    lineups = L.build_lineups(scored, slots, share, trades, trade_ids=[], sit_out=[])   # all 32 teams
    L.lineup_strength(lineups)                                                    # WAR per game and per 84

How a coach picks (the "coach score"), within forwards, defensemen or goalies:
    score = w x z(usage) + (1 - w) x z(projected WAR/60)
    usage = ice time per game (skaters) or games per 82 (goalies) in his most recent season that counts, looking
    back LOOKBACK seasons (normally last season). z = standardized across the league's pool.
    A player without usable NHL usage gets it from a rule (R1 rookie default or R2 expected from projection).
    Notebook 23's backtest picks w, the rule and the lookback.
The best 12 forwards dress (at least 4 natural centers), then the best 6 defensemen and 2 goalies.
Minutes come from the slot: the k-th best forward plays what a team's k-th forward typically plays.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from development import SEASON_GAMES, FULL_GAMES

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
N_DRESSED = {"F": 12, "D": 6, "G": 2}
N_CENTERS = 4
MIN_GP_USAGE = 10            # games in a season for a skater's own ice time per game to count
CENTER_FO_60 = 20            # faceoffs per 60 that make a forward a natural center
MIN_GP_POOL = 20             # backtest: players with 20+ games for a team that season form its pool
MAX_RANK = {"F": 16, "D": 9}  # ranks kept in the raw minutes table (extras in the backtest)
SEASON_GAMES_2026 = 84       # the 2026-27 schedule
USAGE_FIRST = "usage first"  # the setting "real usage first, rating breaks ties"

POOL_STATUSES = ["NHL roster", "NHL org, off active roster", "suspended / holdout", "unsigned RFA",
                 "AHL/ECHL (NHL contract)"]
DISPUTE_STATUSES = ["suspended / holdout", "unsigned RFA"]
AHL_STATUSES = ["AHL/ECHL (NHL contract)"]          # tier 2 when call-ups are tiered (notebook 24)


def season_games(season):
    """Games per team in a season (82, except the short ones)."""
    return pd.Series(season).map(SEASON_GAMES).fillna(FULL_GAMES).to_numpy()


# ---------------------------------------------------------------------------
# 2. Usage from real data: ice time per game, games per 82, faceoffs per 60
# ---------------------------------------------------------------------------
def faceoff_rates(skater_files):
    """Faceoffs taken per 60 minutes (all situations) for every skater-season in MoneyPuck's skater files."""
    cols = ["playerId", "season", "situation", "icetime", "faceoffsWon", "faceoffsLost"]
    s = pd.concat([pd.read_csv(f, usecols=cols) for f in skater_files], ignore_index=True)
    s = s[s["situation"] == "all"]
    rate = (s["faceoffsWon"] + s["faceoffsLost"]) / (s["icetime"] / 3600)
    return s[["playerId", "season"]].assign(fo_60=rate.to_numpy())


def usage_table(skater_war, goalie_war, faceoffs):
    """
    One row per player-season: team (his final team that season), position, group, games_played,
    usage (skaters: all-situations ice time per game; goalies: games per 82), and fo_60 (skaters).
    A traded player has one row: his combined numbers, under his final team.
    """
    sk = skater_war[["playerId", "season", "name", "team", "position", "pos_group", "games_played", "all_toi"]]
    sk = sk.rename(columns={"pos_group": "group"})
    sk = sk.assign(usage=sk["all_toi"] / sk["games_played"]).merge(faceoffs, on=["playerId", "season"], how="left")
    gk = goalie_war[["playerId", "season", "name", "team", "games_played", "toi"]].rename(columns={"toi": "all_toi"})
    gk = gk.assign(position="G", group="G", usage=gk["games_played"] * FULL_GAMES / season_games(gk["season"]),
                   fo_60=np.nan)
    return pd.concat([sk, gk[sk.columns]], ignore_index=True)


def last_usage(usage, players, season, lookback=1):
    """
    Each player's usage in his most recent season that counts (skaters 10+ games, goalies 1+), looking back up
    to `lookback` seasons before `season` (normally that's last season). NaN if none: then a rule fills it in.
    Looking back means a player who missed last season (injured, for example) keeps his real role.
    """
    u = usage[usage["season"].between(season - lookback, season - 1)]
    ok = np.where(u["group"] == "G", u["games_played"] >= 1, u["games_played"] >= MIN_GP_USAGE)
    recent = u[ok].sort_values("season").groupby("playerId")["usage"].last()
    return pd.Series(players).map(recent).to_numpy()


def usage_by_group(usage, pool, season, lookback):
    """usage_last for a pool, with each group's own lookback ({group: seasons}, e.g. {"F": 2, "D": 1, "G": 1})."""
    out = np.full(len(pool), np.nan)
    for g, lb in lookback.items():
        m = (pool["group"] == g).to_numpy()
        out[m] = last_usage(usage, pool.loc[m, "playerId"], season, lb)
    return out


def natural_center(usage, players, positions, season):
    """
    A natural center: 20+ faceoffs per 60 in his most recent NHL season (10+ games) before `season`.
    A player with no such season counts as a center if his listed position is C.
    """
    u = usage[(usage["season"] < season) & (usage["group"] == "F") & (usage["games_played"] >= MIN_GP_USAGE)]
    last = u.sort_values("season").groupby("playerId")["fo_60"].last()
    fo = pd.Series(players).map(last)
    return np.where(fo.notna(), fo >= CENTER_FO_60, pd.Series(positions).to_numpy() == "C")


def rookie_defaults(usage, experience, seasons):
    """
    Rule R1: median usage of true rookies (no NHL experience before that season) in `seasons`, per group.
    Skaters need 10+ games, so it's a typical rookie's ice time per game; goalies, games per 82.
    experience: DataFrame (playerId, season, experience) from development.experience.
    """
    u = usage[usage["season"].isin(seasons)].merge(experience, on=["playerId", "season"], how="left")
    ok = np.where(u["group"] == "G", u["games_played"] >= 1, u["games_played"] >= MIN_GP_USAGE)
    rk = u[ok & (u["experience"] == 0)]
    return rk.groupby("group")["usage"].median()


def fill_usage(pool, rule, defaults):
    """
    toi_input = his own usage last season, or, for players without it:
      R1: the rookie default for his group (a typical rookie's usage);
      R2: expected usage from his projection (a straight line fit, per group, of usage on projected WAR/60
          among this pool's players who have their own usage).
    toi_rule says which one each player got.
    """
    out = pool["usage_last"].copy()
    rule_col = np.where(out.notna(), "own", rule)
    for g, d in pool.groupby("group"):
        miss = d.index[d["usage_last"].isna()]
        if not len(miss):
            continue
        if rule == "R1":
            out[miss] = defaults[g]
        else:
            has = d[d["usage_last"].notna()]
            slope, intercept = np.polyfit(has["proj_war_60"], has["usage_last"], 1)
            out[miss] = np.clip(intercept + slope * d.loc[miss, "proj_war_60"], 0, None)
    return out, rule_col


# ---------------------------------------------------------------------------
# 3. Coach score
# ---------------------------------------------------------------------------
def coach_scores(pool, settings, defaults):
    """
    Adds toi_input, toi_rule, coach_score and three sort keys (key1-key3) to every pool player.
    pool needs usage_last (see last_usage) and proj_war_60; d_rank_prior is optional.
    settings: {group: (w, rule)}, w in [0, 1] or USAGE_FIRST; rule "R1" or "R2".
    z-scores are taken across the whole pool of that group (all teams), so a player's score doesn't depend on
    his team. Higher is better. Ties are broken by the projection, then by d_rank_prior (defense prospects all
    share notebook 19's position-average projection, so their draft-based prior orders them).
    USAGE_FIRST: usage rounded to the nearest minute (goalies: nearest 5 games per 82) decides; the projection
    orders players within the same minute.
    """
    p = pool.copy()
    p["toi_input"], p["toi_rule"] = np.nan, ""
    p["coach_score"], p["key1"] = np.nan, np.nan
    for g, (w, rule) in settings.items():
        m = p["group"] == g
        if not m.any():
            continue
        toi, how = fill_usage(p[m], rule, defaults)
        p.loc[m, "toi_input"], p.loc[m, "toi_rule"] = toi, how
        z = lambda x: (x - x.mean()) / x.std()
        if w == USAGE_FIRST:
            step = 5.0 if g == "G" else 1.0
            p.loc[m, "coach_score"] = np.round(p.loc[m, "toi_input"] / step) * step
        else:
            p.loc[m, "coach_score"] = w * z(p.loc[m, "toi_input"]) + (1 - w) * z(p.loc[m, "proj_war_60"])
    p["key1"] = p["coach_score"]
    p["key2"] = p["proj_war_60"]
    p["key3"] = p["d_rank_prior"].fillna(-np.inf) if "d_rank_prior" in p else -np.inf
    return p


def _order(d, tiered=False):
    """Best first: coach score, projection, d_rank_prior, then playerId (so the order is always the same).
    tiered=True: the NHL level (tier 1) comes before AHL/ECHL players on NHL contracts (tier 2), then the same order."""
    if tiered:
        tier = np.where(d["status"].isin(AHL_STATUSES), 2, 1) if "status" in d else np.ones(len(d))
        return (d.assign(_tier=tier).sort_values(["_tier", "key1", "key2", "key3", "playerId"],
                                                 ascending=[True, False, False, False, True]).drop(columns="_tier"))
    return d.sort_values(["key1", "key2", "key3", "playerId"], ascending=[False, False, False, True])


# ---------------------------------------------------------------------------
# 4. Picking a lineup
# ---------------------------------------------------------------------------
def pick_lineup(team_pool, tiered=False):
    """
    One team's lineup from its pool (already scored). Returns every pool player with:
      dressed (bool), rank (1 = best in his group; dressed players first, then extras),
      slot (F1-F12, D1-D6, G1-G2 or "extra"), line (1-4 forwards, 1-3 pairs, 1-2 goalies), role (C, W, D, G),
      flag (text, empty if nothing unusual).
    Forwards: the top 12; if fewer than 4 are natural centers, the best center outside the 12 replaces the
    lowest-ranked winger, as often as needed. If no center is left, a winger plays center (flagged).
    tiered=True (call-ups only when needed, notebook 24): the NHL level fills the slots first; AHL/ECHL players on
    NHL contracts dress only for slots the NHL level can't fill (a missing 4th center included). Once dressed,
    players are ranked (and get minutes) by coach score alone. Default False = notebook 23's rule.
    """
    parts = []
    for g in ["F", "D", "G"]:
        d = _order(team_pool[team_pool["group"] == g], tiered).copy()
        d["flag"] = ""
        n = N_DRESSED[g]
        dressed = list(d.index[:n])
        if g == "F":
            centers = [i for i in dressed if d.at[i, "natural_center"]]
            outside = [i for i in d.index[n:] if d.at[i, "natural_center"]]
            while len(centers) < N_CENTERS and outside:
                wingers = [i for i in dressed if not d.at[i, "natural_center"]]
                out_i, in_i = wingers[-1], outside.pop(0)
                dressed.remove(out_i)
                dressed.append(in_i)
                centers.append(in_i)
                d.at[in_i, "flag"] = f"center rule: dressed over {d.at[out_i, 'name']}"
                d.at[out_i, "flag"] = f"center rule: sits for {d.at[in_i, 'name']}"
        d["dressed"] = d.index.isin(dressed)
        d = pd.concat([_order(d[d["dressed"]]), _order(d[~d["dressed"]], tiered)])
        d["rank"] = np.arange(1, len(d) + 1)
        d["slot"] = np.where(d["dressed"], g + d["rank"].astype(str), "extra")
        d["line"], d["role"] = np.nan, g if g != "F" else "W"
        if g == "F":
            top = d[d["dressed"]]
            cs = list(top.index[top["natural_center"]][:N_CENTERS])
            for i in top.index[~top["natural_center"]]:
                if len(cs) >= N_CENTERS:
                    break
                cs.append(i)                                   # a winger has to play center
                d.at[i, "flag"] = (d.at[i, "flag"] + "; " if d.at[i, "flag"] else "") + "winger at center"
            wings = [i for i in top.index if i not in cs]
            for k, i in enumerate(cs):
                d.at[i, "line"], d.at[i, "role"] = k + 1, "C"
            for k, i in enumerate(wings):
                d.at[i, "line"] = k // 2 + 1
        elif g == "D":
            d.loc[d["dressed"], "line"] = (d.loc[d["dressed"], "rank"] - 1) // 2 + 1
        else:
            d.loc[d["dressed"], "line"] = d.loc[d["dressed"], "rank"]
        if d["dressed"].sum() < n:                            # not enough players in the pool
            short = d["dressed"] & (d["flag"] == "")
            d.loc[short, "flag"] = f"only {int(d['dressed'].sum())} {g} in the pool"
        parts.append(d)
    return pd.concat(parts)


# ---------------------------------------------------------------------------
# 5. Minutes per slot
# ---------------------------------------------------------------------------
def slot_minutes(usage, seasons, min_gp=MIN_GP_POOL):
    """
    Raw table: the median ice time per game of each team's k-th forward (k = 1-16) and k-th defenseman (1-9),
    ranking the players with min_gp+ games for the team that season by ice time per game.
    Returns a DataFrame indexed by (group, rank) with column 'raw'.
    """
    u = usage[usage["season"].isin(seasons) & (usage["group"] != "G") & (usage["games_played"] >= min_gp)].copy()
    u["rank"] = u.groupby(["season", "team", "group"])["usage"].rank(ascending=False, method="first").astype(int)
    rows = []
    for g in ["F", "D"]:
        med = u[(u["group"] == g) & (u["rank"] <= MAX_RANK[g])].groupby("rank")["usage"].median()
        rows.append(pd.DataFrame({"group": g, "rank": med.index, "raw": med.to_numpy()}))
    return pd.concat(rows).set_index(["group", "rank"])


def team_minutes(usage, seasons):
    """A team's forwards' and defensemen's minutes per game (all situations, overtime included):
    the league's total minutes / (teams x games), per season, then the average over `seasons`."""
    out = []
    for s in seasons:
        u = usage[(usage["season"] == s) & (usage["group"] != "G")]
        teams = usage.loc[(usage["season"] == s) & (usage["group"] == "G"), "team"].nunique()
        out.append(u.groupby("group")["all_toi"].sum() / (teams * season_games(s)[0]))
    return pd.concat(out, axis=1).mean(axis=1)


def scaled_slots(table, totals):
    """Ranks 1-12 (F) and 1-6 (D) scaled so a dressed lineup adds up to a real game's minutes (`totals`)."""
    t = table.copy()
    t["scaled"] = np.nan
    for g in ["F", "D"]:
        idx = [(g, k) for k in range(1, N_DRESSED[g] + 1)]
        t.loc[idx, "scaled"] = t.loc[idx, "raw"] * totals[g] / t.loc[idx, "raw"].sum()
    return t


def starter_share(usage, seasons):
    """The starter's share of his team's top two goalies' games, every team-season in `seasons`.
    Returns (per team-season shares, median over all of them)."""
    g = usage[usage["season"].isin(seasons) & (usage["group"] == "G")]
    g = g.assign(rk=g.groupby(["season", "team"])["games_played"].rank(ascending=False, method="first"))
    top = g[g["rk"] <= 2].pivot_table(index=["season", "team"], columns="rk", values="games_played")
    shares = (top[1.0] / (top[1.0] + top[2.0])).dropna().rename("starter_share")
    return shares, float(shares.median())


def assign_minutes(lineup, slots, share, col="scaled", extras=False, own_weight=0.0):
    """
    minutes_per_game for every player in a lineup table (from pick_lineup):
    skaters by their rank in the slot table's column `col` ("scaled" for 2026-27, "raw" in the backtest);
    goalies 60 x share (starter) or 60 x (1 - share) (backup).
    extras=False: players who don't dress play 0. extras=True (backtest): they get the minutes for their rank
    too (ranks past the table get its last value).
    own_weight (notebook 24; a number, or {"F": ..., "D": ...}): blend the slot with each skater's own recent ice
    time per game, (1 - own_weight) x slot + own_weight x own (players without their own usage keep the slot). Then each team's
    dressed forwards and defensemen are rescaled so they still add up to the slots' total. 0 = notebook 23.
    """
    out = pd.Series(0.0, index=lineup.index)
    for g in ["F", "D"]:
        m = (lineup["group"] == g) & (lineup["dressed"] | extras)
        tab = slots.loc[g][col].dropna()                      # minutes by rank
        out[m] = lineup.loc[m, "rank"].clip(upper=tab.index.max()).map(tab).to_numpy()
    weights = own_weight if isinstance(own_weight, dict) else {"F": own_weight, "D": own_weight}
    if any(weights.values()):
        sk = (lineup["group"] != "G") & (out > 0)
        own = lineup["toi_input"].where(lineup["toi_rule"] == "own", out)
        wt = lineup["group"].map(weights).fillna(0.0)
        blend = (1 - wt) * out + wt * own
        keys = [c for c in ["season", "team"] if c in lineup] + ["group"]
        dr = sk & lineup["dressed"]
        slot_sum = out[dr].groupby([lineup.loc[dr, k] for k in keys]).transform("sum")
        blend_sum = blend[dr].groupby([lineup.loc[dr, k] for k in keys]).transform("sum")
        out[sk] = blend[sk]
        out[dr] = blend[dr] * slot_sum / blend_sum
    gm = lineup["group"] == "G"
    out[gm & (lineup["rank"] == 1)] = 60 * share
    out[gm & (lineup["rank"] == 2)] = 60 * (1 - share)
    return out


# ---------------------------------------------------------------------------
# 6. 2026-27: the players, trades, sit-outs, all 32 lineups, strength
# ---------------------------------------------------------------------------
def load_players(data_dir, usage, projections, lookback, season=2026):
    """
    Every 2026-27 player with his opening-night team and status (rosters_<label>.csv, notebook 22), his
    projection (forecast.current_projections; d_rank_prior for defense prospects), his usage (last_usage with
    each group's lookback), and whether he's a natural center.
    in_pool: his status puts him in his team's pool (POOL_STATUSES).
    """
    label = f"{season}-{str(season + 1)[-2:]}"
    r = pd.read_csv(Path(data_dir) / f"rosters_{label}.csv")
    p = projections.set_index("playerId")
    out = r[["playerId", "name", "position", "team_opening_night", "status"]].rename(columns={"team_opening_night": "team"})
    out["group"] = out["playerId"].map(p["group"])
    out["proj_war_60"] = out["playerId"].map(p["proj_war_60"])
    out["d_rank_prior"] = out["playerId"].map(p["d_rank_prior"])
    out["usage_last"] = usage_by_group(usage, out, season, lookback)
    gp = usage[usage["season"] == season - 1].set_index("playerId")["games_played"]
    out["gp_last"] = out["playerId"].map(gp)
    out["natural_center"] = natural_center(usage, out["playerId"], out["position"], season) & (out["group"] == "F")
    out["in_pool"] = out["status"].isin(POOL_STATUSES) & out["team"].notna()
    return out


def dispute_ids(players):
    """Players whose status is a dispute (Larkin: suspended / holdout; Edvinsson: unsigned RFA)."""
    return list(players.loc[players["status"].isin(DISPUTE_STATUSES), "playerId"])


def apply_trades(players, trades, trade_ids):
    """Moves every player in the listed trades to his to_team. Statuses don't change. trade_ids=[]: no change."""
    p = players.copy()
    t = trades[trades["trade_id"].isin(trade_ids)]
    missing = set(trade_ids) - set(trades["trade_id"])
    if missing:
        raise ValueError(f"unknown trade ids: {sorted(missing)}")
    p.loc[p["playerId"].isin(t["playerId"]), "team"] = p["playerId"].map(t.set_index("playerId")["to_team"])
    return p


def team_pool(players, team, sit_out=()):
    """A team's pool: its players with a pool status, minus anyone sitting out."""
    return players[(players["team"] == team) & players["in_pool"] & ~players["playerId"].isin(list(sit_out))]


def build_lineups(scored, slots, share, trades=None, trade_ids=(), sit_out=(), tiered=False, own_weight=0.0):
    """
    All 32 lineups from scored players (coach_scores already applied to the whole league pool, so scores don't
    move when a trade or a sit-out changes a team). Returns one row per pool player with minutes and WAR.
    tiered, own_weight: notebook 24's call-up tiers and minutes blend (see pick_lineup, assign_minutes);
    the defaults are notebook 23's rules.
    """
    p = scored if not len(trade_ids) else apply_trades(scored, trades, list(trade_ids))
    teams = sorted(p.loc[p["in_pool"], "team"].dropna().unique())
    out = []
    for team in teams:
        lu = pick_lineup(team_pool(p, team, sit_out), tiered)
        lu["team"] = team
        out.append(lu)
    out = pd.concat(out)
    out["minutes_per_game"] = assign_minutes(out, slots, share, own_weight=own_weight)
    out["war_per_game"] = out["proj_war_60"] * out["minutes_per_game"] / 60
    out["war_per_84"] = out["war_per_game"] * SEASON_GAMES_2026
    return out.reset_index(drop=True)


def lineup_strength(lineups):
    """Per team: WAR per game (F, D, G and total) and per 84 games, from the dressed players."""
    d = lineups[lineups["dressed"]]
    s = d.pivot_table(index="team", columns="group", values="war_per_game", aggfunc="sum").fillna(0)
    s = s[["F", "D", "G"]]
    s["war_per_game"] = s.sum(axis=1)
    s["war_per_84"] = s["war_per_game"] * SEASON_GAMES_2026
    return s.sort_values("war_per_game", ascending=False)
