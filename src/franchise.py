"""
Franchise mode (notebook 30): set any team's lineup, play a stretch of games, and see the results two ways:
one simulated season as a story, and 10,000 seasons as a projection for the same choices.

    import franchise as FR
    inp = FR.load_inputs(DATA / "simulator_2026-27", DATA / "trades_2026-27.csv", "2026-11-16", use_update=True,
                         cache_root=DATA / "simulator_2026-27" / "updated")
    base = FR.base_lineups(inp, sit_out=[8477946, 8482762])
    det = FR.swap_lines(FR.lineup_table(base, "DET"), "DET", "F", 1, 4)          # a franchise lineup
    s = FR.story(inp, "DET", n_games=10, sit_out=[...], overrides=det, seed=0)    # one season, re-roll with seed
    p = FR.projection(inp, "DET", n_games=10, sit_out=[...], overrides=det)       # 10,000 seasons

Nothing here is a new model. The pieces come from the files that were built and checked before:
  - season.run_scenario plays the games (lineups, random seasons, team bonus, stat lines, standings);
  - lineups.assign_minutes turns slot order into minutes;
  - refresh.load_update_cache gives the in-season update (notebook 29), with the simulator switch's rules.
This file only decides what goes in (lineups, the game window) and how the results are shown.

What a lineup change can and can't do: a team's strength is the sum of each dressed player's projected value
(WAR/60) x his minutes. So a change matters through who dresses and how many minutes each player gets. Who his
linemates are doesn't matter: there is no chemistry in the model. A "line" is just three slots in a row (F1-F3 is
line 1, the three forwards with the most minutes); a D pair is two slots (D1-D2).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import lineups as L
import refresh as R
import scoring as SC
import season as SE

SLOTS = [f"F{k}" for k in range(1, 13)] + [f"D{k}" for k in range(1, 7)] + ["G1", "G2"]
LINE_SIZE = {"F": 3, "D": 2, "G": 1}
MINUTES_RULES = ["moved", "blend", "slot"]
OVERRIDE_COLS = ["team", "slot", "playerId"]
NOT_SIMULATED = ("Only what the engine simulates: goals, assists, wins and goals against. No shots, save %, ice time "
                 "or +/-. GF counts goals players scored, so a shootout win is +1 in the score but not in GF.")


def _slot_group(slot):
    return str(slot)[0]


def _slot_number(slot):
    return int(str(slot)[1:])


# ---------------------------------------------------------------------------
# 1. Inputs: the bundle, with or without the in-season update (the simulator switch's rules)
# ---------------------------------------------------------------------------
def load_inputs(bundle_dir, trades_path, start, use_update=False, cache_root=None):
    """
    What a franchise run plays from. The same rules as the simulator's "Use 2026-27 results so far" switch:
      - use_update=True: the newest update cache (notebook 29): its real results, each player's updated draws and
        the updated team bonus. Only for a start on or after the cache's start (earlier, the update has already
        seen games this run would simulate).
      - use_update=False: preseason ratings. After the bundle's download day, the real results come from the
        newest cache (the bundle doesn't have them); before it, from the bundle, exactly as notebooks 26-27.
    Returns a dict: bundle, updated (None or run_scenario's updated), start, ratings (a short note), cache_meta.
    """
    B = SE.load_bundle(bundle_dir, trades_path)
    last_day = B["schedule_downloaded"][:10]
    cache = R.newest_cache(cache_root) if cache_root is not None else None
    if use_update or start > last_day:
        if cache is None:
            raise ValueError("this needs an update cache (notebook 29's last cell), and there isn't one"
                             if use_update else f"a start after {last_day} needs real results from an update cache "
                                                "(notebook 29), and there isn't one")
        Bn, upd, meta = R.load_update_cache(cache, B, bundle_dir)
        if use_update and start < meta["start"]:
            raise ValueError(f"the update through {meta['evidence_through']} can only be used from {meta['start']} on: "
                             "from an earlier start it would already know games being simulated")
        ratings = (f"updated through {meta['evidence_through']} (notebook 29)" if use_update
                   else f"preseason (real results through {meta['evidence_through']} from notebook 29)")
        return {"bundle": Bn, "updated": upd if use_update else None, "start": start, "ratings": ratings,
                "cache_meta": meta}
    return {"bundle": B, "updated": None, "start": start, "ratings": "preseason", "cache_meta": None}


def check_start(bundle, start):
    """A start date must not skip games: every game before it has to be a real, finished game."""
    s = bundle["schedule"]
    gap = s[(s["date"] < start) & ~s["played"]]
    if len(gap):
        raise ValueError(f"{len(gap)} games before {start} have no real result in these inputs (first: "
                         f"{gap['date'].min()}); start on or before the day after the last real results")


# ---------------------------------------------------------------------------
# 2. Lineups: the baseline, the override table, and applying it
# ---------------------------------------------------------------------------
def base_lineups(inputs, trade_ids=None, sit_out=()):
    """All 32 baseline lineups for the inputs' start date (season.scenario_lineups: trades default to those dated on
    or before the start)."""
    B = inputs["bundle"]
    trades = SE.default_trades(B["trades"], inputs["start"]) if trade_ids is None else sorted(trade_ids)
    return SE.scenario_lineups(B, trades, sit_out)


def lineup_table(lineups, team):
    """A team's 20 dressed players as an override table (team, slot, playerId, name, position), in slot order.
    The baseline's slot is its rank, so passing this table back unchanged gives the baseline exactly."""
    d = lineups[(lineups["team"] == team) & lineups["dressed"]]
    t = d[["team", "slot", "playerId", "name", "position"]].copy()
    t["_k"] = t["slot"].map({s: i for i, s in enumerate(SLOTS)})
    return t.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def _swap(table, team, a, b):
    t = table.copy()
    ia = t.index[(t["team"] == team) & (t["slot"] == a)]
    ib = t.index[(t["team"] == team) & (t["slot"] == b)]
    if len(ia) != 1 or len(ib) != 1:
        raise ValueError(f"{team}: slots {a} and {b} must each be in the table once")
    cols = [c for c in t.columns if c not in ("team", "slot")]
    va, vb = t.loc[ia, cols].to_numpy(), t.loc[ib, cols].to_numpy()
    t.loc[ia, cols], t.loc[ib, cols] = vb, va
    return t


def swap_slots(table, team, a, b):
    """The players in slots a and b trade places (for example "G1" and "G2": the backup becomes the starter)."""
    return _swap(table, team, a, b)


def swap_lines(table, team, group, i, j):
    """Forward lines (3 slots each) or D pairs (2 slots each) i and j trade places, slot by slot."""
    n = LINE_SIZE[group]
    t = table
    for k in range(n):
        t = _swap(t, team, f"{group}{(i - 1) * n + k + 1}", f"{group}{(j - 1) * n + k + 1}")
    return t


def replace(table, team, out_id, in_id, lineups=None):
    """in_id takes out_id's slot (out_id sits). lineups (optional) fills in the new player's name and position."""
    t = table.copy()
    i = t.index[(t["team"] == team) & (t["playerId"] == out_id)]
    if len(i) != 1:
        raise ValueError(f"{team}: player {out_id} isn't in the table")
    t.loc[i, "playerId"] = in_id
    if lineups is not None:
        who = lineups.drop_duplicates("playerId").set_index("playerId")
        for c in ["name", "position"]:
            if c in t:
                t.loc[i, c] = who[c].get(in_id, np.nan)
    return t


def write_overrides(table, path, goalie_share=None):
    """Save an override table as CSV (team, slot, playerId; start_share on the G1 row when a share is set).
    This is the format the app can write later."""
    t = table[OVERRIDE_COLS].copy()
    t["start_share"] = [goalie_share.get(tm) if goalie_share and s == "G1" else np.nan for tm, s in zip(t["team"], t["slot"])]
    t.to_csv(path, index=False)
    return Path(path)


def read_overrides(path):
    """(overrides, goalie_share) from write_overrides' CSV."""
    t = pd.read_csv(path)
    share = {tm: float(v) for tm, v in zip(t["team"], t.get("start_share", pd.Series(np.nan, index=t.index))) if pd.notna(v)}
    return t[OVERRIDE_COLS], share


def check_overrides(lineups, overrides=None, goalie_share=None, players=None):
    """
    (errors, warnings) for an override table against the baseline lineups (after trades and sit-outs).
    Errors (a run refuses): a team that doesn't list F1-F12, D1-D6, G1, G2 exactly once; a player listed twice; a
    player not in that team's pool (traded away, sitting out, another team's); a goalie in a skater slot or a skater
    in a goalie slot; a start share outside 0 to 1. players (optional, the bundle's): names players who aren't
    in any lineup (sitting out) in the messages.
    Warnings (a run goes ahead): fewer than 4 natural centers among the 12 forwards, a skater out of position (a
    defenseman at forward or the other way), a call-up (AHL/ECHL player) dressed.
    """
    errors, warnings = [], []
    teams = set(lineups["team"])
    pool = lineups.drop_duplicates("playerId").set_index("playerId")
    everyone = (players if players is not None else lineups).drop_duplicates("playerId").set_index("playerId")
    if overrides is not None and len(overrides):
        o = overrides
        for team, d in o.groupby("team", sort=True):
            if team not in teams:
                errors.append(f"{team}: not a team")
                continue
            counts = d["slot"].value_counts()
            missing = [s for s in SLOTS if s not in counts]
            extra = sorted(set(counts.index) - set(SLOTS)) + sorted(s for s, c in counts.items() if c > 1 and s in SLOTS)
            if missing or extra:
                errors.append(f"{team}: needs F1-F12, D1-D6, G1 and G2 once each"
                              + (f"; missing {missing}" if missing else "") + (f"; extra or twice {extra}" if extra else ""))
            twice = d["playerId"][d["playerId"].duplicated()].unique()
            if len(twice):
                errors.append(f"{team}: listed twice: {[pool['name'].get(p, p) for p in twice]}")
            mine = lineups[lineups["team"] == team]
            for p, s in zip(d["playerId"], d["slot"]):
                if p not in set(mine["playerId"]):
                    where = pool["team"].get(p)
                    errors.append(f"{team} {s}: {everyone['name'].get(p, p)} isn't in {team}'s pool"
                                  + (f" (he's {where}'s)" if where is not None else
                                     " (he's sitting out)" if p in everyone.index else " (not a pool player)"))
                    continue
                g = pool.at[p, "group"]
                sg = _slot_group(s)
                if (g == "G") != (sg == "G"):
                    errors.append(f"{team} {s}: {pool.at[p, 'name']} is a {'goalie' if g == 'G' else 'skater'}; "
                                  "goalies and skaters can't trade slots")
                elif g != sg:
                    warnings.append(f"{team} {s}: {pool.at[p, 'name']} ({pool.at[p, 'position']}) plays out of "
                                    f"position; he gets {sg} minutes and keeps his own value")
            fwd = d[d["slot"].astype(str).str.startswith("F") & d["playerId"].isin(mine["playerId"])]
            centers = int(pool["natural_center"].reindex(fwd["playerId"]).fillna(False).astype(bool).sum())
            if centers < L.N_CENTERS:
                warnings.append(f"{team}: only {centers} natural centers among the forwards (the coach rule wants "
                                f"{L.N_CENTERS}); the model doesn't penalize it, but it isn't a real lineup")
            ups = d[d["playerId"].map(pool["status"]).isin(L.AHL_STATUSES)]
            if len(ups):
                warnings.append(f"{team}: call-ups dressed: {', '.join(pool['name'].reindex(ups['playerId']))}")
    for team, s in (goalie_share or {}).items():
        if team not in teams:
            errors.append(f"{team}: not a team (start share)")
        elif not 0 <= s <= 1:
            errors.append(f"{team}: start share {s} must be between 0 and 1")
    return errors, warnings


def apply_overrides(lineups, bundle, overrides=None, goalie_share=None, minutes_rule="moved"):
    """
    The lineups with franchise changes, and the matching backup-goalie games. Returns (lineups, goalie_starts,
    warnings). Raises ValueError (listing every problem) if check_overrides finds errors.
      overrides: team, slot, playerId rows (a team with overrides lists all 20 slots). Slot number = rank, and
        minutes follow rank exactly as lineups.assign_minutes does, under minutes_rule:
          "moved" (Ryan's rule, the default): a player whose slot changed (a newly dressed player included) gets
            100% slot minutes; everyone else keeps the 75/25 blend of slot and his own recent ice time; then the
            team's forwards and defensemen are rescaled to the slot totals;
          "blend": everyone keeps the 75/25 blend (notebook 24); "slot": the whole team gets slot minutes only.
        A skater out of position gets his slot's minutes but keeps his own value and random draw.
      goalie_share: {team: starter's share of games}. That team's goalies play 60 x share and 60 x (1 - share)
        minutes, and its backup's games are rebuilt with season.goalie_starts at that share (back-to-backs first).
    No overrides and no share: the baseline lineups and goalie games, unchanged.
    """
    if minutes_rule not in MINUTES_RULES:
        raise ValueError(f"minutes_rule must be one of {MINUTES_RULES}")
    errors, warnings = check_overrides(lineups, overrides, goalie_share, bundle["players"])
    if errors:
        raise ValueError("The franchise lineup can't be used:\n  - " + "\n  - ".join(errors))
    starts = bundle["goalie_starts"]
    if (overrides is None or not len(overrides)) and not goalie_share:
        return lineups, starts, warnings
    lu = lineups.copy()
    o = overrides if overrides is not None else pd.DataFrame(columns=OVERRIDE_COLS)
    changed = sorted(set(o["team"]))
    weights = pd.Series(np.nan, index=lu.index)
    for team in changed:
        rows = lu.index[lu["team"] == team]
        new_slot = lu.loc[rows, "playerId"].map(o[o["team"] == team].set_index("playerId")["slot"])
        moved = (new_slot.fillna("extra") != lu.loc[rows, "slot"]).to_numpy()
        lu.loc[rows, "dressed"] = new_slot.notna().to_numpy()
        lu.loc[rows, "slot"] = new_slot.fillna("extra").to_numpy()
        # Extras keep their order, after the dressed players of their group
        for g in ["F", "D", "G"]:
            ex = lu.loc[rows][(lu.loc[rows, "group"] == g) & ~lu.loc[rows, "dressed"]].sort_values("rank")
            lu.loc[ex.index, "rank"] = L.N_DRESSED[g] + np.arange(1, len(ex) + 1)
        dr = rows[lu.loc[rows, "dressed"].to_numpy()]
        lu.loc[dr, "rank"] = lu.loc[dr, "slot"].map(_slot_number).to_numpy()
        sg = lu.loc[dr, "slot"].map(_slot_group)
        lu.loc[dr, "line"] = [(_slot_number(s) - 1) // LINE_SIZE[g] + 1 for s, g in zip(lu.loc[dr, "slot"], sg)]
        lu.loc[dr, "role"] = np.where(sg == "F", np.where(lu.loc[dr, "natural_center"].astype(bool), "C", "W"), sg)
        lu.loc[rows[~lu.loc[rows, "dressed"].to_numpy()], ["line", "role"]] = [np.nan, ""]
        lu.loc[rows, "flag"] = np.where(lu.loc[rows, "dressed"] & (lu.loc[rows, "group"] != lu.loc[rows, "slot"].map(_slot_group)),
                                        "franchise: out of position", "franchise lineup")
        if minutes_rule == "moved":
            weights[rows[moved]] = 0.0
        elif minutes_rule == "slot":
            weights[rows] = 0.0

    # Minutes: assign_minutes on the changed teams, with each dressed player in his slot's group
    teams_m = sorted(set(changed) | set(goalie_share or {}))
    m_rows = lu.index[lu["team"].isin(changed)]
    if len(m_rows):
        tmp = lu.loc[m_rows].copy()
        dr = tmp["dressed"].astype(bool)
        tmp.loc[dr, "group"] = tmp.loc[dr, "slot"].map(_slot_group)
        w = weights[m_rows] if minutes_rule != "blend" else None
        lu.loc[m_rows, "minutes_per_game"] = L.assign_minutes(tmp, bundle["slots"], bundle["share"],
                                                              own_weight=SE.OWN_WEIGHT, own_weight_by_player=w)
    for team, share in (goalie_share or {}).items():
        g = lu.index[(lu["team"] == team) & (lu["group"] == "G")]
        lu.loc[g, "minutes_per_game"] = np.select([lu.loc[g, "rank"] == 1, lu.loc[g, "rank"] == 2],
                                                  [60 * share, 60 * (1 - share)], 0.0)
        alt = SE.goalie_starts(bundle["schedule"], share)
        starts = starts.copy()
        s = bundle["schedule"]
        for side in ["home", "away"]:
            mine = (s[side] == team).to_numpy()
            starts.loc[mine, f"{side}_backup"] = alt.loc[mine, f"{side}_backup"]
    rows = lu.index[lu["team"].isin(teams_m)]
    lu.loc[rows, "war_per_game"] = lu.loc[rows, "proj_war_60"] * lu.loc[rows, "minutes_per_game"] / 60
    lu.loc[rows, "war_per_84"] = lu.loc[rows, "war_per_game"] * L.SEASON_GAMES_2026
    lu["dressed"] = lu["dressed"].astype(bool)
    return lu, starts, warnings


# ---------------------------------------------------------------------------
# 3. The game range
# ---------------------------------------------------------------------------
def game_range(schedule, start, team, n_games=None, end_date=None):
    """
    The window to play: from start through the date of `team`'s n-th game (n_games), or through end_date, or (both
    None) the rest of the season. Every game in the window is played, for all 32 teams.
    full_season: the window reaches the last scheduled game; only then are there seeds and playoff odds.
    """
    if n_games is not None and end_date is not None:
        raise ValueError("give n_games or end_date, not both")
    last = schedule["date"].max()
    mine = schedule[((schedule["home"] == team) | (schedule["away"] == team)) & (schedule["date"] >= start)]
    mine = mine.sort_values("date", kind="stable")
    if team not in set(schedule["home"]) | set(schedule["away"]):
        raise ValueError(f"{team} isn't in the schedule")
    if n_games is not None:
        if not 1 <= n_games <= len(mine):
            raise ValueError(f"{team} has {len(mine)} games left from {start}; n_games must be 1 to {len(mine)}")
        end = mine["date"].iloc[n_games - 1]
    elif end_date is not None:
        if end_date < start:
            raise ValueError("end_date is before start")
        end = min(end_date, last)
    else:
        end = last
    window = schedule[(schedule["date"] >= start) & (schedule["date"] <= end)]
    gp = pd.concat([window["home"], window["away"]]).value_counts()
    return {"start": start, "end": end, "full_season": end >= last, "games": len(window),
            "team_games": int(gp.get(team, 0)), "gp": gp}


def window_bundle(bundle, end, goalie_starts):
    """The bundle with the schedule (and the matching backup-goalie games, worked out on the full season first) cut
    after `end`, so run_scenario plays exactly the window."""
    keep = (bundle["schedule"]["date"] <= end).to_numpy()
    return {**bundle, "schedule": bundle["schedule"][keep], "goalie_starts": goalie_starts[keep]}


# ---------------------------------------------------------------------------
# 4. Running the choices
# ---------------------------------------------------------------------------
def play(inputs, team, n_games=None, end_date=None, trade_ids=None, sit_out=(), overrides=None, goalie_share=None,
         minutes_rule="moved", n_sims=10000, seed=0, story=False, keep_z=None):
    """One run with franchise choices: run_scenario on the window, with the franchise lineups. story=True keeps
    every game and every player's draw, and shifts updated players' draws by the seed (a new seed re-rolls them).
    keep_z: keep every player's draw without the games (notebook 31's many-season progression); None = story."""
    B, start = inputs["bundle"], inputs["start"]
    check_start(B, start)
    trades = SE.default_trades(B["trades"], start) if trade_ids is None else sorted(trade_ids)
    base = SE.scenario_lineups(B, trades, sit_out)
    lu, starts, warnings = apply_overrides(base, B, overrides, goalie_share, minutes_rule)
    rng = game_range(B["schedule"], start, team, n_games, end_date)
    Bw = window_bundle(B, rng["end"], starts)
    res = SE.run_scenario(Bw, start_date=start, trade_ids=trades, sit_out=sit_out, n_sims=n_sims, seed=seed,
                          stat_lines=True, updated=inputs["updated"], lineups=lu, keep_games=story, keep_z=story if keep_z is None else keep_z,
                          update_shift=seed if story else 0)
    res.update({"range": rng, "warnings": warnings, "bundle_used": Bw, "team": team, "ratings": inputs["ratings"],
                "minutes_rule": minutes_rule, "overrides": overrides, "goalie_share": goalie_share or {}})
    return res


def _record(res):
    """Per team and simulated season, the range only: W, L, OTL, points, GP (real results before the start
    taken back out of run_scenario's tallies)."""
    real = SE.real_so_far(res["bundle_used"]["schedule"], res["start_date"]).reindex(res["teams"]).fillna(0)
    gp = res["range"]["gp"].reindex(res["teams"]).fillna(0).to_numpy()
    t = res["tallies"]
    out = {k: t[k] - real[k].to_numpy()[None, :] for k in ["points", "wins", "otl", "rw", "row", "gd"]}
    out["gp"] = np.broadcast_to(gp, out["points"].shape)
    out["losses"] = out["gp"] - out["wins"] - out["otl"]
    return out, real


def _names(bundle):
    return bundle["players"].drop_duplicates("playerId").set_index("playerId")["name"]


def _two_views(res, n_one):
    """Skater and goalie lines two ways: range only (the simulated window) and season to date (real stats before
    the start + the range, season.season_lines' rule)."""
    sk_season, g_season = SE.season_lines(res, res["bundle_used"])
    sk = res["skaters"].copy()
    gk = res["goalies"].copy()
    gk["sim_gaa"] = np.where(gk["sim_gp"] > 0, gk["sim_ga"] / (60 * gk["sim_gp"].where(gk["sim_gp"] > 0, 1)) * 60, np.nan)
    if n_one:                                   # one season: whole numbers
        for d in (sk, sk_season, gk, g_season):
            cols = [c for c in d.columns if (c.startswith(("sim_", "real_")) or c in ("gp", "g", "a", "p", "w", "l", "otl", "ga"))
                    and c not in ("sim_gaa", "real_min")]
            d[cols] = d[cols].round().astype(int)
    return {"range_skaters": sk, "range_goalies": gk, "season_skaters": sk_season, "season_goalies": g_season}


def _leaders(sk, cols, n=10):
    """Top n by points, goals and assists (ties: the other stat, then name)."""
    g, a, p = cols
    return {"points": sk.sort_values([p, g, "name"], ascending=[False, False, True]).head(n),
            "goals": sk.sort_values([g, p, "name"], ascending=[False, False, True]).head(n),
            "assists": sk.sort_values([a, p, "name"], ascending=[False, False, True]).head(n)}


def story(inputs, team, n_games=None, end_date=None, trade_ids=None, sit_out=(), overrides=None, goalie_share=None,
          minutes_rule="moved", seed=0):
    """
    ONE simulated season with the franchise choices, as a story (seeded; a new seed re-rolls it):
      game_log (the team's games: date, opponent, home/away, score, OT/SO, result, starting goalie, record so far),
      record (W, L, OTL, points, GF, GA, points pace per 84), standings at the end of the range (real before the
      start + the range; seeds only for a full season), skater and goalie lines and leaders two ways (range only;
      season to date = real so far + range, which for a full season is the final season), and draws: every pool
      player's random season (z), for notebook 31.
    """
    res = play(inputs, team, n_games, end_date, trade_ids, sit_out, overrides, goalie_share, minutes_rule, 1, seed, True)
    B = res["bundle_used"]
    names = _names(B)
    g = res["games"]
    sched = B["schedule"].loc[g["rows"]]
    hg, ag, ot, so = g["home_goals"][0], g["away_goals"][0], g["ot"][0], g["so"][0]
    games = pd.DataFrame({"date": sched["date"].to_numpy(), "home": sched["home"].to_numpy(), "away": sched["away"].to_numpy(),
                          "home_goals": hg, "away_goals": ag, "ot": ot, "so": so,
                          "home_goalie": g["home_goalie"], "away_goalie": g["away_goalie"]})
    hw = games["home_goals"] > games["away_goals"]
    # Goals players scored = the score minus the shootout winner's +1
    games["home_gf"] = games["home_goals"] - (games["so"] & hw)
    games["away_gf"] = games["away_goals"] - (games["so"] & ~hw)

    def team_games(tm):
        d = games[(games["home"] == tm) | (games["away"] == tm)]
        home = d["home"] == tm
        won = np.where(home, d["home_goals"] > d["away_goals"], d["away_goals"] > d["home_goals"])
        return pd.DataFrame({"date": d["date"].to_numpy(), "opponent": np.where(home, d["away"], d["home"]),
                             "home_away": np.where(home, "home", "away"),
                             "goals_for": np.where(home, d["home_goals"], d["away_goals"]),
                             "goals_against": np.where(home, d["away_goals"], d["home_goals"]),
                             "gf": np.where(home, d["home_gf"], d["away_gf"]), "ga": np.where(home, d["away_gf"], d["home_gf"]),
                             "extra": np.where(d["so"], "SO", np.where(d["ot"], "OT", "")),
                             "result": np.where(won, "W", np.where(d["ot"], "OTL", "L")),
                             "goalie": pd.Series(np.where(home, d["home_goalie"], d["away_goalie"])).map(names).to_numpy()})

    log = team_games(team)
    w, otl = (log["result"] == "W").cumsum(), (log["result"] == "OTL").cumsum()
    log["record"] = [f"{a}-{b}-{c}" for a, b, c in zip(w, (log["result"] == "L").cumsum(), otl)]
    log["points"] = 2 * w + otl
    log["score"] = [f"{a}-{b}" + (f" {e}" if e else "") for a, b, e in zip(log["goals_for"], log["goals_against"], log["extra"])]

    # Every team's range: W-L-OTL, points, GF, GA (from the games), and the standings at the end of the range
    rec, real = _record(res)
    teams = res["teams"]
    per = []
    for tm in teams:
        d = team_games(tm)
        per.append({"team": tm, "gp": len(d), "wins": int((d["result"] == "W").sum()), "losses": int((d["result"] == "L").sum()),
                    "otl": int((d["result"] == "OTL").sum()), "gf": int(d["gf"].sum()), "ga": int(d["ga"].sum())})
    per = pd.DataFrame(per)
    per["points"] = 2 * per["wins"] + per["otl"]
    per["pace_84"] = np.where(per["gp"] > 0, per["points"] / per["gp"].where(per["gp"] > 0, 1) * SE.GAMES_PER_TEAM, np.nan)
    div = B["divisions"].set_index("team").loc[teams]
    st = pd.DataFrame({"team": teams, "division": div["division"].to_numpy(), "conference": div["conference"].to_numpy(),
                       "gp": (real["gp"].to_numpy() + per["gp"].to_numpy()).astype(int),
                       "wins": res["tallies"]["wins"][0].astype(int), "otl": res["tallies"]["otl"][0].astype(int),
                       "points": res["tallies"]["points"][0].astype(int), "rw": res["tallies"]["rw"][0].astype(int),
                       "row": res["tallies"]["row"][0].astype(int), "goal_diff": res["tallies"]["gd"][0].astype(int)})
    st["losses"] = st["gp"] - st["wins"] - st["otl"]
    if res["range"]["full_season"]:
        st["league_rank"] = res["table"]["avg_rank"].astype(int).to_numpy()
        st["seed"] = [SE.SEED_LABELS[int(x)] for x in res["seeds"][0]]
        st = st.sort_values("league_rank")
    else:
        st = st.sort_values(["points", "rw", "row", "team"], ascending=[False, False, False, True])
    standings = st.reset_index(drop=True)

    views = _two_views(res, True)
    me = per.set_index("team").loc[team]

    # Every pool player's random season (for notebook 31)
    lu = res["lineups"]
    z = pd.DataFrame({"playerId": res["z_ids"], "z": res["z"][0]})
    ss = B["stat_settings"]
    pool_id = B["players"].drop_duplicates("playerId").set_index("playerId")["pool_id"]
    z["pool_id"] = z["playerId"].map(pool_id).astype(int)
    norms = SC.multiplier_norms(B["pools"], ss["k"])
    z["scoring_multiplier"] = SC.scoring_multiplier(z["z"].to_numpy(), z["pool_id"].to_numpy(), norms, ss["k"])
    lrow = lu.assign(draw_size=res["draw_size"])[["playerId", "name", "team", "group", "dressed", "minutes_per_game",
                                                  "proj_war_60", "draw_size"]]
    draws = z.merge(lrow, on="playerId", how="left")
    draws["name"] = draws["name"].fillna(draws["playerId"].map(names))
    draws["dressed"] = draws["dressed"].fillna(False).astype(bool)
    draws.loc[~draws["dressed"], "draw_size"] = np.nan               # he didn't play, so no WAR change this run
    draws["war60_change"] = draws["z"] * draws["draw_size"]
    draws["season_war60"] = draws["proj_war_60"] + draws["war60_change"]
    draws["games_in_range"] = np.where(draws["dressed"], draws["team"].map(res["range"]["gp"]).fillna(0), 0).astype(int)
    upd = inputs["updated"]
    draws["draw_from"] = np.where(draws["playerId"].isin(upd["playerId"]) if upd is not None else False, "updated", "pool")
    draws = draws[["playerId", "name", "team", "group", "dressed", "games_in_range", "minutes_per_game", "z", "draw_from",
                   "draw_size", "war60_change", "proj_war_60", "season_war60", "scoring_multiplier"]]

    full = res["range"]["full_season"]
    season_label = "final season totals" if full else f"season to date (real before {res['start_date']} + the range)"
    rs, ss_ = views["range_skaters"], views["season_skaters"]
    return {"team": team, "seed": seed, "range": {k: v for k, v in res["range"].items() if k != "gp"},
            "game_log": log[["date", "opponent", "home_away", "score", "result", "goalie", "record", "points", "gf", "ga"]],
            "record": {"gp": int(me["gp"]), "wins": int(me["wins"]), "losses": int(me["losses"]), "otl": int(me["otl"]),
                       "points": int(me["points"]), "gf": int(me["gf"]), "ga": int(me["ga"]), "pace_84": float(me["pace_84"])},
            "teams_range": per, "standings": standings,
            "team_skaters": {"range": rs[rs["team"] == team].sort_values(["sim_p", "sim_g"], ascending=False),
                             "season": ss_[ss_["team"] == team].sort_values(["p", "g"], ascending=False)},
            "team_goalies": {"range": views["range_goalies"][views["range_goalies"]["team"] == team],
                             "season": views["season_goalies"][views["season_goalies"]["team"] == team]},
            "leaders": {"range": _leaders(rs, ["sim_g", "sim_a", "sim_p"]), "season": _leaders(ss_, ["g", "a", "p"])},
            "season_label": season_label, "draws": draws, "warnings": res["warnings"], "notes": NOT_SIMULATED,
            "ratings": res["ratings"], "minutes_rule": res["minutes_rule"], "run": res, **views}


def projection(inputs, team, n_games=None, end_date=None, trade_ids=None, sit_out=(), overrides=None, goalie_share=None,
               minutes_rule="moved", n_sims=10000, seed=0):
    """
    The same choices over n_sims seasons: the team's W, L, OTL and points over the range (average, 10th, 90th) and
    pace; for a full season, final points and playoff, division and Presidents' Trophy odds; every team's range;
    skater and goalie lines two ways (range only; season to date = real so far + range, ranges shifted by the
    real count).
    """
    res = play(inputs, team, n_games, end_date, trade_ids, sit_out, overrides, goalie_share, minutes_rule, n_sims, seed)
    rec, real = _record(res)
    teams = list(res["teams"])
    rows = []
    for i, tm in enumerate(teams):
        r = {"team": tm, "gp": int(rec["gp"][0, i])}
        for k in ["wins", "losses", "otl", "points"]:
            r[k] = rec[k][:, i].mean()
            r[f"{k}_p10"], r[f"{k}_p90"] = np.percentile(rec[k][:, i], [10, 90])
        rows.append(r)
    tr = pd.DataFrame(rows)
    tr["pace_84"] = np.where(tr["gp"] > 0, tr["points"] / tr["gp"].where(tr["gp"] > 0, 1) * SE.GAMES_PER_TEAM, np.nan)
    tr["points_to_date"] = real["points"].to_numpy() + tr["points"]
    full = res["range"]["full_season"]
    table = res["table"]
    if full:
        tr = tr.merge(table[["team", "points", "p10", "p90", "playoff_pct", "division_pct", "presidents_pct", "avg_rank"]]
                      .rename(columns={"points": "final_points", "p10": "final_p10", "p90": "final_p90"}), on="team")
    views = _two_views(res, False)
    rs, ss_ = views["range_skaters"], views["season_skaters"]
    me = tr.set_index("team").loc[team]
    return {"team": team, "n_sims": n_sims, "seed": seed, "range": {k: v for k, v in res["range"].items() if k != "gp"},
            "team_range": me, "teams_range": tr,
            "team_skaters": {"range": rs[rs["team"] == team].sort_values("sim_p", ascending=False),
                             "season": ss_[ss_["team"] == team].sort_values("p", ascending=False)},
            "team_goalies": {"range": views["range_goalies"][views["range_goalies"]["team"] == team],
                             "season": views["season_goalies"][views["season_goalies"]["team"] == team]},
            "leaders": {"range": _leaders(rs, ["sim_g", "sim_a", "sim_p"]), "season": _leaders(ss_, ["g", "a", "p"])},
            "warnings": res["warnings"], "notes": NOT_SIMULATED, "ratings": res["ratings"],
            "minutes_rule": res["minutes_rule"], "run": res, **views}


def paired_change(base, changed, team):
    """How much a change moved a team, from two projections with the same seed (so the same player luck): the
    average change in points over the range and in playoff % (full season), each with 2 standard errors of the
    paired season-by-season difference (the noise left in the comparison)."""
    i = list(base["run"]["teams"]).index(team)
    d = changed["run"]["tallies"]["points"][:, i] - base["run"]["tallies"]["points"][:, i]
    out = {"points": float(d.mean()), "points_2se": float(2 * d.std(ddof=1) / np.sqrt(len(d)))}
    if base["range"]["full_season"]:
        p = (changed["run"]["seeds"][:, i] > 0).astype(float) - (base["run"]["seeds"][:, i] > 0)
        out.update({"playoff_pct": float(100 * p.mean()), "playoff_pct_2se": float(200 * p.std(ddof=1) / np.sqrt(len(p)))})
    return out
