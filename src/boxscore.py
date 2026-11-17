"""
Per-game box scores for franchise mode (notebook 32): what the calendar and the box-score click need.

    import boxscore as BX
    s = FR.story(inp, "DET", seed=0)                        # notebook 30's one-season story
    sp = BX.split_story(s)                                  # every game's player lines and goal list
    BX.box_score(sp, 2026020123)                            # one game, for the click
    wp = BX.win_probabilities(inp, "DET")                   # pre-game home / away / OT chances, every game
    BX.calendar(sp["games"], "DET", "2026-12", probs=wp["games"])

What a box score can show: each skater's goals, assists, points and minutes; each starting goalie's goals against
and decision; who scored and who assisted each goal. What it can't: periods, goal times, shots, saves, +/-,
penalties. The engine doesn't simulate them, so they aren't invented here.

How the games get their player stats. The story already has each player's goals and assists for the whole range
(notebook 27's season-level split) and each game's score. Lineups are fixed, every dressed player plays every
game with the same minutes, so every goal his team scores has the same chance of being his, whichever game it
is in. Handing out goals one at a time with the same chances, then counting, gives each player a total; and once
the totals are known, every way of arranging those goals across the games is equally likely. So shuffling each
player's goals into the team's games gives exactly the same statistics as playing it goal by goal, and both
totals hold by construction: every game's player goals = the team's goals, every player's games = his total.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import season as SE
import simulate as S

NOT_SIMULATED = ("Box scores show goals, assists, points and minutes (skaters) and goals against and the decision "
                 "(starting goalies). No periods, goal times, shots, saves, +/- or penalties: the engine doesn't "
                 "simulate them. Minutes are each player's lineup minutes per game (the engine doesn't vary ice "
                 "time from game to game).")
RETRIES = 20                                       # random fills tried before the max-flow solver


# ---------------------------------------------------------------------------
# 1. The games of a story (simulated) and the real games before it
# ---------------------------------------------------------------------------
def story_games(story):
    """Every simulated game of a story: game_id, date, home, away, the score (as the standings count it), the
    goals players scored (gf: the score minus the shootout winner's +1), OT, SO, starting goalies, and each side's
    empty-net goals (scored into the other team's empty net)."""
    res = story["run"]
    g = res["games"]
    sched = res["bundle_used"]["schedule"].loc[g["rows"]]
    d = pd.DataFrame({"game_id": sched["game_id"].to_numpy(), "date": sched["date"].to_numpy(),
                      "home": sched["home"].to_numpy(), "away": sched["away"].to_numpy(),
                      "home_score": g["home_goals"][0], "away_score": g["away_goals"][0],
                      "ot": g["ot"][0], "so": g["so"][0], "home_goalie": g["home_goalie"], "away_goalie": g["away_goalie"],
                      "home_en": g["home_en"][0], "away_en": g["away_en"][0]})
    hw = d["home_score"] > d["away_score"]
    d["home_gf"] = d["home_score"] - (d["so"] & hw)
    d["away_gf"] = d["away_score"] - (d["so"] & ~hw)
    d["kind"] = "simulated"
    return d


def real_games_table(bundle, start):
    """The real games before `start` (finished, in the schedule): game_id, date, home, away, score, OT/SO."""
    s = bundle["schedule"]
    r = s[(s["date"] < start) & s["played"]]
    d = pd.DataFrame({"game_id": r["game_id"].to_numpy(), "date": r["date"].to_numpy(), "home": r["home"].to_numpy(),
                      "away": r["away"].to_numpy(), "home_score": r["home_goals"].astype(int).to_numpy(),
                      "away_score": r["away_goals"].astype(int).to_numpy(),
                      "ot": (r["last_period"] != "REG").to_numpy(), "so": (r["last_period"] == "SO").to_numpy()})
    d["kind"] = "real"
    return d


def season_games(bundle, start):
    """Every game of the season for projection mode: the real games before `start` with their results, then the
    games still to play (no score; kind "to play")."""
    s = bundle["schedule"]
    later = s[s["date"] >= start]
    d = pd.DataFrame({"game_id": later["game_id"].to_numpy(), "date": later["date"].to_numpy(),
                      "home": later["home"].to_numpy(), "away": later["away"].to_numpy(), "kind": "to play"})
    return pd.concat([real_games_table(bundle, start), d], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Assists: who assisted which goal
# ---------------------------------------------------------------------------
def _n2_given_total(goals, total, mix, rng):
    """How many goals get a second assist, given the team's goals and total assists, from notebook 27's model:
    n1 ~ Binomial(goals, p1 + p2) goals with 1+ assist, n2 ~ Binomial(n1, p2 / (p1 + p2)), total = n1 + n2."""
    from scipy.stats import binom
    q1 = mix["p1"] + mix["p2"]
    q2 = mix["p2"] / q1
    n1 = np.arange(int(np.ceil(total / 2)), min(total, goals) + 1)
    w = binom.pmf(n1, goals, q1) * binom.pmf(total - n1, n1, q2)
    n1 = rng.choice(n1, p=w / w.sum())
    return n1, total - n1


def _greedy(scorer, slots, assists, rng):
    """A random legal fill: players with the most assists first, each on distinct goals he didn't score that still
    have a free slot. Returns {goal: [assisters]} or None if it gets stuck."""
    free = slots.copy()
    out = {j: [] for j in range(len(scorer))}
    order = sorted(assists, key=lambda p: (-assists[p], rng.random()))
    for p in order:
        ok = np.flatnonzero((free > 0) & (scorer != p))
        if len(ok) < assists[p]:
            return None
        w = free[ok].astype(float)
        pick = rng.choice(ok, size=assists[p], replace=False, p=w / w.sum())
        for j in pick:
            out[j].append(p)
            free[j] -= 1
    return out


def _flow(scorer, assists, cap=2):
    """The max-flow solver: a legal arrangement with each goal taking up to `cap` assists (it decides how many
    goals get 2). Returns ({goal: [assisters]}, assists it couldn't place)."""
    players = list(assists)
    P, G = len(players), len(scorer)
    src, sink = 0, P + G + 1
    rows, cols, caps = [], [], []
    for i, p in enumerate(players):
        rows.append(src); cols.append(1 + i); caps.append(assists[p])
        for j in np.flatnonzero(scorer != p):
            rows.append(1 + i); cols.append(1 + P + j); caps.append(1)
    for j in range(G):
        rows.append(1 + P + j); cols.append(sink); caps.append(cap)
    m = csr_matrix((np.array(caps, dtype=np.int32), (rows, cols)), shape=(P + G + 2, P + G + 2))
    f = maximum_flow(m, src, sink).flow.tocoo()
    out = {j: [] for j in range(G)}
    placed = {p: 0 for p in players}
    for r, c, v in zip(f.row, f.col, f.data):
        if v > 0 and 1 <= r <= P and P < c <= P + G:
            out[c - 1 - P].append(players[r - 1])
            placed[players[r - 1]] += 1
    return out, {p: assists[p] - placed[p] for p in players if assists[p] > placed[p]}


def assign_assists(scorer, assists, mix, rng, weights=None):
    """
    Who assisted each goal of one team: scorer = the scorer of every goal (array), assists = {player: season
    assists}. Each goal gets 0, 1 or 2 assists (how many, from notebook 27's model given the total), never the
    scorer, never the same player twice on one goal, and every player's total stays exact.
    Returns (assisters per goal, how: "random", "flow" or "impossible", leftovers moved {from: to}).
    If no legal arrangement exists, each assist that can't be placed goes to the eligible teammate with the highest
    weight (weights = {player: assist weight}; default his assist total), on a goal with a free slot.
    """
    assists = {p: int(a) for p, a in assists.items() if a > 0}
    total = sum(assists.values())
    G = len(scorer)
    if total == 0:
        return {j: [] for j in range(G)}, "random", []
    for _ in range(RETRIES):
        n1, n2 = _n2_given_total(G, total, mix, rng)
        slots = np.zeros(G, int)
        perm = rng.permutation(G)
        slots[perm[:n2]] = 2
        slots[perm[n2:n1]] = 1
        out = _greedy(np.asarray(scorer), slots, assists, rng)
        if out is not None:
            return out, "random", []
    out, left = _flow(np.asarray(scorer), assists)
    if not left:
        return out, "flow", []
    moved = []
    w = weights or assists
    for p, k in left.items():
        for _ in range(k):
            best = None
            for j in range(G):
                if len(out[j]) >= 2:
                    continue
                for q in sorted(w, key=lambda x: -w[x]):
                    if q != p and q != scorer[j] and q not in out[j]:
                        if best is None or w[q] > w[best[1]]:
                            best = (j, q)
                        break
            if best is None:
                continue
            out[best[0]].append(best[1])
            moved.append((p, best[1]))
    return out, "impossible", moved


# ---------------------------------------------------------------------------
# 3. The split: every game's player lines and goal list
# ---------------------------------------------------------------------------
def split_story(story, seed=0):
    """
    Per-game player stats for a one-season story (notebook 30's FR.story).
    Returns a dict:
      games: every game of the season (real before the start, then simulated), with scores;
      player_games: game_id, date, team, opponent, home_away, kind, playerId, name, position, g, a, p, minutes,
        and for starting goalies ga and decision (only dressed players, only games they played);
      goals: game_id, team, goal_no, scorer, assist1, assist2 (simulated games only);
      assists_how: per team, "random", "flow" or "impossible"; moved: leftover assists given to a teammate.
    """
    res = story["run"]
    B = res["bundle_used"]
    rng = np.random.default_rng([seed, 32])
    sim = story_games(story)
    lu = res["lineups"]
    lud = lu[lu["dressed"]].set_index("playerId")
    sk = story["range_skaters"].set_index("playerId")
    mix = B["stat_settings"]["mix"]
    names = lu.drop_duplicates("playerId").set_index("playerId")["name"]

    goal_rows, line_rows, how, moved_all = [], [], {}, []
    for team in sorted(set(sim["home"]) | set(sim["away"])):
        tg = sim[(sim["home"] == team) | (sim["away"] == team)].sort_values(["date", "game_id"])
        home = (tg["home"] == team).to_numpy()
        gf = np.where(home, tg["home_gf"], tg["away_gf"]).astype(int)
        mine = sk[sk["team"] == team]
        g_tot = mine["sim_g"].astype(int)
        tokens = np.repeat(g_tot.index.to_numpy(), g_tot.to_numpy())
        if len(tokens) != gf.sum():
            raise ValueError(f"{team}: player goals {len(tokens)} != team goals {gf.sum()}")
        scorer = rng.permutation(tokens)
        game_of = np.repeat(np.arange(len(tg)), gf)
        a_tot = mine["sim_a"].astype(int)
        ast, h, mv = assign_assists(scorer, a_tot.to_dict(), mix, rng)
        how[team] = h
        moved_all += [(team, a, b) for a, b in mv]
        gid = tg["game_id"].to_numpy()
        goal_no = np.concatenate([np.arange(1, k + 1) for k in gf]) if len(gf) else np.array([], int)
        for j in range(len(scorer)):
            a = ast[j] + [np.nan, np.nan]
            goal_rows.append((gid[game_of[j]], team, goal_no[j], scorer[j], a[0], a[1]))
        # Skater lines: every dressed skater in every game of the range (games x skaters count tables)
        skaters = lud[(lud["team"] == team) & (lud["group"] != "G")]
        sid = skaters.index.to_numpy()
        col = {p: i for i, p in enumerate(sid)}
        G = np.zeros((len(tg), len(sid)), int)
        A = np.zeros((len(tg), len(sid)), int)
        np.add.at(G, (game_of, [col[p] for p in scorer]), 1)
        for j in range(len(scorer)):
            for p in ast[j]:
                A[game_of[j], col[p]] += 1
        opp = np.where(home, tg["away"], tg["home"])
        n_g, n_s = len(tg), len(sid)
        line_rows.append(pd.DataFrame({
            "game_id": np.repeat(gid, n_s), "date": np.repeat(tg["date"].to_numpy(), n_s), "team": team,
            "opponent": np.repeat(opp, n_s), "home_away": np.repeat(np.where(home, "home", "away"), n_s),
            "kind": "simulated", "playerId": np.tile(sid, n_g), "name": np.tile(skaters["name"].to_numpy(), n_g),
            "position": np.tile(skaters["position"].to_numpy(), n_g), "g": G.ravel(), "a": A.ravel(),
            "p": (G + A).ravel(), "minutes": np.tile(skaters["minutes_per_game"].to_numpy(), n_g)}))
        # Goalies: the starter of each game, his goals against (no shootout or empty-net goals) and decision
        my = np.where(home, tg["home_score"], tg["away_score"])
        their = np.where(home, tg["away_score"], tg["home_score"])
        their_so = tg["so"].to_numpy() & (their > my)
        their_en = np.where(home, tg["away_en"], tg["home_en"])
        gk = np.where(home, tg["home_goalie"], tg["away_goalie"])
        line_rows.append(pd.DataFrame({
            "game_id": gid, "date": tg["date"].to_numpy(), "team": team, "opponent": opp,
            "home_away": np.where(home, "home", "away"), "kind": "simulated", "playerId": gk,
            "name": pd.Series(gk).map(names).to_numpy(), "position": "G", "g": 0, "a": 0, "p": 0, "minutes": 60.0,
            "ga": (their - their_so - their_en).astype(int),
            "decision": np.where(my > their, "W", np.where(tg["ot"].to_numpy(), "OTL", "L"))}))
    goals = pd.DataFrame(goal_rows, columns=["game_id", "team", "goal_no", "scorer", "assist1", "assist2"])
    for c in ["scorer", "assist1", "assist2"]:
        goals[f"{c}_name"] = goals[c].map(names)
    pg = pd.concat(line_rows, ignore_index=True)

    # Real games before the start: real player lines (no goal list: the boxscores don't pair scorers and assists)
    real = real_games_table(B, res["start_date"])
    rg = B.get("real_games")
    if rg is not None and len(real):
        r = rg[(rg["date"] < res["start_date"]) & (rg["toi"] > 0)].merge(real[["game_id", "home", "away"]], on="game_id")
        home = r["team"] == r["home"]
        pr = pd.DataFrame({"game_id": r["game_id"], "date": r["date"], "team": r["team"],
                           "opponent": np.where(home, r["away"], r["home"]), "home_away": np.where(home, "home", "away"),
                           "kind": "real", "playerId": r["playerId"], "name": r["name"], "position": r["position"],
                           "g": r["goals"].fillna(0).astype(int), "a": r["assists"].fillna(0).astype(int),
                           "minutes": r["toi"], "ga": r["goals_against"],
                           "decision": r["decision"].map({"W": "W", "L": "L", "O": "OTL"})})
        pr["p"] = pr["g"] + pr["a"]
        pg = pd.concat([pr, pg], ignore_index=True)
    games = pd.concat([real, sim.drop(columns=["home_gf", "away_gf"])], ignore_index=True).sort_values(["date", "game_id"])
    return {"games": games.reset_index(drop=True), "player_games": pg, "goals": goals, "assists_how": how,
            "moved": moved_all, "note": NOT_SIMULATED, "start": res["start_date"]}


def box_score(split, game_id):
    """One game, for the box-score click: the score line, both teams' skaters and goalies, and (simulated games)
    the goal list in order (no times: the engine has none)."""
    g = split["games"].set_index("game_id").loc[game_id]
    pg = split["player_games"][split["player_games"]["game_id"] == game_id]
    sk = pg[pg["position"] != "G"].sort_values(["team", "p", "g", "minutes"], ascending=[True, False, False, False])
    gk = pg[pg["position"] == "G"]
    extra = " (SO)" if g["so"] else (" (OT)" if g["ot"] else "")
    return {"game_id": game_id, "date": g["date"], "kind": g["kind"],
            "score": f"{g['away']} {int(g['away_score'])} at {g['home']} {int(g['home_score'])}{extra}",
            "skaters": sk[["team", "name", "position", "g", "a", "p", "minutes"]].reset_index(drop=True),
            "goalies": gk[["team", "name", "ga", "decision", "minutes"]].reset_index(drop=True),
            "goals": split["goals"][split["goals"]["game_id"] == game_id][
                ["team", "goal_no", "scorer_name", "assist1_name", "assist2_name"]].reset_index(drop=True),
            "note": split["note"]}


# ---------------------------------------------------------------------------
# 4. Pre-game win probabilities
# ---------------------------------------------------------------------------
GRID_STEP = 0.0005                                  # edge grid (normal goals per game) for the fast lookup
GRID_MAX = 8.0                                      # grid from -8 to +8 normal goals per game (99% of edges are within 2.5)
PROB_KEYS = ["home_win", "away_win", "ot", "home_reg", "away_reg"]


def prob_grid(settings):
    """The exact chances on a fine grid of edges, computed once; lookup() interpolates between grid points."""
    grid = np.arange(-GRID_MAX, GRID_MAX + GRID_STEP / 2, GRID_STEP)
    return grid, S.game_probabilities(grid, settings)


def lookup(edge, grid, probs, settings=None):
    """Straight-line interpolation on the grid (one index for all five chances). Edges beyond the grid (none in
    practice) are computed exactly with settings."""
    edge = np.asarray(edge, dtype=float)
    out_of_grid = np.abs(edge) > GRID_MAX - GRID_STEP
    x = (np.clip(edge, -GRID_MAX + GRID_STEP, GRID_MAX - GRID_STEP) - grid[0]) / GRID_STEP
    i = np.floor(x).astype(int)
    f = x - i
    out = {k: probs[k][i] * (1 - f) + probs[k][i + 1] * f for k in PROB_KEYS}
    if out_of_grid.any():
        if settings is None:
            raise ValueError("an edge is outside the probability grid; pass settings to compute it exactly")
        ex = S.game_probabilities(edge[out_of_grid], settings)
        for k in PROB_KEYS:
            out[k][out_of_grid] = ex[k]
    return out


def game_edges(res, draws_on=True):
    """
    The strength edge of every simulated game, for every simulated season (n_sims x games), exactly as
    run_scenario builds it: skater and goalie WAR per game from the lineups (+ each season's draws), the game's
    starting goalies, the team bonus. draws_on=False: every draw 0 (the edge "at projection", one row).
    """
    B = res["bundle_used"]
    lu = res["lineups"]
    sched = B["schedule"]
    sim = (sched["date"] >= res["start_date"]).to_numpy()
    games = sched[sim]
    starts = B["goalie_starts"][sim]
    teams = res["teams"]
    T = len(teams)
    home = np.searchsorted(teams, games["home"].to_numpy())
    away = np.searchsorted(teams, games["away"].to_numpy())
    gh = home + T * starts["home_backup"].to_numpy()
    ga = away + T * starts["away_backup"].to_numpy()
    bonus_map = res["bonus_used"]
    bonus = np.array([bonus_map.get(t, 0.0) for t in teams])
    goalies = lu[lu["dressed"] & (lu["group"] == "G")]
    g_row = {(t, r): i for i, t, r in zip(goalies.index, goalies["team"], goalies["rank"])}
    starter_rows = np.array([g_row[(t, 1)] for t in teams])
    backup_rows = np.array([g_row.get((t, 2), g_row[(t, 1)]) for t in teams])
    if draws_on:
        pos = np.searchsorted(res["z_ids"], lu["playerId"].to_numpy())
        draws = (res["z"][:, pos] * np.asarray(res["draw_size"])[None, :]).T          # lineup rows x n_sims
    else:
        draws = np.zeros((len(lu), 1))
    _, sk, _ = S.team_strength(lu, draws)
    war60 = lu["proj_war_60"].to_numpy()[:, None] + draws
    gk = np.concatenate([war60[starter_rows].T, war60[backup_rows].T], axis=1)
    s = B["game"]
    edge = s["a_s"] * (sk[:, home] - sk[:, away]) + s["a_g"] * (gk[:, gh] - gk[:, ga]) + (bonus[home] - bonus[away])[None, :]
    return games, edge


def win_probabilities(inputs, team, n_sims=10000, seed=0, sit_out=(), trade_ids=None, overrides=None,
                      goalie_share=None, minutes_rule="moved", n_games=None, end_date=None):
    """
    Pre-game chances for every game in the range, from the game model the simulator uses, with the lineups in
    effect (franchise choices, each game's starting goalies, the team bonus, the in-season update if it's on):
      home_win, away_win, ot (goes past regulation), home_reg, away_reg: averaged over the players' random seasons
        (the same n_sims draws as the projection with this seed), since player seasons vary;
      home_win_at_proj, ot_at_proj: the same with every draw at 0 (each player exactly at his projection).
    Also returns the run (10,000 seasons) for the agreement check.
    """
    import franchise as FR
    res = FR.play(inputs, team, n_games, end_date, trade_ids, sit_out, overrides, goalie_share, minutes_rule, n_sims,
                  seed, story=False, keep_z=True)
    B = res["bundle_used"]
    res["bonus_used"] = (inputs["updated"]["bonus"] if inputs["updated"] is not None and "bonus" in inputs["updated"]
                         else B["bonus"])
    games, e0 = game_edges(res, draws_on=False)
    sums = {k: np.zeros(len(games)) for k in PROB_KEYS}
    grid, pg = prob_grid(B["game"])
    teams = res["teams"]
    H = (games["home"].to_numpy()[:, None] == teams[None, :]).astype(float)     # games x teams
    A = (games["away"].to_numpy()[:, None] == teams[None, :]).astype(float)
    per_sim = {"wins": np.zeros((n_sims, len(teams))), "ot": np.zeros((n_sims, len(teams)))}
    for c0 in range(0, n_sims, 1000):
        r = {**res, "z": res["z"][c0:c0 + 1000]}
        _, e = game_edges(r)
        v = lookup(e, grid, pg, B["game"])
        for k in PROB_KEYS:
            sums[k] += v[k].sum(0)
        sl = slice(c0, c0 + e.shape[0])
        per_sim["wins"][sl] = v["home_win"] @ H + v["away_win"] @ A                # each season's expected wins
        per_sim["ot"][sl] = v["ot"] @ (H + A)
    out = pd.DataFrame({"game_id": games["game_id"].to_numpy(), "date": games["date"].to_numpy(),
                        "home": games["home"].to_numpy(), "away": games["away"].to_numpy(),
                        **{k: v / n_sims for k, v in sums.items()}})
    p0 = S.game_probabilities(e0[0], B["game"])
    out["home_win_at_proj"] = p0["home_win"]
    out["ot_at_proj"] = p0["ot"]
    return {"games": out, "run": res, "per_sim": per_sim}


def check_against_runs(wp):
    """Each team's expected wins and OT games over the range (sum of the chances) vs. the average over the same
    10,000 simulated seasons. The standard error pairs each simulated season with its own expected wins (same
    player draws), so it measures only the game-to-game luck the chances leave out."""
    import franchise as FR
    res = wp["run"]
    rec, _ = FR._record(res)
    g = wp["games"]
    rows = []
    for i, t in enumerate(res["teams"]):
        h, a = g["home"] == t, g["away"] == t
        exp_w = g.loc[h, "home_win"].sum() + g.loc[a, "away_win"].sum()
        exp_ot = g.loc[h | a, "ot"].sum()
        w = rec["wins"][:, i]
        ot = (rec["wins"][:, i] - rec["rw"][:, i]) + rec["otl"][:, i]
        n = len(w)
        dw = w - wp["per_sim"]["wins"][:, i]
        dot = ot - wp["per_sim"]["ot"][:, i]
        rows.append({"team": t, "games": int(h.sum() + a.sum()), "expected_wins": exp_w, "sim_wins": w.mean(),
                     "se_wins": dw.std() / np.sqrt(n), "expected_ot": exp_ot, "sim_ot": ot.mean(),
                     "se_ot": dot.std() / np.sqrt(n)})
    t = pd.DataFrame(rows)
    t["wins_gap_in_se"] = (t["expected_wins"] - t["sim_wins"]) / t["se_wins"]
    t["ot_gap_in_se"] = (t["expected_ot"] - t["sim_ot"]) / t["se_ot"]
    return t


# ---------------------------------------------------------------------------
# 5. The calendar
# ---------------------------------------------------------------------------
def calendar(games, team, month, probs=None):
    """
    One row per game of `team` in `month` ("2026-12"): date, opponent, home/away, game_id, kind.
    games: split_story's games (story mode: result and record after it, counting the real games before the start)
    or the schedule's games for projection mode; probs: win_probabilities' table (projection mode: win %, OT %,
    loss %; real games keep their real result).
    """
    g = games[(games["home"] == team) | (games["away"] == team)].sort_values(["date", "game_id"]).copy()
    home = g["home"] == team
    g["opponent"] = np.where(home, g["away"], g["home"])
    g["home_away"] = np.where(home, "home", "away")
    has = g["home_score"].notna() if "home_score" in g else pd.Series(False, index=g.index)
    for c in ["ot", "so"]:
        g[c] = g[c].fillna(False).astype(bool) if c in g else False
    if has.any():
        my = np.where(home, g["home_score"], g["away_score"])
        their = np.where(home, g["away_score"], g["home_score"])
        won = my > their
        res = np.where(won, "W", np.where(g["ot"], "OTL", "L"))
        res = np.where(has, res, "")
        extra = np.where(g["so"], " (SO)", np.where(g["ot"], " (OT)", ""))
        g["result"] = [f"{r} {int(a)}-{int(b)}{e}" if r else "" for r, a, b, e in zip(res, np.nan_to_num(my), np.nan_to_num(their), extra)]
        w, otl, l = np.cumsum(res == "W"), np.cumsum(res == "OTL"), np.cumsum(res == "L")
        g["record"] = [f"{a}-{b}-{c}" if r else "" for a, b, c, r in zip(w, l, otl, res)]
    if probs is not None:
        p = probs.set_index("game_id")
        hw = g["game_id"].map(p["home_win"])
        aw = g["game_id"].map(p["away_win"])
        g["win_pct"] = np.where(home, hw, aw) * 100            # blank for real games (already played)
        g["ot_pct"] = g["game_id"].map(p["ot"]) * 100
        g["loss_pct"] = 100 - g["win_pct"]
    g = g[g["date"].str.startswith(month)]
    cols = ["date", "opponent", "home_away", "kind", "game_id"] + [c for c in ["result", "record", "win_pct", "ot_pct",
                                                                               "loss_pct"] if c in g]
    return g[cols].reset_index(drop=True)
