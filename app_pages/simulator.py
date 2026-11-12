"""
The Simulator tab of the Forecast page (app_pages/forecast.py calls show_simulator()). Not a page of its own any
more: app.py lists only Almanac and Forecast.

It plays the rest of the 2026-27 season with the trades and sit-outs you pick, either once (a full season with
standings, leaders and box-score-style stat lines) or 10,000 times (playoff odds and ranges). The simulator
itself is src/season.py.

It reads the bundle notebooks 26 and 27 save (data/simulator_2026-27/) and the hand-kept trades file
(data/trades_2026-27.csv). data/ is gitignored, so the live site doesn't have them: then this tab shows a short
message, and the rest of the app keeps working. Nothing here recomputes WAR or calls the NHL.

"Use 2026-27 results so far": notebook 29's last cell saves an update cache per weekly entry
(data/simulator_2026-27/updated/<evidence date>/: real results through that date, every player's updated
rest-of-season draws and the updated team bonus). The newest one is offered as a switch, for start dates on or
after its start date. Without a cache the switch is hidden.

To test the missing-files message, point it somewhere empty:
    HOCKEY_SIMULATOR_DIR=/nonexistent /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent            # the project folder (this file is in app_pages/)
BUNDLE_DIR = Path(os.environ.get("HOCKEY_SIMULATOR_DIR", ROOT / "data" / "simulator_2026-27"))
UPDATED_DIR = BUNDLE_DIR / "updated"                      # notebook 29's update caches, one folder per entry
TRADES_PATH = ROOT / "data" / "trades_2026-27.csv"
NEEDED = [BUNDLE_DIR / f for f in ["players.csv", "schedule.csv", "settings.json", "scoring.csv", "scoring.json",
                                   "real_games.csv"]] + [TRADES_PATH]
OPENING_NIGHT = date(2026, 9, 29)
DEFAULT_TEAM = "DET"
MODES = ["Play one season", "Run 10,000"]


# ---------------------------------------------------------------------------
# Loading and running (cached, because Streamlit reruns the page on every click)
# ---------------------------------------------------------------------------
def _season():
    """The simulator module, imported only when the files are there (the live site never gets this far, so it
    never needs the simulator's extra packages, like scipy)."""
    if str(ROOT / "src") not in sys.path:
        sys.path.append(str(ROOT / "src"))
    import season
    return season


def file_stamp():
    """The files' modification times: part of every cache key, so new notebook runs are picked up."""
    return tuple(p.stat().st_mtime for p in NEEDED)


@st.cache_resource
def load(stamp):
    """Read the bundle once (stamp isn't used inside; it only makes rewritten files load again)."""
    return _season().load_bundle(BUNDLE_DIR, TRADES_PATH)


def _refresh():
    """refresh.py (notebook 29's functions), imported only when an update cache is there."""
    _season()                                             # puts src/ on the path
    import refresh
    return refresh


def latest_cache():
    """The newest complete update cache folder, or None."""
    if not UPDATED_DIR.exists():
        return None
    return _refresh().newest_cache(UPDATED_DIR)


@st.cache_resource
def load_update(stamp, cache_dir, cache_stamp):
    """Read one update cache once: (the bundle with the cache's real results, updated draws and bonus, meta).
    Returns an error message instead if the cache doesn't fit the bundle (notebook 26 or 27 was rerun since)."""
    try:
        return _refresh().load_update_cache(cache_dir, load(stamp), BUNDLE_DIR)
    except ValueError as e:
        return str(e)


@st.cache_data(show_spinner=False, max_entries=30)
def run(stamp, start_date, trade_ids, sit_out, n_sims, seed, stat_lines=True, cache=None, use_update=False):
    """One scenario, cached by its inputs: the same inputs come back instantly the second time.
      cache: None = the bundle's real results (notebooks 26-27), or (folder, stamp) of an update cache whose
        real results to use; use_update: also its updated player draws and team bonus.
    Returns the team table, the season lines (real so far + simulated), seeds and the lineups."""
    SE = _season()
    if cache is None:
        bundle, updated = load(stamp), None
    else:
        bundle, upd, _ = load_update(stamp, *cache)
        updated = upd if use_update else None
    res = SE.run_scenario(bundle, start_date=start_date, trade_ids=list(trade_ids), sit_out=list(sit_out),
                          n_sims=n_sims, seed=seed, stat_lines=stat_lines, updated=updated)
    out = {k: res[k] for k in ["table", "lineups", "teams", "start_date", "trades", "sit_out", "n_sims", "seed",
                               "games_simulated", "games_locked", "seconds"]}
    out["seeds"] = res["seeds"][0] if n_sims == 1 else None
    if stat_lines:
        out["skaters"], out["goalies"] = SE.season_lines(res, bundle)
    return out


def trade_label(t):
    """'T1, Oct 23: Hellebuyck to NJD (Mercer, Daws, Bjugstad, Command to WPG)' from one trade's rows."""
    d = pd.Timestamp(t["date"].iloc[0])
    last = lambda names: ", ".join(n.split(" ", 1)[-1] for n in names)
    first_to = t["to_team"].iloc[0]
    a, b = t[t["to_team"] == first_to], t[t["to_team"] != first_to]
    text = f"{t['trade_id'].iloc[0]}, {d:%b} {d.day}: {last(a['name'])} to {first_to}"
    if len(b):
        text += f" ({last(b['name'])} to {b['to_team'].iloc[0]})"
    return text


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------
def show_simulator():
    if not all(p.exists() for p in NEEDED):
        st.info("Simulator runs locally for now. It needs the season inputs built by notebooks 26 and 27, which "
                "aren't on the live site yet. The other tabs and the Almanac work as usual.")
        return
    SE = _season()
    stamp = file_stamp()
    B = load(stamp)
    players, trades = B["players"], B["trades"]
    last_day = date.fromisoformat(B["schedule_downloaded"][:10])

    # notebook 29's newest update cache, if there is one that fits this bundle
    cache, cache_meta, cache_note = None, None, "Run notebook 29 to use 2026-27 results so far."
    cache_dir = latest_cache()
    if cache_dir is not None:
        cache = (str(cache_dir), (cache_dir / "draws.npz").stat().st_mtime)
        loaded = load_update(stamp, *cache)
        if isinstance(loaded, str):
            cache, cache_note = None, f"The update cache {cache_dir.name} can't be used ({loaded})."
        else:
            cache_meta = loaded[2]
    if cache_meta:
        through = date.fromisoformat(cache_meta["evidence_through"])
        latest_start = date.fromisoformat(cache_meta["start"])

    caption = (f"The real 2026-27 schedule, with real results and real stats from every game before {last_day} "
               "(when notebooks 26-27 last downloaded them).")
    if cache_meta:
        caption += f" With \"Use 2026-27 results so far\": every game through {through} (notebook 29's latest weekly entry)."
    st.caption(caption)
    with st.expander("How it works"):
        st.markdown(
            """
- **Simulated:** every game on or after the start date. Each score is random, tilted by the two lineups'
  projected WAR (skaters and that night's goalie), home ice and each team's bonus (about 40% of how much it beat
  or missed its players' projection in 2025-26). **Locked in:** every real game before the start date.
- **Lineups are fixed at the start date,** after the trades and sit-outs you pick, at **full health**.
  The backup goalie starts the second night of back-to-backs, plus enough games to make about 30 of 84.
- **Stat lines:** a team's simulated goals go to its dressed skaters by goals per 60 x minutes; each goal gets
  0, 1 or 2 assists at the real mix, by assists per 60 x minutes, never to the scorer. Each player's rates
  move with his random season (a breakout in WAR is a breakout in scoring). Goalies get the decision and goals
  against of the games they start. After opening night, lines are **real stats so far + the simulated rest**.
  No save % yet (the simulator has no shots).
- **Results are a range, not a prediction.** One season is one possible season; 10,000 show the range.
- **"Use 2026-27 results so far"** (after notebook 29): plays from the newest weekly entry, with real results
  through its date, each player's updated rest-of-season draws (notebook 28's update from his goals and assists
  so far) and the updated team bonuses. Only for a start date on or after that entry's start: from an earlier
  date, the update would already know games being simulated. Sit-outs and trades change who dresses, not a
  player's draws (in the adopted version his update doesn't depend on his minutes). The team bonuses stay the
  entry's, because they measure the real results so far. **Off** on that date: the same real results with
  preseason ratings, so the two show what the update changes.
"""
        )

    # ---- Controls ----
    max_day = max(last_day, latest_start) if cache_meta else last_day
    if "sim_start" not in st.session_state or st.session_state["sim_start"] > max_day:
        st.session_state["sim_start"] = OPENING_NIGHT

    def from_latest():
        st.session_state["sim_start"] = latest_start
        st.session_state["sim_go"] = True              # run right away

    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", min_value=OPENING_NIGHT, max_value=max_day, key="sim_start",
                          help="Real results before this date are locked in; the rest of the season is simulated.")
    mode = c2.radio("Mode", MODES, horizontal=True, key="sim_mode")
    start_str = start.isoformat()
    use_update = False
    if cache_meta:
        u1, u2 = st.columns([3, 1])
        label = f"Use 2026-27 results so far (updated through {through:%b} {through.day})"
        if start >= latest_start:
            use_update = u1.toggle(label, value=True, key=f"sim_use_{start_str}")      # on by default, per date
        else:
            u1.toggle(label, value=False, disabled=True, key="sim_use_disabled")
            u1.caption(f"Off for start dates before {latest_start:%b} {latest_start.day}: "
                       "the update has already seen games this run would simulate.")
        u2.button("Start from latest results", on_click=from_latest, key="sim_latest")
    else:
        st.caption(cache_note)
    # Real results: the bundle's up to its download day (exactly as before); after it, only the cache has them
    run_cache = cache if cache_meta and (use_update or start > last_day) else None
    labels = {tid: trade_label(t) for tid, t in trades.groupby("trade_id", sort=True)}
    chosen = st.multiselect("Trades in effect", list(labels), default=SE.default_trades(trades, start_str),
                            format_func=labels.get, key=f"sim_trades_{start_str}",   # resets when the date changes
                            help="Default: every trade made on or before the start date. Add or remove any.")
    pool = players[players["in_pool"]].sort_values(["team", "name"])
    who = {p: f"{n} ({t}, {pos})" for p, n, t, pos in zip(pool["playerId"], pool["name"], pool["team"], pool["position"])}
    sit = st.multiselect("Players sitting out (type to search)", list(who), format_func=who.get, key="sim_sit",
                         help="They leave their team's lineup for every simulated game; the next player up dresses.")
    if "sim_seed" not in st.session_state:
        st.session_state["sim_seed"] = 0

    def again():
        st.session_state["sim_seed"] += 1           # a new seed is a new season
        st.session_state["sim_go"] = True

    c3, c4, c5 = st.columns([2, 1, 1])
    c3.number_input("Seed", step=1, key="sim_seed", help="The same inputs and seed always give the same results.")
    go = c4.button("Run", type="primary", key="sim_run")
    if mode == MODES[0]:
        c5.button("Simulate again", on_click=again, key="sim_again")
    n_sims = 1 if mode == MODES[0] else 10000
    inputs = (start_str, tuple(sorted(chosen)), tuple(sorted(sit)), n_sims, int(st.session_state["sim_seed"]),
              run_cache, use_update)
    if go or st.session_state.pop("sim_go", False) or "sim_inputs" not in st.session_state \
            or st.session_state["sim_inputs"][3] != n_sims:
        st.session_state["sim_inputs"] = inputs     # first visit or a new mode: run right away

    shown = st.session_state["sim_inputs"]
    with st.spinner("Simulating..." if shown[3] == 1 else "Simulating 10,000 seasons (about 10 seconds the first time)..."):
        res = run(stamp, *shown[:5], cache=shown[5], use_update=shown[6])
    if shown != inputs:
        st.warning("The controls have changed since this run. Press **Run** to update the results.")
    names = players.set_index("playerId")["name"]
    real_part = res["games_locked"] > 0
    if shown[5] is None:
        ratings = "preseason"
    else:
        entry = Path(shown[5][0]).name                     # the cache folder = its evidence date
        ratings = (f"updated through {entry} (notebook 29)" if shown[6]
                   else f"preseason (real results through {entry} from notebook 29)")
    st.markdown(
        f"**Showing:** from {res['start_date']} ({res['games_locked']} real games locked in, {res['games_simulated']:,} "
        f"simulated), {'one season' if res['n_sims'] == 1 else f'{res['n_sims']:,} seasons'}, seed {res['seed']}. "
        f"Trades: {', '.join(res['trades']) or 'none'}. Sitting out: {', '.join(names[p] for p in res['sit_out']) or 'nobody'}. "
        f"Ratings: {ratings}.")
    if res["n_sims"] == 1:
        show_one_season(res, real_part)
    else:
        show_many(res, stamp, real_part)


def team_picker(res, key):
    teams = list(res["teams"])
    return st.selectbox("Team", teams, index=teams.index(DEFAULT_TEAM), key=key)


def show_one_season(res, real_part):
    """One simulated season: standings with seeds, league leaders, a team's box score."""
    SE = _season()
    t = res["table"].copy()
    t["seed"] = [SE.SEED_LABELS[int(x)] for x in res["seeds"]]
    st.subheader("Final standings")
    st.caption("Seed: 1-3 in the division, or the conference's first and second wild card (WC1, WC2). "
               "Ties: regulation wins, then regulation + overtime wins, then a coin flip.")
    cols = {"seed": st.column_config.TextColumn("Seed"), "team": st.column_config.TextColumn("Team"),
            "wins": st.column_config.NumberColumn("W", format="%d"), "losses": st.column_config.NumberColumn("L", format="%d"),
            "otl": st.column_config.NumberColumn("OTL", format="%d"), "points": st.column_config.NumberColumn("PTS", format="%d"),
            "goal_diff": st.column_config.NumberColumn("GD", format="%+d")}
    divs = sorted(t["division"].unique())
    for tab, div in zip(st.tabs(divs), divs):
        with tab:
            d = t[t["division"] == div].sort_values("avg_rank")       # league rank, tiebreakers included
            st.dataframe(d[list(cols)], column_config=cols, hide_index=True, width="stretch")

    sk, gk = res["skaters"], res["goalies"]
    st.subheader("League leaders")
    who = lambda d: d["name"] + " (" + d["team"] + ")"          # "Connor McDavid (EDM)"
    boards = [(sk.sort_values(["p", "g"], ascending=False), "p", "Points"),
              (sk.sort_values(["g", "p"], ascending=False), "g", "Goals"),
              (sk.sort_values(["a", "p"], ascending=False), "a", "Assists"),
              (gk.sort_values(["w", "gaa"], ascending=[False, True]), "w", "Goalie wins")]
    for i in range(0, 4, 2):                                    # two boards per row, so the numbers fit
        for col, (d, stat, title) in zip(st.columns(2), boards[i:i + 2]):
            top = d.head(10)
            col.markdown(f"**{title}**")
            col.dataframe(pd.DataFrame({"Player": who(top).to_numpy(), title.split()[-1][0]: top[stat].astype(int).to_numpy()}),
                          hide_index=True, width="stretch")

    st.subheader("Team stats")
    team = team_picker(res, "sim_team_one")
    if real_part:
        st.caption(f"Season (GP, G, A, P) = real games before {res['start_date']} (Real) + the simulated rest (Sim).")
    s = sk[sk["team"] == team].sort_values(["p", "g"], ascending=False)
    scols = ["name", "position", "gp", "g", "a", "p"]
    heads = {"name": "Player", "position": "Pos", "gp": "GP", "g": "G", "a": "A", "p": "P"}
    if real_part:
        scols = scols + ["real_gp", "real_g", "real_a", "real_p", "sim_gp", "sim_g", "sim_a", "sim_p"]   # season first
        heads.update({"real_gp": "Real GP", "real_g": "Real G", "real_a": "Real A", "real_p": "Real P",
                      "sim_gp": "Sim GP", "sim_g": "Sim G", "sim_a": "Sim A", "sim_p": "Sim P"})
    st.dataframe(s[scols].rename(columns=heads).astype({h: int for c, h in heads.items() if c not in ("name", "position")}),
                 hide_index=True, width="stretch", height=35 * (len(s) + 1) + 3)     # every player, no scrolling
    g = gk[gk["team"] == team].sort_values("gp", ascending=False)
    gcols = {"name": "Goalie", "role": "Role", "gp": "GP", "w": "W", "l": "L", "otl": "OTL", "gaa": "GAA"}
    st.dataframe(g[list(gcols)].rename(columns=gcols).astype({"GP": int, "W": int, "L": int, "OTL": int}).round({"GAA": 2}),
                 hide_index=True, width="stretch")
    st.caption("A goalie's GAA leaves out empty-net goals and shootouts; each start counts as 60 minutes. No save % yet.")


def show_many(res, stamp, real_part):
    """10,000 seasons: playoff odds and point ranges for teams, and stat-line ranges for players."""
    t = res["table"]
    cols = {"team": st.column_config.TextColumn("Team"),
            "points": st.column_config.NumberColumn("Points", format="%.1f", help="Average over the simulated seasons"),
            "p10": st.column_config.NumberColumn("10th", format="%.0f"), "p90": st.column_config.NumberColumn("90th", format="%.0f"),
            "playoff_pct": st.column_config.NumberColumn("Playoffs %", format="%.1f"),
            "division_pct": st.column_config.NumberColumn("Division %", format="%.1f"),
            "presidents_pct": st.column_config.NumberColumn("Presidents' %", format="%.1f"),
            "avg_rank": st.column_config.NumberColumn("Avg rank", format="%.1f")}
    st.subheader("Standings: averages and odds")
    divs = sorted(t["division"].unique())
    for tab, div in zip(st.tabs(divs), divs):
        with tab:
            st.dataframe(t[t["division"] == div].sort_values("points", ascending=False)[list(cols)], column_config=cols,
                         hide_index=True, width="stretch")

    base = run(stamp, OPENING_NIGHT.isoformat(), (), (), res["n_sims"], res["seed"], stat_lines=False)
    st.subheader("Team focus")
    team = team_picker(res, "sim_team_many")
    row, brow = t.set_index("team").loc[team], base["table"].set_index("team").loc[team]
    m1, m2, m3 = st.columns(3)
    m1.metric("Points (average)", f"{row['points']:.1f}", f"{row['points'] - brow['points']:+.1f} vs. opening night")
    m2.metric("Playoffs", f"{row['playoff_pct']:.1f}%", f"{row['playoff_pct'] - brow['playoff_pct']:+.1f} vs. opening night")
    m3.metric("Range (10th to 90th)", f"{row['p10']:.0f} to {row['p90']:.0f}")
    st.caption("\"vs. opening night\": the opening-night baseline (no trades, nobody sitting), same seed.")

    sk = res["skaters"]
    rng_cols = {"name": "Player", "team": "Team", "position": "Pos", "gp": "GP", "g": "G", "g_p10": "G 10th", "g_p90": "G 90th",
                "a": "A", "a_p10": "A 10th", "a_p90": "A 90th", "p": "P", "p_p10": "P 10th", "p_p90": "P 90th"}
    fmt = {h: st.column_config.NumberColumn(h, format="%.1f" if h in ("G", "A", "P", "GP") else "%.0f")
           for c, h in rng_cols.items() if c not in ("name", "team", "position")}
    note = f" Real stats before {res['start_date']} are included." if real_part else ""
    st.markdown(f"**{team}: player ranges** (average and 10th to 90th over {res['n_sims']:,} seasons)")
    team_cols = [c for c in rng_cols if c != "team"]                 # one team: no need for a Team column
    st.dataframe(sk[sk["team"] == team].sort_values("p", ascending=False)[team_cols].rename(columns=rng_cols),
                 column_config=fmt, hide_index=True, width="stretch")
    st.markdown("**League top 25 scorers** (average points)")
    st.dataframe(sk.head(25)[list(rng_cols)].rename(columns=rng_cols), column_config=fmt, hide_index=True, width="stretch")
    st.caption("Fixed lineups at full health: a dressed skater plays every game, so totals run a little high." + note)
