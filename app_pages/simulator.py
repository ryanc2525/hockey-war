"""
Simulator page: play the rest of the 2026-27 season thousands of times, with the trades and sit-outs you pick,
and see each team's points and playoff odds.

Reads the bundle notebook 26 saves (data/simulator_2026-27/: players.csv, schedule.csv, settings.json) and the
hand-kept trades file (data/trades_2026-27.csv). data/ is gitignored, so the live site doesn't have them; then
this page shows a short message and stops, and the Almanac page keeps working. Nothing here recomputes WAR or
calls the NHL: the simulator itself is src/season.py.

To test with the missing-files message, point it somewhere empty:
    HOCKEY_SIMULATOR_DIR=/nonexistent /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent           # the project folder (this file is in app_pages/)
BUNDLE_DIR = Path(os.environ.get("HOCKEY_SIMULATOR_DIR", ROOT / "data" / "simulator_2026-27"))
TRADES_PATH = ROOT / "data" / "trades_2026-27.csv"
NEEDED = [BUNDLE_DIR / "players.csv", BUNDLE_DIR / "schedule.csv", BUNDLE_DIR / "settings.json", TRADES_PATH]
DEFAULT_TEAM = "DET"
OPENING_NIGHT = date(2026, 9, 29)

st.title("Simulator: 2026-27")

if not all(p.exists() for p in NEEDED):
    st.info("Simulator runs locally for now. It needs the season inputs built by notebook 26, which aren't on "
            "the live site yet. The Almanac (menu at the top) works as usual.")
    st.stop()          # stops this page only; the Almanac page is unaffected

# Imported only now: the live site stops above, so it never needs the simulator's extra packages (scipy).
sys.path.append(str(ROOT / "src"))
import season as SE  # noqa: E402


# ---------------------------------------------------------------------------
# Loading and running (cached, because Streamlit reruns this file on every click)
# ---------------------------------------------------------------------------
def file_stamp():
    """The files' modification times: part of every cache key, so a new notebook 26 run is picked up."""
    return tuple(p.stat().st_mtime for p in NEEDED)


@st.cache_resource
def load(stamp):
    """Read the bundle once (stamp isn't used inside; it only makes a rewritten file load again)."""
    return SE.load_bundle(BUNDLE_DIR, TRADES_PATH)


@st.cache_data(show_spinner=False, max_entries=20)
def run(stamp, start_date, trade_ids, sit_out, n_sims, seed):
    """One scenario. Cached by its inputs: the same inputs come back instantly the second time."""
    res = SE.run_scenario(load(stamp), start_date=start_date, trade_ids=list(trade_ids), sit_out=list(sit_out),
                          n_sims=n_sims, seed=seed)
    keep = ["table", "points", "lineups", "start_date", "trades", "sit_out", "n_sims", "seed",
            "games_simulated", "games_locked", "seconds", "teams"]
    return {k: res[k] for k in keep}


stamp = file_stamp()
B = load(stamp)
players = B["players"]
trades = B["trades"]
last_day = date.fromisoformat(B["schedule_downloaded"][:10])      # results are saved through the day before

st.caption(f"The real 2026-27 schedule (84 games per team), with real results of every game before {last_day} "
           "(the day it was downloaded; notebook 26 refreshes it). Each run plays the season thousands of times. "
           "Data: MoneyPuck.com, NHL.")

with st.expander("How it works"):
    st.markdown(
        """
- **What's simulated:** every game on or after the start date, thousands of times. Each game's score is random,
  tilted by the two lineups' projected WAR (skaters and the starting goalie), home ice and each team's bonus.
  Late empty-net and 6-on-5 goals, overtime and shootouts follow what really happens in NHL games.
- **What's locked in:** every real game before the start date counts with its real result (points, wins,
  regulation wins). Opening night (Sept 29) locks in nothing.
- **Lineups are fixed at the start date.** Each team dresses 12 forwards (4 centers), 6 defensemen and 2 goalies
  by notebook 23's coach score, after the trades you tick and the players you sit. Everyone else is at
  **full health**: no injuries and no in-season changes.
- **Goalies:** the backup starts the second night of every back-to-back, plus enough other games (spread evenly)
  to make about 30 of 84; the starter plays the rest.
- **Each player's season is random too:** better or worse than projected, by how much players like him
  usually miss (notebook 17).
- **Team bonus:** about 40% of how much each team beat (or missed) what its players predicted in 2025-26.
  It stays with the team through trades and sit-outs.
- **Playoffs:** top 3 in each division plus 2 wild cards per conference. Ties: regulation wins, then
  regulation + overtime wins, then a coin flip (head-to-head, goal difference and goals for are skipped).
- **Results are a range, not a prediction.** The 10th-90th range is what a single season can look like.
"""
    )

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
st.subheader("Scenario")
start = st.date_input("Start date", value=OPENING_NIGHT, min_value=OPENING_NIGHT, max_value=last_day,
                      help="Real results before this date are locked in; the rest of the season is simulated.")
start_str = start.isoformat()
default_trades = set(SE.default_trades(trades, start_str))

st.markdown("**Trades** (ticked: made on or before the start date; tick or untick any)")
chosen_trades = []
for tid, t in trades.groupby("trade_id", sort=True):
    moves = "; ".join(f"{n} {a} to {b}" for n, a, b in zip(t["name"], t["from_team"], t["to_team"]))
    # The key includes the start date, so changing the date resets the ticks to that date's default
    if st.checkbox(f"{tid} ({t['date'].iloc[0]}): {moves}", value=tid in default_trades, key=f"trade_{tid}_{start_str}"):
        chosen_trades.append(tid)

pool = players[players["in_pool"]].sort_values(["team", "name"])
labels = {pid: f"{n} ({tm}, {pos})" for pid, n, tm, pos in zip(pool["playerId"], pool["name"], pool["team"], pool["position"])}
sit_out = st.multiselect("Players sitting out (type to search)", options=list(labels), format_func=labels.get,
                         help="They leave their team's lineup for every simulated game; the next player up dresses.")
c1, c2 = st.columns(2)
n_sims = c1.radio("Simulated seasons", [1000, 10000], index=1, horizontal=True,
                  help="10,000 is steadier and takes several seconds; 1,000 is quick.")
seed = int(c2.number_input("Seed", value=0, step=1, help="The same inputs and seed always give the same results."))
inputs = (start_str, tuple(sorted(chosen_trades)), tuple(sorted(sit_out)), int(n_sims), seed)
if st.button("Run", type="primary") or "inputs" not in st.session_state:
    st.session_state["inputs"] = inputs           # the first visit runs the defaults (opening night)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
shown = st.session_state["inputs"]
with st.spinner("Simulating..."):
    res = run(stamp, *shown)
    base = run(stamp, OPENING_NIGHT.isoformat(), (), (), shown[3], shown[4])   # opening night, nothing changed
if shown != inputs:
    st.warning("The controls have changed since this run. Press **Run** to update the results.")

name_of = players.set_index("playerId")["name"]
st.markdown(
    f"**Showing:** from {res['start_date']} ({res['games_locked']} real games locked in, "
    f"{res['games_simulated']:,} simulated), {res['n_sims']:,} seasons, seed {res['seed']}. "
    f"Trades: {', '.join(res['trades']) or 'none'}. "
    f"Sitting out: {', '.join(name_of[p] for p in res['sit_out']) or 'nobody'}. "
    f"({res['seconds']:.1f} seconds the first time; cached after that.)")

COLS = {
    "team": st.column_config.TextColumn("Team"),
    "points": st.column_config.NumberColumn("Points", format="%.1f", help="Average over the simulated seasons"),
    "p10": st.column_config.NumberColumn("10th", format="%.0f", help="10% of seasons end at or below this"),
    "p90": st.column_config.NumberColumn("90th", format="%.0f", help="10% of seasons end at or above this"),
    "wins": st.column_config.NumberColumn("W", format="%.1f"),
    "losses": st.column_config.NumberColumn("L", format="%.1f"),
    "otl": st.column_config.NumberColumn("OTL", format="%.1f"),
    "goal_diff": st.column_config.NumberColumn("GD", format="%+.0f"),
    "playoff_pct": st.column_config.NumberColumn("Playoffs %", format="%.1f"),
    "division_pct": st.column_config.NumberColumn("Division %", format="%.1f"),
    "presidents_pct": st.column_config.NumberColumn("Presidents' %", format="%.1f"),
    "avg_rank": st.column_config.NumberColumn("Avg rank", format="%.1f", help="Final league rank, 1 = best"),
    "real_points": st.column_config.NumberColumn("Real pts so far", format="%d"),
    "real_gp": st.column_config.NumberColumn("GP so far", format="%d"),
}

st.subheader("Standings")
table = res["table"]
cols = list(COLS) if res["games_locked"] else [c for c in COLS if not c.startswith("real")]
tabs = st.tabs(sorted(table["division"].unique()))
for tab, div in zip(tabs, sorted(table["division"].unique())):
    with tab:
        d = table[table["division"] == div].sort_values("points", ascending=False)
        st.dataframe(d[cols], column_config=COLS, hide_index=True, width="stretch")   # click a header to sort

# ---------------------------------------------------------------------------
# One team up close
# ---------------------------------------------------------------------------
st.subheader("Team focus")
teams = list(res["teams"])
team = st.selectbox("Team", teams, index=teams.index(DEFAULT_TEAM))
i = teams.index(team)
row = table.set_index("team").loc[team]
brow = base["table"].set_index("team").loc[team]
m1, m2, m3 = st.columns(3)
m1.metric("Points (average)", f"{row['points']:.1f}", f"{row['points'] - brow['points']:+.1f} vs. opening night",
          delta_color="normal")
m2.metric("Playoffs", f"{row['playoff_pct']:.1f}%", f"{row['playoff_pct'] - brow['playoff_pct']:+.1f} vs. opening night")
m3.metric("Range (10th to 90th)", f"{row['p10']:.0f} to {row['p90']:.0f}")
st.caption("\"vs. opening night\" compares with the opening-night baseline (no trades, nobody sitting) with the same "
           "number of seasons and seed.")

pts = res["points"][:, i]
dist = pd.Series(pts).value_counts(normalize=True).sort_index() * 100
dist = dist.reindex(np.arange(int(pts.min()), int(pts.max()) + 1), fill_value=0)
st.markdown(f"**{team}'s final points**, share of simulated seasons (%)")
st.bar_chart(dist.rename_axis("points").rename("% of seasons"))

lu = res["lineups"]
lu = lu[(lu["team"] == team) & lu["dressed"]].copy()
lu["OVR"] = lu["ovr"].round().astype("Int64")
lu["Minutes"] = lu["minutes_per_game"].round(1)
lu["Note"] = np.where(lu["status"] == "NHL roster", "", lu["status"]) + np.where(lu["flag"].fillna("") != "", " " + lu["flag"].fillna(""), "")
st.markdown(f"**{team}'s lineup in this run** (fixed for every simulated game)")
f = lu[lu["group"] == "F"]
lines = []
for k in range(1, 5):
    ln = f[f["line"] == k]
    c = ln[ln["role"] == "C"]
    w = ln[ln["role"] == "W"]
    for _, r in pd.concat([w.head(1), c, w.iloc[1:]]).iterrows():      # left wing, center, right wing
        lines.append({"Line": f"F{k}", "Role": r["role"], "Player": r["name"], "OVR": r["OVR"], "Minutes": r["Minutes"],
                      "Note": r["Note"]})
d_ = lu[lu["group"] == "D"].sort_values("rank")
for _, r in d_.iterrows():
    lines.append({"Line": f"D{int(r['line'])}", "Role": "D", "Player": r["name"], "OVR": r["OVR"], "Minutes": r["Minutes"],
                  "Note": r["Note"]})
starts = B["goalie_starts"][(B["schedule"]["date"] >= res["start_date"]).to_numpy()]
games = B["schedule"][(B["schedule"]["date"] >= res["start_date"]).to_numpy()]
backup = int(((games["home"] == team) & starts["home_backup"]).sum() + ((games["away"] == team) & starts["away_backup"]).sum())
total = int(((games["home"] == team) | (games["away"] == team)).sum())
for _, r in lu[lu["group"] == "G"].sort_values("rank").iterrows():
    n_starts = total - backup if r["rank"] == 1 else backup
    lines.append({"Line": "G", "Role": "Starter" if r["rank"] == 1 else "Backup", "Player": r["name"], "OVR": r["OVR"],
                  "Minutes": 60.0, "Note": f"{n_starts} of {total} starts" + (f"; {r['Note'].strip()}" if r["Note"].strip() else "")})
st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch", height=35 * (len(lines) + 1) + 3,   # all rows, no scrolling
             column_config={"OVR": st.column_config.NumberColumn("OVR", format="%d"),
                            "Minutes": st.column_config.NumberColumn("Min/game", format="%.1f")})
st.caption("OVR is the Forecast page's rating. Minutes per game come from notebook 24's blend of the slot and the "
           "player's own recent ice time; a goalie plays the full 60 of every game he starts. Note shows starts, anyone "
           "not on the NHL roster (a dispute, a call-up) and lineup rules used.")
