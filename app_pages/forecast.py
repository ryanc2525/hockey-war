"""
Forecast page: 2026-27 OVR ratings and projections for every NHL player and prospect. View only.

Reads one file, data/forecast_table_2026-27.csv (written by notebook 20). data/ is gitignored, so the live
site doesn't have it; then this page shows a short message and stops, and the Almanac page keeps working.

To test with a different file (or a missing one), set the environment variable HOCKEY_FORECAST_TABLE:
    HOCKEY_FORECAST_TABLE=/nonexistent.csv /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent    # the project folder (this file is in app_pages/)
DEFAULT_PATH = ROOT / "data" / "forecast_table_2026-27.csv"
FORECAST_PATH = Path(os.environ.get("HOCKEY_FORECAST_TABLE", DEFAULT_PATH))
SEASON = "2026-27"
DEFAULT_TEAM = "DET"

GROUP_NAMES = {"F": "Forward", "D": "Defenseman", "G": "Goalie"}
GROUP_FILTER = {"All positions": None, "Forwards": "F", "Defensemen": "D", "Goalies": "G"}


@st.cache_data
def load_forecast(path, file_mtime):
    """Read the forecast table once; Streamlit reruns this script on every click, so cache it.
    file_mtime isn't used inside; it's part of the cache key, so a rewritten file (new notebook 20 run) is read again."""
    table = pd.read_csv(path)
    table["age"] = table["age"].round().astype(int)
    table["type"] = table["prospect"].map({True: "Prospect", False: "NHL"})
    return table


def ovr_color(ovr):
    """Badge colors by tier: (background, text)."""
    if ovr >= 90:
        return "#C9A227", "#1A1A1A"      # gold
    if ovr >= 80:
        return "#2E7D32", "#FFFFFF"      # green
    if ovr >= 70:
        return "#1565C0", "#FFFFFF"      # blue
    return "#6B6B6B", "#FFFFFF"          # gray


def badge_html(row):
    """The big EA-style OVR badge, with WAR per full-time season underneath."""
    bg, fg = ovr_color(row["ovr"])
    role = "about 50 starts" if row["group"] == "G" else "82 games at full-time minutes"
    return f"""
    <div style="display:flex; align-items:center; gap:20px; margin:8px 0 4px 0;">
      <div style="background:{bg}; color:{fg}; border-radius:14px; width:120px; height:120px;
                  display:flex; flex-direction:column; align-items:center; justify-content:center;
                  box-shadow:0 2px 6px rgba(0,0,0,0.25);">
        <div style="font-size:56px; font-weight:800; line-height:1;">{row['ovr']}</div>
        <div style="font-size:14px; font-weight:700; letter-spacing:2px;">OVR</div>
      </div>
      <div>
        <div style="font-size:28px; font-weight:700;">{row['war_std']:.2f} WAR</div>
        <div style="font-size:14px; opacity:0.75;">per full-time season ({role})</div>
      </div>
    </div>
    """


def num(x, fmt="{:.3f}", missing="-"):
    """Format a number, or a dash if it's missing."""
    return missing if pd.isna(x) else fmt.format(x)


def player_label(i):
    """How a player appears in the search list: name (team, position, NHL or prospect)."""
    r = table.loc[i]
    return f"{r['name']}  ({r['team']}, {r['position']}, {r['type']})"


def show_player(r):
    """The player view: badge, then the details. Prospects also get draft info and 2025-26 stats."""
    st.subheader(r["name"])
    if r["prospect"]:
        st.caption(f"Prospect: the rating means **if he played NHL minutes this season** ({SEASON}).")
    st.html(badge_html(r))

    team_label = "Team (draft team)" if r["prospect"] else "Team"
    lines = [
        f"**Position:** {GROUP_NAMES[r['group']]} ({r['position']})",
        f"**{team_label}:** {r['team']}",
        f"**Age:** {r['age']}",
        f"**Projected WAR/60:** {num(r['proj_war_60'])}",
    ]
    if not r["prospect"]:
        if pd.isna(r["war_2025_minutes"]):
            lines.append("**WAR at his 2025-26 minutes:** - (didn't play in 2025-26)")
        else:
            lines.append(f"**WAR at his 2025-26 minutes:** {r['war_2025_minutes']:.2f} "
                         f"({r['minutes_2025_26']:,.0f} minutes)")
    st.markdown("  \n".join(lines))

    if r["prospect"]:
        st.markdown("**Prospect details**")
        draft = f"{int(r['draft_year'])}, round {int(r['round'])}, pick {int(r['pick'])} overall, by {r['draft_team']}"
        p_lines = [f"**Drafted:** {draft}"]
        if pd.notna(r["league_2025_26"]):
            stats = f"{r['league_2025_26']}, {num(r['gp_2025_26'], '{:.0f}')} GP"
            if r["group"] != "G":
                stats += f", {num(r['points_2025_26'], '{:.0f}')} points"
            p_lines.append(f"**2025-26:** {stats}")
        else:
            p_lines.append("**2025-26:** no league stats")
        if r["group"] != "G" and pd.notna(r["nhle_last2"]):
            p_lines.append(f"**NHLe:** {r['nhle_last2']:.2f} NHL-equivalent points per game (last two seasons)")
        if pd.notna(r["war_std_p10"]):
            p_lines.append(f"**Range (10th to 90th percentile):** {r['war_std_p10']:.2f} to {r['war_std_p90']:.2f} WAR "
                           f"per full-time season ({r['war_60_p10']:.3f} to {r['war_60_p90']:.3f} WAR/60)")
        st.markdown("  \n".join(p_lines))
        if r["group"] == "D":
            st.caption("Defenseman prospects are rated on a ranking prior (d_rank_prior). Their projected WAR/60, "
                       "and the range, is the position average, the same for every D prospect.")
        if r["group"] == "G":
            st.caption("Goalie prospects use the goalie prior from notebook 21 (NHL experience, age and pre-NHL save%). No range yet.")


# Columns shown in the tables, with readable headers and number formats
TABLE_COLUMNS = {
    "name": st.column_config.TextColumn("Player"),
    "team": st.column_config.TextColumn("Team"),
    "position": st.column_config.TextColumn("Pos"),
    "age": st.column_config.NumberColumn("Age", format="%d"),
    "type": st.column_config.TextColumn("NHL / prospect"),
    "ovr": st.column_config.NumberColumn("OVR", format="%d"),
    "war_std": st.column_config.NumberColumn("WAR (full-time)", format="%.2f",
                                             help="WAR per full-time season; compares across positions"),
    "proj_war_60": st.column_config.NumberColumn("WAR/60", format="%.3f"),
    "war_2025_minutes": st.column_config.NumberColumn("WAR at 2025-26 min", format="%.2f"),
    "minutes_2025_26": st.column_config.NumberColumn("2025-26 min", format="%.0f"),
    "draft_year": st.column_config.NumberColumn("Draft", format="%d"),
    "pick": st.column_config.NumberColumn("Pick", format="%d"),
    "league_2025_26": st.column_config.TextColumn("2025-26 league"),
    "gp_2025_26": st.column_config.NumberColumn("GP", format="%d"),
    "points_2025_26": st.column_config.NumberColumn("Pts", format="%d"),
}


def show_table(df, columns, height=None):
    """A sortable table (click a header to sort) with the readable headers above."""
    st.dataframe(df[columns], column_config={c: TABLE_COLUMNS[c] for c in columns},
                 hide_index=True, width="stretch", height=height or "auto")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
st.title(f"Forecast: {SEASON}")

if not FORECAST_PATH.exists():
    st.info("Forecast runs locally for now. The 2026-27 ratings are built from private data that isn't on the "
            "live site yet. The Almanac (menu at the top) works as usual.")
    st.stop()          # stops this page only; the Almanac page is unaffected

table = load_forecast(str(FORECAST_PATH), FORECAST_PATH.stat().st_mtime)
st.caption(f"OVR ratings and projected WAR for {(~table['prospect']).sum():,} NHL players and "
           f"{table['prospect'].sum():,} prospects. Data: MoneyPuck.com, NHL EDGE.")

with st.expander("How to read this"):
    st.markdown(
        """
- **OVR** is an EA NHL-style overall rating from 40 to 99, built from **projected WAR** in a full-time role.
  Replacement level (a player a team can call up for free) is **70**; the median NHL regular at his position is **80**.
- **WAR (full-time)** is the projected wins above replacement over a full-time season: 82 games at full-time
  minutes for skaters, about 50 starts for goalies.
- **The scale is frozen.** The anchors were set once from the 2026-27 projections and never get re-percentiled,
  so in future seasons ratings can rise or fall league-wide.
- **Goalies have their own scale** (their median regular is 80, the top goalie 95), because goalie projections are
  far less certain. To compare a goalie with a skater, use **WAR (full-time)**, which is on one scale for everyone.
- **Defenseman prospects** are ranked by a ranking prior (`d_rank_prior`); their projected WAR/60 is the
  position average, the same for all of them.
- **Prospect ratings mean "if he plays"** NHL minutes this season. Whether he gets them isn't part of the rating.
"""
    )

player_tab, board_tab, team_tab = st.tabs(["Player", "Leaderboard", "Team"])

with player_tab:
    choice = st.selectbox(
        "Player or prospect",
        options=table.index,                 # the table is sorted by OVR, so the best players come first
        format_func=player_label,
        index=None,
        placeholder="Start typing a name, e.g. Seider or McKenna",
        key="forecast_player",
    )
    if choice is None:
        st.info("Pick an NHL player or a prospect to see his rating.")
    else:
        show_player(table.loc[choice])

with board_tab:
    c1, c2 = st.columns(2)
    teams = c1.multiselect("Team", sorted(table["team"].unique()), placeholder="All teams", key="board_teams")
    group = c2.selectbox("Position", list(GROUP_FILTER), key="board_group")
    c3, c4 = st.columns(2)
    kind = c3.radio("Players", ["All", "NHL", "Prospects"], horizontal=True, key="board_kind")
    lo, hi = int(table["age"].min()), int(table["age"].max())
    ages = c4.slider("Age", lo, hi, (lo, hi), key="board_ages")

    board = table[table["age"].between(*ages)]
    if teams:
        board = board[board["team"].isin(teams)]
    if GROUP_FILTER[group]:
        board = board[board["group"] == GROUP_FILTER[group]]
    if kind != "All":
        board = board[board["prospect"] == (kind == "Prospects")]
    st.caption(f"{len(board):,} players. Click a column header to sort.")
    show_table(board, ["name", "team", "position", "age", "type", "ovr", "war_std", "proj_war_60", "war_2025_minutes"],
               height=560)

with team_tab:
    nhl_teams = sorted(table.loc[~table["prospect"], "team"].unique())
    team = st.selectbox("Team", nhl_teams, index=nhl_teams.index(DEFAULT_TEAM), key="team_pick")

    roster = table[~table["prospect"] & (table["team"] == team)].sort_values(["ovr", "war_std"], ascending=False)
    st.markdown(f"**NHL roster** ({len(roster)})")
    st.caption("Each player's last team in our data, including players who didn't play in 2025-26.")
    show_table(roster, ["name", "position", "age", "ovr", "war_std", "proj_war_60", "war_2025_minutes", "minutes_2025_26"])

    prospects = table[table["prospect"] & (table["team"] == team)].sort_values(["ovr", "war_std"], ascending=False)
    st.markdown(f"**Prospects** ({len(prospects)})")
    st.caption("Team = the team that drafted him. Ratings mean \"if he played NHL minutes this season\".")
    show_table(prospects, ["name", "position", "age", "ovr", "war_std", "draft_year", "pick", "league_2025_26",
                           "gp_2025_26", "points_2025_26"])
