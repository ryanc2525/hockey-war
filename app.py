"""
Almanac player cards: search any NHL skater, pick a season, and see his card.
Turn on Compare to put a second player-season next to him in a stat-by-stat table.

    /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py

Reads the slim public card table (app_data/cards.csv, written by notebook 10) only.
No WAR math, no NHL API calls. season = MoneyPuck start year (2025 = 2025-26); the app shows season_label.
"""

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # draw to images, not to a desktop window
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))          # so "from player_card import ..." works, like in the notebooks
import player_card
import compare
importlib.reload(player_card)               # pick up pushed changes to src/ without rebooting the app
importlib.reload(compare)                   # after player_card, because compare imports from it
from player_card import draw_card
from compare import compare_rows, compare_notes, compare_html

CARDS_PATH = ROOT / "app_data" / "cards.csv"
SHOT_GRID_PATH = ROOT / "app_data" / "shot_grid.csv"        # pre-counted 5 ft cells (notebook 12)
SHOT_TOTALS_PATH = ROOT / "app_data" / "shot_totals.csv"    # unblocked shots and goals per player-season


@st.cache_data
def load_cards():
    """Read the card table once; Streamlit reruns this script on every click, so cache it.
    Returns every player-season (with shot-map totals), plus one row per player for the search list."""
    cards = pd.read_csv(CARDS_PATH)
    totals = pd.read_csv(SHOT_TOTALS_PATH)
    cards = cards.merge(totals, on=["playerId", "season"], how="left")
    shot_cols = ["unblocked_shots", "unblocked_goals", "oz_shots"]
    cards[shot_cols] = cards[shot_cols].fillna(0)                 # no unblocked shots that season
    latest = cards.sort_values("season").groupby("playerId").tail(1).set_index("playerId")   # most recent season
    players = latest[["name", "team", "position"]].copy()
    players["total_war"] = cards.groupby("playerId")["war"].sum()
    players = players.sort_values("total_war", ascending=False)
    return cards, players


@st.cache_data
def load_shot_grid():
    """Shot-map cells indexed by (playerId, season), so one player-season's rows are a quick lookup."""
    return pd.read_csv(SHOT_GRID_PATH).set_index(["playerId", "season"]).sort_index()


def shot_rows(player_id, season):
    """This player-season's shot-map cells (always a table, possibly empty)."""
    if (player_id, season) in shot_grid.index:
        return shot_grid.loc[[(player_id, season)]].reset_index(drop=True)
    return shot_grid.iloc[:0].reset_index(drop=True)


def pick(prefix, label):
    """Player search, then a season picker with only his seasons. Returns his card row, or None."""
    # Show team and position (from his latest season) next to each name, so players with the same name can be told apart
    player_id = st.selectbox(
        label,
        options=players.index,
        format_func=lambda pid: f"{players.at[pid, 'name']}  ({players.at[pid, 'team']}, {players.at[pid, 'position']})",
        index=None,
        placeholder="Start typing a name, e.g. Seider",
        key=f"{prefix}_player",
    )
    if player_id is None:
        return None
    seasons = cards[cards["playerId"] == player_id].sort_values("season", ascending=False).set_index("season")
    season = st.selectbox(
        "Season",
        options=seasons.index,                              # only the seasons he played, newest first
        format_func=lambda s: seasons.at[s, "season_label"],
        key=f"{prefix}_season_{player_id}",                 # a new player starts at his newest season
    )
    row = seasons.loc[season].copy()
    row["season"] = season
    return row


st.set_page_config(page_title="Hockey WAR player cards", layout="centered")
st.title("Hockey WAR player cards")
st.caption("2021-22 to 2025-26 NHL skaters. Type a name to search. Data: MoneyPuck.com, NHL EDGE.")

cards, players = load_cards()
shot_grid = load_shot_grid()

a = pick("a", "Player")
if a is None:
    st.info("Pick a player to see his card. The list is sorted by total WAR from 2021-22 to 2025-26, highest first.")
else:
    card_slot = st.container()                          # the card goes here, unless the comparison replaces it
    show_card = True
    if st.toggle("Compare", key="compare"):             # a toggle (not a button) stays on across reruns
        b = pick("b", "Compare with")
        if b is not None:
            if b["playerId"] == a["playerId"] and b["season"] == a["season"]:
                st.info("That's the same player and season on both sides. Pick a different player or season to compare.")
            else:
                for note in compare_notes(a, b):
                    st.caption(note)
                st.html(compare_html(a, b, compare_rows(a, b)))
                show_card = False
    if show_card:
        with card_slot:
            fig = draw_card(a, a["season_label"],       # edge=None: uses the saved EDGE columns in the CSV
                            shot_grid=shot_rows(int(a["playerId"]), int(a["season"])))
            st.pyplot(fig)
            plt.close(fig)                              # free memory; each pick draws a new figure
