"""
Almanac player cards: search any NHL skater, pick a season, and see his card.

    /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py

Reads the slim public card table (app_data/cards.csv, written by notebook 10) only.
No WAR math, no NHL API calls. season = MoneyPuck start year (2025 = 2025-26); the app shows season_label.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # draw to images, not to a desktop window
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))          # so "from player_card import ..." works, like in the notebooks
from player_card import draw_card

CARDS_PATH = ROOT / "app_data" / "cards.csv"


@st.cache_data
def load_cards():
    """Read the card table once; Streamlit reruns this script on every click, so cache it.
    Returns every player-season, plus one row per player for the search list."""
    cards = pd.read_csv(CARDS_PATH)
    latest = cards.sort_values("season").groupby("playerId").tail(1).set_index("playerId")   # most recent season
    players = latest[["name", "team", "position"]].copy()
    players["total_war"] = cards.groupby("playerId")["war"].sum()
    players = players.sort_values("total_war", ascending=False)
    return cards, players


st.set_page_config(page_title="Hockey WAR player cards", layout="centered")
st.title("Hockey WAR player cards")
st.caption("2021-22 to 2025-26 NHL skaters. Type a name to search. Data: MoneyPuck.com, NHL EDGE.")

cards, players = load_cards()

# Show team and position (from his latest season) next to each name, so players with the same name can be told apart
player_id = st.selectbox(
    "Player",
    options=players.index,
    format_func=lambda pid: f"{players.at[pid, 'name']}  ({players.at[pid, 'team']}, {players.at[pid, 'position']})",
    index=None,
    placeholder="Start typing a name, e.g. Seider",
)

if player_id is None:
    st.info("Pick a player to see his card. The list is sorted by total WAR from 2021-22 to 2025-26, highest first.")
else:
    seasons = cards[cards["playerId"] == player_id].sort_values("season", ascending=False).set_index("season")
    season = st.selectbox(
        "Season",
        options=seasons.index,                              # only the seasons he played, newest first
        format_func=lambda s: seasons.at[s, "season_label"],
        key=f"season_{player_id}",                          # a new player starts at his newest season
    )
    row = seasons.loc[season]
    fig = draw_card(row, row["season_label"])           # edge=None: uses the saved EDGE columns in the CSV
    st.pyplot(fig)
    plt.close(fig)                                      # free memory; each pick draws a new figure
