"""
Almanac player cards: search any NHL skater and see his 2025-26 card.

    /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py

Reads the slim public card table (app_data/cards_2025-26.csv, written by notebook 09) only.
No WAR math, no NHL API calls.
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

SEASON_LABEL = "2025-26"
CARDS_PATH = ROOT / "app_data" / "cards_2025-26.csv"


@st.cache_data
def load_cards():
    """Read the card table once; Streamlit reruns this script on every click, so cache it."""
    cards = pd.read_csv(CARDS_PATH)
    return cards.sort_values("war", ascending=False).reset_index(drop=True)


st.set_page_config(page_title="Hockey WAR player cards", layout="centered")
st.title("Hockey WAR player cards")
st.caption(f"{SEASON_LABEL} NHL skaters. Type a name to search. Data: MoneyPuck.com, NHL EDGE.")

cards = load_cards()

# Show team and position next to each name, so players with the same name (two Elias Petterssons) can be told apart
choice = st.selectbox(
    "Player",
    options=cards.index,
    format_func=lambda i: f"{cards.at[i, 'name']}  ({cards.at[i, 'team']}, {cards.at[i, 'position']})",
    index=None,
    placeholder="Start typing a name, e.g. Seider",
)

if choice is None:
    st.info("Pick a player to see his card. The list is sorted by WAR, highest first.")
else:
    fig = draw_card(cards.loc[choice], SEASON_LABEL)   # edge=None: uses the saved EDGE columns in the CSV
    st.pyplot(fig)
    plt.close(fig)                                      # free memory; each pick draws a new figure
