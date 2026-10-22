"""
Side-by-side comparison of two player-seasons (Stathead style) for the card app.

    from compare import compare_rows, compare_notes, compare_html
    rows = compare_rows(a, b)          # a, b = one row each from app_data/cards.csv

Rows, labels, formats and directions come from player_card (SECTIONS, EDGE_ROWS),
so the table always matches the card. Winners are decided on the values as displayed.

    /opt/anaconda3/envs/hockey/bin/python src/compare.py     # prints two test comparisons
"""

import html

import numpy as np
import pandas as pd

from player_card import SECTIONS, EDGE_ROWS, HIDDEN_ROWS, MIN_EV_TOI, SURFACE, INK, MUTED, TRACK, STYLE_GRAY

MISSING = "–"
WIN_BG = "#fbe0df"                # light version of the card's red (better)
EDGE_SECTION = "Skating and shot  (NHL EDGE)"
LABEL_SUFFIX = {"style": "  (style)", "context": "  (team context)"}

# Header totals, same labels as the card. direction None = shown, never highlighted. fmt None = m:ss.
TOTALS = [
    ("games_played", "GP",     None,   "{:.0f}"),
    ("goals",        "G",      "high", "{:.0f}"),
    ("assists",      "A",      "high", "{:.0f}"),
    ("points",       "P",      "high", "{:.0f}"),
    ("toi_per_gp",   "TOI/GP", None,   None),
]


# ---------------------------------------------------------------------------
# 1. Rows and winners
# ---------------------------------------------------------------------------
def _row_specs():
    """(section, column, label, direction, fmt) for every compared row, in card order."""
    specs = [("Totals", col, label, direction, fmt) for col, label, direction, fmt in TOTALS]
    for title, rows in SECTIONS:
        specs += [(title, col, label, direction, fmt) for col, label, direction, fmt in rows]
    specs += [(EDGE_SECTION, key, label, "high", fmt)          # EDGE rows are all higher = better (as in add_edge)
              for key, label, fmt in EDGE_ROWS if key not in HIDDEN_ROWS]
    return specs


def _shown(value, fmt):
    """(text as displayed on the card, that text as a number). Missing -> ('–', NaN)."""
    if pd.isna(value):
        return MISSING, np.nan
    if fmt is None:                                         # TOI per game, m:ss like the card header
        mins, secs = divmod(round(value * 60), 60)
        return f"{mins}:{secs:02d}", np.nan
    text = fmt.format(value)
    return text, float(text.replace("+", "").replace("%", ""))


def _winner(a_num, b_num, direction):
    """'a', 'b', or None. Ties, missing values, and style/context rows have no winner."""
    if direction not in ("high", "low") or np.isnan(a_num) or np.isnan(b_num) or a_num == b_num:
        return None
    a_better = a_num > b_num if direction == "high" else a_num < b_num
    return "a" if a_better else "b"


def compare_rows(a, b):
    """One row per stat, in card order: section, label, direction, a_text, b_text, a_pct, b_pct, winner.
    a, b = card rows (Series). Values are compared as displayed, so equal displayed values tie."""
    out = []
    for section, col, label, direction, fmt in _row_specs():
        a_text, a_num = _shown(a[col], fmt)
        b_text, b_num = _shown(b[col], fmt)
        out.append({
            "section": section, "label": label, "direction": direction,
            "a_text": a_text, "b_text": b_text,
            "a_pct": a.get(f"pct_{col}", np.nan), "b_pct": b.get(f"pct_{col}", np.nan),
            "winner": _winner(a_num, b_num, direction),
        })
    return pd.DataFrame(out)


def ordinal(pct):
    """95 -> '95th', 1 -> '1st', 22 -> '22nd', 13 -> '13th'. Missing -> ''."""
    if pd.isna(pct):
        return ""
    n = int(round(pct))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def compare_notes(a, b):
    """Short notes to show above the table, when relevant."""
    notes = []
    if a["pos_group"] != b["pos_group"]:
        notes.append("Different positions: percentiles compare each player to his own position.")
    for p in (a, b):
        if not p["qualified"]:
            notes.append(f"Small sample: {p['name']} ({p['season_label']}) played under {MIN_EV_TOI} minutes "
                         "at 5v5, so his percentiles are less reliable.")
    return notes


# ---------------------------------------------------------------------------
# 2. HTML table (light like the card, readable on phones)
# ---------------------------------------------------------------------------
def _player_head(p, align):
    bits = [p["team"], p["position"], p["season_label"]]
    if not pd.isna(p.get("age")):
        bits.append(f"Age {p['age']:.0f}")
    return (f'<th style="text-align:{align};padding:10px 8px;vertical-align:top;font-weight:400">'
            f'<div style="font-size:16px;font-weight:700;color:{INK}">{html.escape(str(p["name"]))}</div>'
            f'<div style="font-size:12px;color:{MUTED}">'
            + "  ·  ".join(f'<span style="white-space:nowrap">{html.escape(str(x))}</span>' for x in bits)
            + '</div></th>')


def _value_cell(text, pct, win, gray):
    style = f"text-align:center;padding:6px 4px;border-top:1px solid {TRACK};color:{STYLE_GRAY if gray else INK};"
    if win:
        style += f"background:{WIN_BG};font-weight:700;"
    sub = f'<div style="font-size:11px;font-weight:400;color:{MUTED}">{ordinal(pct)}</div>' if ordinal(pct) else ""
    return f'<td style="{style}">{html.escape(text)}{sub}</td>'


def compare_html(a, b, rows):
    """The comparison as one HTML table: player A left, stat names in the middle, player B right."""
    body, section = [], None
    for r in rows.itertuples():
        if r.section != section:
            section = r.section
            body.append(f'<tr><td colspan="3" style="padding:14px 8px 4px;font-size:11px;font-weight:700;'
                        f'letter-spacing:0.04em;color:{MUTED}">{html.escape(section.upper())}</td></tr>')
        gray = r.direction in ("style", "context")
        label = r.label + LABEL_SUFFIX.get(r.direction, "")
        body.append(
            "<tr>"
            + _value_cell(r.a_text, r.a_pct, r.winner == "a", gray)
            + f'<td style="text-align:center;padding:6px 4px;border-top:1px solid {TRACK};font-size:13px;'
              f'color:{STYLE_GRAY if gray else MUTED};overflow-wrap:anywhere">{html.escape(label)}</td>'
            + _value_cell(r.b_text, r.b_pct, r.winner == "b", gray)
            + "</tr>")

    return (
        f'<div style="background:{SURFACE};border-radius:8px;padding:8px;color:{INK};'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif">'
        f'<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:14px">'
        f'<colgroup><col style="width:34%"><col style="width:32%"><col style="width:34%"></colgroup>'
        f'<thead><tr>{_player_head(a, "left")}<th></th>{_player_head(b, "right")}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
        f'<div style="font-size:11px;color:{MUTED};padding:10px 8px 4px">Small text = percentile vs. qualified players '
        f'at the same position. Red = better on that row. Gray = style or team context, not compared. '
        f'Data: MoneyPuck.com, NHL EDGE.</div></div>')


# ---------------------------------------------------------------------------
# Check by eye: python src/compare.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    cards = pd.read_csv(Path(__file__).resolve().parent.parent / "app_data" / "cards.csv")

    def find(name, season):
        return cards[(cards["name"] == name) & (cards["season"] == season)].iloc[0]

    pd.set_option("display.width", 200)
    for a, b in [(find("Connor McDavid", 2022), find("Macklin Celebrini", 2025)),
                 (find("Connor McDavid", 2025), find("Moritz Seider", 2025))]:
        print(f"\nA = {a['name']} ({a['position']}) {a['season_label']}    B = {b['name']} ({b['position']}) {b['season_label']}")
        for note in compare_notes(a, b):
            print("Note:", note)
        t = compare_rows(a, b)
        t["a_pct"], t["b_pct"] = t["a_pct"].map(ordinal), t["b_pct"].map(ordinal)
        t["winner"] = t["winner"].fillna("")
        t["direction"] = t["direction"].fillna("shown")
        print(t[["label", "direction", "a_text", "a_pct", "b_text", "b_pct", "winner"]].to_string(index=False))
