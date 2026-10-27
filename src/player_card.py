"""
Almanac player cards: percentile tables and a Baseball Savant-style card.

    from player_card import build_metrics, add_percentiles, fetch_edge, draw_card

Percentiles compare each player to qualified players at his position
(forwards vs. forwards, defensemen vs. defensemen) in the same season.

Data: MoneyPuck.com (season stats) and NHL EDGE (skating and shot speed).
"""

import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from war_model import rate60, run_season, V04_PARAMS

MIN_EV_TOI = 200  # minutes at 5v5 to qualify; below this the card is drawn faint

# ---------------------------------------------------------------------------
# Card metrics. direction: "high" = higher is better, "low" = lower is better,
# "style" = describes how he plays, not good or bad (drawn in gray),
# "context" = mostly reflects his team, not him (drawn in gray; percentile ranks lower = better).
# ---------------------------------------------------------------------------
SECTIONS = [
    ("Value", [
        ("war",       "Total WAR",               "high",  "{:.1f}"),
        ("off_war",   "Offensive WAR",           "high",  "{:.1f}"),
        ("def_war",   "Defensive WAR",           "high",  "{:.1f}"),
        ("other_war", "Other WAR (pen + FO)",    "high",  "{:.1f}"),
    ]),
    ("Scoring  (5v5, per 60)", [
        ("g_60",      "Goals",                   "high",  "{:.2f}"),
        ("ixg_60",    "Expected goals",          "high",  "{:.2f}"),
        ("sog_60",    "Shots on goal",           "high",  "{:.1f}"),
        ("fin_60",    "Finishing (G minus xG)",  "high",  "{:+.2f}"),
    ]),
    ("Playmaking  (5v5, per 60)", [
        ("a1_60",     "Primary assists",         "high",  "{:.2f}"),
        ("a_60",      "All assists",             "high",  "{:.2f}"),
        ("pts_60",    "Points",                  "high",  "{:.2f}"),
    ]),
    ("Defense  (5v5)", [
        ("ev_rel_xga_60", "xGA/60 vs. team on bench", "high", "{:+.2f}"),
        ("onice_xga_60",  "On-ice xGA/60",            "context", "{:.2f}"),
    ]),
    ("Stuff  (5v5, per 60)", [
        ("hits_60",   "Hits",                    "style", "{:.1f}"),
        ("blk_60",    "Blocked shots",           "style", "{:.1f}"),
        ("take_60",   "Takeaways",               "style", "{:.2f}"),   # both track puck possession, so
        ("give_60",   "Giveaways",               "style", "{:.2f}"),   # stars rank low on giveaways
        ("take_share", "Takeaway share",         "high",  "{:.0%}"),   # takeaways / (takeaways + giveaways)
        ("pend_60",   "Penalties drawn (all)",   "high",  "{:.2f}"),
    ]),
]

EDGE_ROWS = [
    ("top_speed",  "Top skating speed (mph)",       "{:.1f}"),
    ("bursts_20_60", "Bursts over 20 mph (per 60)", "{:.2f}"),
    ("shot_speed", "Top shot speed (mph)",          "{:.1f}"),
    ("oz_pct_ev",  "Offensive zone time (EV)",      "{:.0%}"),
]
HIDDEN_ROWS = {"oz_pct_ev"}   # still computed and saved by add_edge, just not drawn; remove a key to bring it back

# Before 2024-25, home-arena scorers recorded takeaways and giveaways very differently, and the NHL's
# recording changed in 2024-25. Rows in ERA_SENSITIVE are drawn gray (not rated) for earlier seasons.
FIRST_CLEAN_TRACKING_SEASON = 2024   # MoneyPuck start year: 2024 = 2024-25
ERA_SENSITIVE = {"take_share"}

# ---------------------------------------------------------------------------
# 1. Metrics table
# ---------------------------------------------------------------------------
RAW_EXTRA = ["I_F_shotsOnGoal", "I_F_hits", "shotsBlockedByPlayer", "I_F_takeaways", "I_F_giveaways"]


def build_metrics(war_table, raw_path):
    """One row per player with every card metric. war_table = output of run_season."""
    raw = pd.read_csv(raw_path)
    ev = (raw.loc[raw["situation"] == "5on5", ["playerId"] + RAW_EXTRA]
             .rename(columns={c: f"ev_{c}" for c in RAW_EXTRA}))
    m = war_table.merge(ev, on="playerId", how="left")
    t = m["ev_toi"]

    m["g_60"] = rate60(m["ev_I_F_goals"], t)
    m["ixg_60"] = rate60(m["ev_I_F_xGoals"], t)
    m["sog_60"] = rate60(m["ev_I_F_shotsOnGoal"], t)
    m["fin_60"] = rate60(m["ev_I_F_goals"] - m["ev_I_F_xGoals"], t)
    m["a1_60"] = rate60(m["ev_I_F_primaryAssists"], t)
    m["a_60"] = rate60(m["ev_I_F_primaryAssists"] + m["ev_I_F_secondaryAssists"], t)
    m["pts_60"] = rate60(m["ev_I_F_goals"] + m["ev_I_F_primaryAssists"] + m["ev_I_F_secondaryAssists"], t)
    m["onice_xga_60"] = rate60(m["ev_OnIce_A_xGoals"], t)
    m["hits_60"] = rate60(m["ev_I_F_hits"], t)
    m["blk_60"] = rate60(m["ev_shotsBlockedByPlayer"], t)
    m["take_60"] = rate60(m["ev_I_F_takeaways"], t)
    m["give_60"] = rate60(m["ev_I_F_giveaways"], t)
    tg = m["ev_I_F_takeaways"] + m["ev_I_F_giveaways"]
    m["take_share"] = np.where(tg > 0, m["ev_I_F_takeaways"] / tg, np.nan)    # NaN if both are 0
    m["pend_60"] = rate60(m["all_penaltiesDrawn"], m["all_toi"])

    # Season totals for the card header (all situations)
    m["goals"] = m["all_I_F_goals"]
    m["assists"] = m["all_I_F_primaryAssists"] + m["all_I_F_secondaryAssists"]
    m["points"] = m["goals"] + m["assists"]
    m["toi_per_gp"] = m["all_toi"] / m["games_played"]
    m["qualified"] = m["ev_toi"] >= MIN_EV_TOI
    return m


# ---------------------------------------------------------------------------
# 2. Percentiles (vs. qualified players at the same position)
# ---------------------------------------------------------------------------
def percentile_in(values, pool, direction):
    """Percentile (1-100) of each value within pool. Ties get the midpoint."""
    pool = np.sort(pool[~np.isnan(pool)])
    lo = np.searchsorted(pool, values, side="left")
    hi = np.searchsorted(pool, values, side="right")
    pct = (lo + hi) / 2 / len(pool) * 100
    if direction in ("low", "context"):
        pct = 100 - pct
    pct = np.clip(np.round(pct), 1, 100)
    return np.where(np.isnan(values), np.nan, pct)


def add_percentiles(metrics):
    """Add a pct_<metric> column for every card metric."""
    m = metrics.copy()
    for _, rows in SECTIONS:
        for col, _, direction, _ in rows:
            m[f"pct_{col}"] = np.nan
            for pos in ["F", "D"]:
                at_pos = m["pos_group"] == pos
                pool = m.loc[at_pos & m["qualified"], col].to_numpy(dtype=float)
                m.loc[at_pos, f"pct_{col}"] = percentile_in(m.loc[at_pos, col].to_numpy(dtype=float), pool, direction)
    m["war_rank_pos"] = m.groupby("pos_group")["war"].rank(ascending=False, method="min")
    return m


# ---------------------------------------------------------------------------
# 3. NHL EDGE (skating and shot speed). Percentiles come from the NHL.
# ---------------------------------------------------------------------------
def fetch_edge(player_id, season="20252026", game_type=2):
    """Return {metric: (value, percentile 0-100)} from NHL EDGE, or None if unavailable."""
    url = f"https://api-web.nhle.com/v1/edge/skater-detail/{player_id}/{season}/{game_type}"
    # The NHL's server rejects Python's default identity (HTTP 403), so identify as a normal browser
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            data = json.load(resp)
    except Exception as err:
        print("NHL EDGE not available:", err)
        return None
    return parse_edge(data)


def parse_edge(data):
    def get(*keys):
        node = data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return np.nan
            node = node[k]
        return float(node) if node is not None else np.nan

    return {
        "top_speed":  (get("skatingSpeed", "speedMax", "imperial"), 100 * get("skatingSpeed", "speedMax", "percentile")),
        "bursts_20":  (get("skatingSpeed", "burstsOver20", "value"), 100 * get("skatingSpeed", "burstsOver20", "percentile")),
        "shot_speed": (get("topShotSpeed", "imperial"), 100 * get("topShotSpeed", "percentile")),
        "oz_pct_ev":  (get("zoneTimeDetails", "offensiveZoneEvPctg"), 100 * get("zoneTimeDetails", "offensiveZoneEvPercentile")),
    }


def add_bio(cards, bio_table):
    """Merge age and draft details (edge_data.build_bio_table) into the card table."""
    from edge_data import draft_text
    m = cards.drop(columns=[c for c in bio_table.columns if c != "playerId" and c in cards.columns])
    m = m.merge(bio_table, on="playerId", how="left")
    m["draft_text"] = m.apply(draft_text, axis=1)
    return m


def add_edge(cards, edge_table):
    """Merge saved EDGE data (edge_data.build_table) into the card table and add our own percentiles,
    computed the same way as every other row: vs. qualified players at the same position."""
    m = cards.drop(columns=[c for c in edge_table.columns if c != "playerId" and c in cards.columns])
    m = m.merge(edge_table, on="playerId", how="left")
    m["has_edge"] = m["has_edge"].fillna(False).astype(bool)
    m["bursts_20_60"] = rate60(m["bursts_20"], m["all_toi"])     # per 60 so short seasons compare fairly
    for key, _, _ in EDGE_ROWS:
        m[f"pct_{key}"] = np.nan
        for pos in ["F", "D"]:
            at_pos = m["pos_group"] == pos
            pool = m.loc[at_pos & m["qualified"] & m["has_edge"], key].to_numpy(dtype=float)
            m.loc[at_pos, f"pct_{key}"] = percentile_in(m.loc[at_pos, key].to_numpy(dtype=float), pool, "high")
    return m


# ---------------------------------------------------------------------------
# 4. Draw the card
# ---------------------------------------------------------------------------
SURFACE, INK, MUTED, TRACK = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE_GRAY = "#8f8e88"
DIVERGING = LinearSegmentedColormap.from_list("savant", ["#2a78d6", "#f0efec", "#e34948"])  # poor -> average -> great

# Shot map: one hue that means nothing else on the card (red/blue are better/worse)
SHOT_CMAP = LinearSegmentedColormap.from_list("shots", [(0, SURFACE), (0.35, "#f8e3b5"), (0.70, "#eeb04a"), (1, "#b8740e")])

# Shot map grid: offensive zone with every shot folded onto one end (net at x = 89), 5 ft cells.
# y > 0 is the shooter's left, seen from the point looking at the net (checked in notebook 12).
SHOT_X = (25, 100)                   # blue line to end boards, feet from center ice
SHOT_Y = (-42.5, 42.5)               # board to board
SHOT_CELL = 5
SHOT_NX = int((SHOT_X[1] - SHOT_X[0]) / SHOT_CELL)     # 15 cells deep
SHOT_NY = int((SHOT_Y[1] - SHOT_Y[0]) / SHOT_CELL)     # 17 cells wide
SMALL_SHOT_SAMPLE = 50               # unblocked shots; below this the map gets a small-sample note
MAP_SCALE = 0.04                     # inches per foot on the card: 75 ft deep = 3.0 in, 85 ft wide = 3.4 in


def shot_cell(fx, fy):
    """Grid cell (gx, gy) for folded shot coordinates, clipped to the offensive-zone grid."""
    gx = np.clip(np.floor((np.asarray(fx, dtype=float) - SHOT_X[0]) / SHOT_CELL), 0, SHOT_NX - 1).astype(int)
    gy = np.clip(np.floor((np.asarray(fy, dtype=float) - SHOT_Y[0]) / SHOT_CELL), 0, SHOT_NY - 1).astype(int)
    return gx, gy


def _smooth(counts, sigma=1.0):
    """Light Gaussian blur (sigma in cells), numpy only: across, then down."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, counts)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def _shot_map(ax, y, grid, player, alpha, heading):
    """Shot map section: offensive zone with the net at the top, seen from the point (screen left = shooter's left).
    grid = this player-season's rows (gx, gy, shots, goals). Returns the y of the section's bottom."""
    y = heading("Shot map  (all situations, unblocked shots)", y)
    top = y - 0.05
    x_left = 0.04

    def to_ax(fx, fy):
        """Rink feet (folded) -> card axes: x across the card (0-1 = 8.5 in), y in inches. Screen left = +fy."""
        h = -np.asarray(fy, dtype=float)
        return x_left + (h - SHOT_Y[0]) * MAP_SCALE / 8.5, top - (SHOT_X[1] - np.asarray(fx, dtype=float)) * MAP_SCALE

    def line(fx, fy, linewidth=0.8):
        ax.plot(*to_ax(fx, fy), color=MUTED, linewidth=linewidth, zorder=3, alpha=0.8 * alpha)

    def circle(cx, cy, r, start=0, end=2 * np.pi):
        t = np.linspace(start, end, 80)
        line(cx + r * np.cos(t), cy + r * np.sin(t))

    # Density: his own shots, lightly smoothed, scaled to his busiest area
    counts = np.zeros((SHOT_NX, SHOT_NY))
    if len(grid):
        np.add.at(counts, (grid["gx"].to_numpy(int), grid["gy"].to_numpy(int)), grid["shots"].to_numpy(float))
    dens = _smooth(counts)
    if dens.max() > 0:
        dens = dens / dens.max()
    left, bottom = to_ax(SHOT_X[0], SHOT_Y[1])          # blue line, shooter's left boards
    right, top_ = to_ax(SHOT_X[1], SHOT_Y[0])           # end boards, shooter's right boards
    image = ax.imshow(dens[:, ::-1], cmap=SHOT_CMAP, vmin=0, vmax=1, origin="lower", aspect="auto",
                      interpolation="bilinear", extent=[left, right, bottom, top_], alpha=alpha, zorder=2)

    # Rink: boards with rounded corners behind the net, clipping the density to the ice
    t = np.linspace(0, np.pi / 2, 30)
    corner = 28
    bx = np.concatenate([[SHOT_X[0]], 72 + corner * np.sin(t), 72 + corner * np.cos(t), [SHOT_X[0]]])
    by = np.concatenate([[SHOT_Y[1]], 14.5 + corner * np.cos(t), -14.5 - corner * np.sin(t), [SHOT_Y[0]]])
    board_x, board_y = to_ax(bx, by)
    image.set_clip_path(plt.Polygon(np.column_stack([board_x, board_y]), closed=True, transform=ax.transData))
    line(bx, by)
    line([SHOT_X[0], SHOT_X[0]], [SHOT_Y[0], SHOT_Y[1]], linewidth=2)                # blue line
    goal_y = 14.5 + np.sqrt(corner ** 2 - (89 - 72) ** 2)                              # where the goal line meets the boards
    line([89, 89], [-goal_y, goal_y])                                                   # goal line
    line([89, 92.3, 92.3, 89], [-3, -3, 3, 3])                                          # net
    circle(89, 0, 6, np.pi / 2, 3 * np.pi / 2)                                          # crease
    for cy in (22, -22):
        circle(69, cy, 15)                                                              # faceoff circles
        ax.scatter(*to_ax(69, cy), s=8, color=MUTED, zorder=3, alpha=alpha)             # faceoff dots

    # Goals: dots at cell centers, sized by count
    scored = grid[grid["goals"] > 0] if len(grid) else grid
    if len(scored):
        gx, gy = to_ax(SHOT_X[0] + SHOT_CELL * (scored["gx"] + 0.5), SHOT_Y[0] + SHOT_CELL * (scored["gy"] + 0.5))
        ax.scatter(gx, gy, s=10 + 14 * scored["goals"].clip(upper=6), color=INK, edgecolor="#ffffff",
                   linewidth=0.8, zorder=4, alpha=alpha)

    # Panel: totals (all zones) and key
    shots = player.get("unblocked_shots", np.nan)
    goals = player.get("unblocked_goals", np.nan)
    if pd.isna(shots):
        shots, goals = (grid["shots"].sum(), grid["goals"].sum()) if len(grid) else (0, 0)
    oz = int(grid["shots"].sum()) if len(grid) else 0
    px, py = 0.52, top - 0.15
    for label, value in [("Unblocked shots", f"{shots:.0f}"), ("Goals", f"{goals:.0f}"),
                         ("Shooting %  (goals / unblocked shots)", f"{goals / shots:.1%}" if shots else "–")]:
        ax.text(px, py, label, fontsize=9, color=MUTED, va="center")
        ax.text(px, py - 0.3, value, fontsize=17, fontweight="bold", color=INK, va="center", alpha=alpha)
        py -= 0.75
    notes = [f"Offensive zone: {oz} of {shots:.0f} shots",
             "Color: where he shoots from (his own shots, smoothed)",
             "Dots: goals (bigger = more)",
             "Net at the top, as seen from the point"]
    if shots < SMALL_SHOT_SAMPLE:
        notes.append(f"Small sample: under {SMALL_SHOT_SAMPLE} unblocked shots")
    for note in notes:
        ax.text(px, py, note, fontsize=8, color=MUTED, va="center")
        py -= 0.22
    return min(bottom, py + 0.22)                       # lowest of the rink and the last note


def _row(ax, y, label, value_text, pct, color, alpha):
    x0, x1 = 0.36, 0.82
    ax.text(0.04, y, label, va="center", ha="left", fontsize=10, color=INK, alpha=alpha)
    ax.plot([x0, x1], [y, y], color=TRACK, linewidth=6, solid_capstyle="round", zorder=1)
    if not np.isnan(pct):
        xe = x0 + (x1 - x0) * pct / 100
        ax.plot([x0, xe], [y, y], color=color, linewidth=6, solid_capstyle="round", zorder=2, alpha=alpha)
        ax.scatter([xe], [y], s=330, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, alpha=alpha)
        dark = pct >= 80 or pct <= 20 or color == STYLE_GRAY
        text_color = ("white" if dark else INK) if alpha == 1 else MUTED   # faint cards: keep numbers readable
        ax.text(xe, y, f"{pct:.0f}", va="center", ha="center", fontsize=7 if pct >= 100 else 8, fontweight="bold",
                color=text_color, zorder=4)
    ax.text(0.96, y, value_text, va="center", ha="right", fontsize=10, color=MUTED, alpha=alpha)


def draw_card(player, season_label, edge=None, path=None, shot_grid=None):
    """player = one row (Series) from add_percentiles. edge = output of fetch_edge, or None.
    shot_grid = this player-season's shot-map rows (gx, gy, shots, goals), or None for no shot map."""
    alpha = 1.0 if player["qualified"] else 0.35
    n_edge = sum(key not in HIDDEN_ROWS for key, _, _ in EDGE_ROWS)
    n_rows = sum(len(r) for _, r in SECTIONS) + n_edge + 2               # +2 for the clutch row and spacing
    n_heads = len(SECTIONS) + 2
    height = 2.7 + 0.36 * n_rows + 0.5 * n_heads
    fig = plt.figure(figsize=(8.5, height), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])      # 1 unit of y = 1 inch, so rows keep their spacing
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Header
    y = height - 0.55
    pos_name = "defensemen" if player["pos_group"] == "D" else "forwards"
    ax.text(0.04, y, player["name"], fontsize=22, fontweight="bold", color=INK, va="center")
    ax.text(0.96, y, f"{player['war']:.1f} WAR", fontsize=20, fontweight="bold", color=INK, va="center", ha="right")
    y -= 0.45
    bits = [player["team"], player["position"], season_label]
    if "age" in player.index and not pd.isna(player["age"]):
        bits.append(f"Age {player['age']:.0f}")
    ax.text(0.04, y, "  ·  ".join(bits), fontsize=11, color=MUTED, va="center")
    ax.text(0.96, y, f"#{player['war_rank_pos']:.0f} among {pos_name}  ·  #{player['war_rank']:.0f} overall",
            fontsize=10, color=MUTED, va="center", ha="right")
    if "draft_text" in player.index and isinstance(player["draft_text"], str):
        y -= 0.36
        ax.text(0.04, y, player["draft_text"], fontsize=10, color=MUTED, va="center")
    y -= 0.42
    mins, secs = divmod(round(player["toi_per_gp"] * 60), 60)
    ax.text(0.04, y, f"GP {player['games_played']:.0f}    G {player['goals']:.0f}    A {player['assists']:.0f}    "
                     f"P {player['points']:.0f}    TOI/GP {mins}:{secs:02d}", fontsize=10, color=INK, va="center")
    if not player["qualified"]:
        ax.text(0.96, y, f"Below {MIN_EV_TOI} 5v5 minutes: small sample", fontsize=9, color=MUTED, va="center", ha="right")
    y -= 0.25
    ax.plot([0.04, 0.96], [y, y], color=TRACK, linewidth=1)

    def heading(text, y):
        ax.text(0.04, y, text.upper(), fontsize=9, fontweight="bold", color=MUTED, va="center")
        return y - 0.4

    season = player.get("season", np.nan)
    early = not pd.isna(season) and season < FIRST_CLEAN_TRACKING_SEASON    # tracking-era rows go gray

    y -= 0.45
    for title, rows in SECTIONS:
        y = heading(title, y)
        for col, label, direction, fmt in rows:
            pct = player[f"pct_{col}"]
            era = early and col in ERA_SENSITIVE
            gray = direction in ("style", "context") or era
            color = STYLE_GRAY if gray else (DIVERGING(pct / 100) if not np.isnan(pct) else TRACK)
            if direction == "style":
                label = f"{label}  (style)"
            elif direction == "context":
                label = f"{label}  (team context)"
            elif era:
                label = f"{label}  (pre-2024-25)"
            _row(ax, y, label, fmt.format(player[col]) if not np.isnan(player[col]) else "–", pct, color, alpha)
            y -= 0.36
        y -= 0.14

    y = heading("Skating and shot  (NHL EDGE)", y)
    from_file = edge is None and f"pct_{EDGE_ROWS[0][0]}" in player.index
    for key, label, fmt in EDGE_ROWS:
        if key in HIDDEN_ROWS:
            continue
        if from_file:                      # saved EDGE data with our own percentiles (add_edge)
            val, pct = float(player[key]), float(player[f"pct_{key}"])
        elif edge:                          # live fetch, NHL's percentiles
            val, pct = edge.get(key, (np.nan, np.nan))
        else:
            val, pct = np.nan, np.nan
        color = DIVERGING(pct / 100) if not np.isnan(pct) else TRACK
        _row(ax, y, label, fmt.format(val) if not np.isnan(val) else "–", pct, color, alpha)
        y -= 0.36
    y -= 0.14

    y = heading("Clutch", y)
    ax.text(0.04, y, "Coming in v1.1", fontsize=10, color=MUTED, va="center", style="italic")

    if shot_grid is not None:
        y = _shot_map(ax, y - 0.55, shot_grid, player, alpha, heading)

    y -= 0.6
    source = " (EDGE percentiles from the NHL)" if (edge and not from_file) else ""
    ax.text(0.04, y, f"Percentiles vs. qualified players at the same position{source}.\n"
                     "Red = better, blue = worse, gray = style or team context (not a rating). Data: MoneyPuck.com, NHL EDGE.",
            fontsize=7.5, color=MUTED, va="top")
    bottom = y - 0.5
    ax.set_ylim(bottom, height)
    fig.set_size_inches(8.5, height - bottom)    # trim unused space at the bottom
    if path is not None:
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Slim table for the public app. Only the columns the card draws, built from
#    SECTIONS and EDGE_ROWS so it stays in sync with the card. No raw MoneyPuck stats.
# ---------------------------------------------------------------------------
HEADER_COLS = ["playerId", "name", "team", "position", "pos_group", "age", "draft_text", "qualified",
               "games_played", "goals", "assists", "points", "toi_per_gp", "war_rank", "war_rank_pos"]


SEASON_COLS = ["season", "season_label"]


def app_columns(by_season=False):
    """Columns the public app needs: playerId, header fields, then each card row and its percentile.
    by_season=True adds season and season_label after playerId, for a table with several seasons."""
    cols = ["playerId"] + (SEASON_COLS if by_season else []) + HEADER_COLS[1:]
    for _, rows in SECTIONS:
        for col, _, _, _ in rows:
            cols += [col, f"pct_{col}"]
    for key, _, _ in EDGE_ROWS:
        if key not in HIDDEN_ROWS:
            cols += [key, f"pct_{key}"]
    return list(dict.fromkeys(cols))       # drop duplicates, keep order


def export_app_table(cards, path, by_season=False):
    """Write the slim card table for the public app and return it. cards = full card table."""
    from pathlib import Path
    path = Path(path)
    slim = cards[app_columns(by_season)]
    path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(path, index=False)
    return slim


# ---------------------------------------------------------------------------
# 6. Full card table for one season (the same steps as notebook 08)
#    season = MoneyPuck start year: 2025 = 2025-26. Display season_label, never the integer.
# ---------------------------------------------------------------------------
def season_label(season):
    """MoneyPuck start year -> display label: 2025 -> '2025-26'."""
    return f"{season}-{str(season + 1)[-2:]}"


def build_season_cards(skaters_path, edge_dir, bio_dir):
    """Full card table for one season: WAR v0.4, card metrics, percentiles, saved EDGE data,
    and age and draft (age as of Feb 1 of the season's end year)."""
    from edge_data import build_table, build_bio_table
    war = run_season(skaters_path, V04_PARAMS)
    season = int(war["season"].iloc[0])
    cards = add_percentiles(build_metrics(war, skaters_path))
    cards = add_edge(cards, build_table(edge_dir))
    cards = add_bio(cards, build_bio_table(bio_dir, season + 1))
    cards["season_label"] = season_label(season)
    return cards
