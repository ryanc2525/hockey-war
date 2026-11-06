"""
OVR ratings (Forecast Phase 1, notebook 20): an EA NHL-style overall rating from projected WAR/60.

How it works, in two steps:
1. war_std: projected WAR/60 x the minutes of a full-time role at his position (STD_MINUTES), so forwards,
   defensemen and goalies are compared on what they'd be worth to a team over a full-time season.
   WAR_std is on one scale for every position: 2 WAR_std from a goalie is worth the same as 2 from a forward.
2. ovr: WAR_std -> OVR through fixed anchors, straight lines between them. Skaters and goalies have their
   own anchors (below). Replacement level (WAR_std = 0) is 70 for both, and the median NHL regular at
   his position is about 80.

Why goalies have their own anchors: the simulator uses WAR, not OVR, so OVR is only a display. Goalie
projections are far less certain than skaters' (about 87% of a surprise goalie season is luck), so the best
goalie is shown at 95, just below the best skaters (98). Compare goalies with skaters on WAR_std, not on OVR.

THE SCALE IS LOCKED. The constants below were set from 2025-26 data and the 2026-27 projection (notebook 20),
refrozen once when goalies got their own anchors, and are now fixed for the season simulator. The simulator
rates every simulated season with these same constants, so ratings can rise and fall league-wide in a
multi-year sim (a strong generation can produce more 90s). Never re-percentile each season: that would force
the same number of 90s every year and hide real change.

Usage:
    from ratings import war_std, ovr, ovr_exact, group_of
    ovr(0.175, "F")            # McDavid's 2026-27 projection -> 98
"""
import numpy as np
import pandas as pd

# Minutes per season in a full-time role (notebook 20, from 2025-26):
# forwards: each team's top 6 by TOI average 17.84 min/game x 82 games
# defensemen: each team's top 4 average 21.35 min/game x 82 games
# goalies: each team's starter (most games) averages 49.78 games x 60 minutes
STD_MINUTES = {"F": 1463, "D": 1751, "G": 2987}

# Skaters (forwards and defensemen): WAR_std -> OVR anchors (notebook 20, skaters only). Straight lines between
# them; below the first and above the last the rating stays at 40 and 99.
ANCHOR_WAR = [-2.55, 0.00, 0.85, 4.25, 4.44]
ANCHOR_OVR = [40, 70, 80, 98, 99]
#  -2.55 -> 40   floor (the replacement-to-median slope, about 11.8 OVR per WAR, continued down)
#   0.00 -> 70   replacement level
#   0.85 -> 80   median regular skater (500+ minutes in 2025-26)
#   4.25 -> 98   the top projected skater for 2026-27 (Connor McDavid, 4.27)
#   4.44 -> 99   cap (the median-to-top slope, about 5.3 OVR per WAR, continued up)

# Goalies: WAR_std -> OVR anchors (notebook 20, goalies only), built by the same rule.
GOALIE_ANCHOR_WAR = [-6.90, 0.00, 2.30, 4.25, 4.38]
GOALIE_ANCHOR_OVR = [40, 70, 80, 95, 96]
#  -6.90 -> 40   floor (the replacement-to-median slope, about 4.3 OVR per WAR, continued down)
#   0.00 -> 70   replacement level
#   2.30 -> 80   median regular goalie (500+ minutes in 2025-26)
#   4.25 -> 95   the top projected goalie for 2026-27 (Logan Thompson, 4.29)
#   4.38 -> 96   cap (the median-to-top slope, about 7.7 OVR per WAR, continued up)


def group_of(position):
    """Position (C, L, R, LW, RW, D, G) -> group (F, D, G). Works on one value or a pandas Series."""
    def one(p):
        return "G" if p == "G" else "D" if p == "D" else "F"
    if isinstance(position, pd.Series):
        return position.map(one)
    return one(position)


def war_std(war_60, group):
    """
    WAR in a full-time role: WAR/60 x STD_MINUTES[group] / 60.
    For skaters that's WAR per 82 games at full-time minutes; for goalies, per starter's season (about 50 games).
    Works on single values or pandas Series (group can be one value or a Series of the same length).
    """
    minutes = group.map(STD_MINUTES) if isinstance(group, pd.Series) else STD_MINUTES[group]
    return war_60 * minutes / 60


def ovr_exact(war_60, group):
    """OVR before rounding (for the simulator's math). Same inputs as war_std. Goalies use the goalie anchors."""
    w = war_std(war_60, group)
    w_arr = np.asarray(w, dtype=float)
    skater = np.interp(w_arr, ANCHOR_WAR, ANCHOR_OVR)
    goalie = np.interp(w_arr, GOALIE_ANCHOR_WAR, GOALIE_ANCHOR_OVR)
    is_goalie = np.asarray(group == "G") if isinstance(group, pd.Series) else group == "G"
    out = np.where(is_goalie, goalie, skater)
    if isinstance(w, pd.Series):
        return pd.Series(out, index=w.index)
    return float(out)


def ovr(war_60, group):
    """OVR as a whole number (halves round up). Same inputs as war_std. Missing WAR/60 gives a missing OVR."""
    x = ovr_exact(war_60, group)
    if isinstance(x, pd.Series):
        return np.floor(x + 0.5).astype("Int64")
    return int(np.floor(x + 0.5)) if np.isfinite(x) else None
