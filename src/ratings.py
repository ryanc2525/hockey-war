"""
OVR ratings (Forecast Phase 1, notebook 20): an EA NHL-style overall rating from projected WAR/60.

How it works, in two steps:
1. war_std: projected WAR/60 x the minutes of a full-time role at his position (STD_MINUTES), so forwards,
   defensemen and goalies are compared on what they'd be worth to a team over a full-time season.
2. ovr: WAR_std -> OVR through fixed anchors (ANCHOR_WAR -> ANCHOR_OVR), straight lines between them,
   clipped to 40-99. Replacement level (WAR_std = 0) is 70 and the median NHL regular is about 80.

THE SCALE IS FROZEN. The constants below were set once, from 2025-26 data and the 2026-27 projection
(notebook 20). The season simulator rates every simulated season with these same constants, so ratings can
rise and fall league-wide in a multi-year sim (a strong generation can produce more 90s). Never re-percentile
each season: that would force the same number of 90s every year and hide real change.

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

# WAR_std -> OVR anchors (notebook 20). Straight lines between them; below the first and above the last the
# rating stays at 40 and 99.
ANCHOR_WAR = [-2.85, 0.00, 0.95, 4.25, 4.43]
ANCHOR_OVR = [40, 70, 80, 98, 99]
#  -2.85 -> 40   floor (the replacement-to-median slope, about 10.5 OVR per WAR, continued down)
#   0.00 -> 70   replacement level
#   0.95 -> 80   median NHL regular (500+ minutes in 2025-26)
#   4.25 -> 98   the top projected player for 2026-27 (Connor McDavid, 4.27)
#   4.43 -> 99   cap (the median-to-top slope, about 5.5 OVR per WAR, continued up)


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
    """OVR before rounding (for the simulator's math). Same inputs as war_std."""
    w = war_std(war_60, group)
    out = np.interp(np.asarray(w, dtype=float), ANCHOR_WAR, ANCHOR_OVR)
    if isinstance(w, pd.Series):
        return pd.Series(out, index=w.index)
    return float(out)


def ovr(war_60, group):
    """OVR as a whole number (halves round up). Same inputs as war_std. Missing WAR/60 gives a missing OVR."""
    x = ovr_exact(war_60, group)
    if isinstance(x, pd.Series):
        return np.floor(x + 0.5).astype("Int64")
    return int(np.floor(x + 0.5)) if np.isfinite(x) else None
