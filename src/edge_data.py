"""
NHL EDGE data: download once per season, save locally, read from the file after that.

    from edge_data import download_season, build_table, download_bios, build_bio_table
    download_season(player_ids, "20252026", Path("../data/edge_raw/2025-26"))
    edge = build_table(Path("../data/edge_raw/2025-26"))
    download_bios(player_ids, Path("../data/bio_raw"))        # age and draft position
    bios = build_bio_table(Path("../data/bio_raw"), 2026)

Each player's raw response is saved as its own JSON file, so the download can be
stopped and restarted without losing progress, and new fields can be pulled out
later without downloading again.

Source: NHL EDGE (api-web.nhle.com), unofficial public API. Regular season only.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://api-web.nhle.com/v1/edge/skater-detail/{player_id}/{season}/2"
BIO_URL = "https://api-web.nhle.com/v1/player/{player_id}/landing"

# The NHL's server rejects Python's default identity (HTTP 403), so identify as a normal browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}

# Fields pulled out of each response: (column name, path inside the JSON)
FIELDS = {
    "top_speed":        ("skatingSpeed", "speedMax", "imperial"),        # mph
    "bursts_20":        ("skatingSpeed", "burstsOver20", "value"),       # count, season
    "shot_speed":       ("topShotSpeed", "imperial"),                    # mph
    "distance_total":   ("totalDistanceSkated", "imperial"),             # miles, season
    "distance_max_game": ("distanceMaxGame", "imperial"),                # miles, best game
    "oz_pct":           ("zoneTimeDetails", "offensiveZonePctg"),        # share of ice time, all situations
    "oz_pct_ev":        ("zoneTimeDetails", "offensiveZoneEvPctg"),      # share of ice time, even strength
    # The NHL's own percentiles (0-1), kept for comparison with ours
    "nhl_pct_top_speed":  ("skatingSpeed", "speedMax", "percentile"),
    "nhl_pct_bursts_20":  ("skatingSpeed", "burstsOver20", "percentile"),
    "nhl_pct_shot_speed": ("topShotSpeed", "percentile"),
    "nhl_pct_oz_pct_ev":  ("zoneTimeDetails", "offensiveZoneEvPercentile"),
}


def fetch_url(url, label, retries=3):
    """Return the parsed response, {} if the NHL has nothing for this player (404), or None on repeated failure."""
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return {}
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))              # server busy: wait, then retry
                continue
            print(f"  {label}: HTTP {err.code}")
            return None
        except Exception as err:
            print(f"  {label}: {err}")
            time.sleep(3)
    return None


def fetch_json(player_id, season, retries=3):
    """One player's EDGE data for one season."""
    return fetch_url(BASE_URL.format(player_id=player_id, season=season), player_id, retries)


def fetch_bio(player_id, retries=3):
    """One player's NHL profile (birth date, draft details, size)."""
    return fetch_url(BIO_URL.format(player_id=player_id), player_id, retries)


def _download(player_ids, fetch, raw_dir, pause):
    """Save fetch(player_id) for every player into raw_dir. Players already saved are skipped."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [pid for pid in player_ids if not (raw_dir / f"{pid}.json").exists()]
    print(f"{len(player_ids) - len(todo)} already saved, {len(todo)} to download "
          f"(about {len(todo) * (pause + 0.3) / 60:.0f} minutes)")

    failed = []
    for i, pid in enumerate(todo, 1):
        data = fetch(pid)
        if data is None:
            failed.append(pid)
        else:
            (raw_dir / f"{pid}.json").write_text(json.dumps(data))
        if i % 50 == 0 or i == len(todo):
            print(f"  {i} / {len(todo)}")
        time.sleep(pause)

    if failed:
        print(f"{len(failed)} failed (not saved). Run again to retry them:", failed[:10])
    else:
        print("Done, no failures.")
    return failed


def download_season(player_ids, season, raw_dir, pause=0.3):
    """Download every player's EDGE data for one season into raw_dir."""
    return _download(player_ids, lambda pid: fetch_json(pid, season), raw_dir, pause)


def download_bios(player_ids, raw_dir, pause=0.3):
    """Download every player's NHL profile into raw_dir (one file per player, any season)."""
    return _download(player_ids, fetch_bio, raw_dir, pause)


def _get(data, path):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node or node[key] is None:
            return np.nan
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return np.nan


def flatten(data):
    """One row of card fields from one player's response."""
    return {col: _get(data, path) for col, path in FIELDS.items()}


def build_table(raw_dir):
    """Read every saved JSON in raw_dir into one table (one row per player)."""
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        data = json.loads(f.read_text())
        row = flatten(data)
        has_edge = any(not np.isnan(row[c]) for c in ("top_speed", "shot_speed", "oz_pct_ev"))
        rows.append({"playerId": int(f.stem), "has_edge": has_edge, **row})
    return pd.DataFrame(rows)


def build_bio_table(raw_dir, season_end_year):
    """Birth date, age, and draft details for every saved profile.
    Age is as of February 1 of the season (hockey's usual "season age"): 2025-26 -> Feb 1, 2026."""
    ref = pd.Timestamp(year=season_end_year, month=2, day=1)
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        d = json.loads(f.read_text())
        draft = d.get("draftDetails") or {}
        birth = pd.to_datetime(d.get("birthDate"), errors="coerce")
        age = np.nan
        if pd.notna(birth):
            age = ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
        rows.append({
            "playerId": int(f.stem),
            "has_bio": bool(d),
            "birth_date": d.get("birthDate"),
            "age": age,
            "draft_year": draft.get("year"),
            "draft_round": draft.get("round"),
            "draft_overall": draft.get("overallPick"),
            "draft_team": draft.get("teamAbbrev"),
        })
    return pd.DataFrame(rows)


def draft_text(row):
    """'2024 Draft, #2 overall (CHI)', 'Undrafted', or None if the profile is missing."""
    if not row.get("has_bio", False):
        return None
    if pd.isna(row.get("draft_year")) or pd.isna(row.get("draft_overall")):
        return "Undrafted"
    team = f" ({row['draft_team']})" if isinstance(row.get("draft_team"), str) else ""
    return f"{int(row['draft_year'])} Draft, #{int(row['draft_overall'])} overall{team}"
