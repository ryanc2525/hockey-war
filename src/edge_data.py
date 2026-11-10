"""
NHL EDGE data: download once per season, save locally, read from the file after that.

    from edge_data import download_season, build_table, download_bios, build_bio_table
    download_season(player_ids, "20252026", Path("../data/edge_raw/2025-26"))
    edge = build_table(Path("../data/edge_raw/2025-26"))
    download_bios(player_ids, Path("../data/bio_raw"))        # age and draft position
    bios = build_bio_table(Path("../data/bio_raw"), 2026)
    download_drafts(range(2019, 2027), Path("../data/draft_raw"))   # every pick, one file per draft year
    drafts = build_draft_table(Path("../data/draft_raw"))

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
DRAFT_URL = "https://records.nhl.com/site/api/draft?cayenneExp=draftYear={year}"   # has playerId, unlike /v1/draft/picks

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


# ---------------------------------------------------------------------------
# Draft classes (records.nhl.com)
# ---------------------------------------------------------------------------
def fetch_draft(year, retries=3):
    """Every pick in one draft year."""
    return fetch_url(DRAFT_URL.format(year=year), f"draft {year}", retries)


def download_drafts(years, raw_dir, pause=0.3):
    """Save each draft year as raw_dir/<year>.json. Years already saved are skipped."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for year in years:
        path = raw_dir / f"{year}.json"
        if path.exists():
            continue
        data = fetch_draft(year)
        if data is None:
            failed.append(year)
        else:
            path.write_text(json.dumps(data))
        time.sleep(pause)
    print(f"Draft years saved: {len(list(raw_dir.glob('*.json')))}" + (f"; failed (run again): {failed}" if failed else ""))
    return failed


def build_draft_table(raw_dir):
    """
    One row per pick: draft year, overall pick, round, drafting team, playerId, name, position (C/LW/RW/D/G),
    group (F/D/G), birth date, amateur league and club. Picks without a playerId are dropped (count printed).
    """
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        rows += json.loads(f.read_text()).get("data", [])
    d = pd.DataFrame(rows)
    missing = d["playerId"].isna().sum()
    if missing:
        print(f"{missing} pick(s) without a playerId dropped")
    d = d[d["playerId"].notna()].copy()
    d["playerId"] = d["playerId"].astype(int)
    d["group"] = d["position"].map({"D": "D", "G": "G"}).fillna("F")
    cols = {"draftYear": "draft_year", "overallPickNumber": "pick", "roundNumber": "round", "triCode": "team",
            "playerId": "playerId", "playerName": "name", "position": "position", "group": "group",
            "birthDate": "birth_date", "amateurLeague": "amateur_league", "amateurClubName": "amateur_club"}
    return d[list(cols)].rename(columns=cols).sort_values(["draft_year", "pick"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2026-27 rosters (notebook 22): today's active rosters, fresh profiles, game logs, the schedule
# ---------------------------------------------------------------------------
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team}/current"           # redirects to the current season
GAME_LOG_URL = "https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"


def _stamp(data):
    """Add the download time to a response, so every saved file says when it was true."""
    data = dict(data)
    data["_downloaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return data


def download_rosters(teams, raw_dir, pause=0.5):
    """Today's active roster for each team (raw_dir/<TEAM>.json). Teams already saved are skipped.
    The active roster leaves out injured, suspended and AHL-assigned players."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for team in teams:
        path = raw_dir / f"{team}.json"
        if path.exists():
            continue
        data = fetch_url(ROSTER_URL.format(team=team), f"roster {team}")
        if not data:
            failed.append(team)
        else:
            path.write_text(json.dumps(_stamp(data)))
        time.sleep(pause)
    print(f"Rosters saved: {len(list(raw_dir.glob('*.json')))}" + (f"; failed (run again): {failed}" if failed else ""))
    return failed


def download_current_profiles(player_ids, raw_dir, pause=0.3):
    """Fresh NHL profiles (same landing page as bio_raw), stamped with the download time. Resumable."""
    def fetch(pid):
        data = fetch_bio(pid)
        return None if data is None else _stamp(data)      # None = failed, retried on the next run
    return _download(player_ids, fetch, raw_dir, pause)


def download_game_logs(player_ids, season, raw_dir, pause=0.3):
    """Each player's game log for one regular season (season like "20262027"). Resumable."""
    return _download(player_ids, lambda pid: fetch_url(GAME_LOG_URL.format(player_id=pid, season=season), pid),
                     raw_dir, pause)


def download_schedule(date, path):
    """The league schedule for the week starting `date` (it carries regularSeasonStartDate). Skipped if saved."""
    path = Path(path)
    if not path.exists():
        data = fetch_url(SCHEDULE_URL.format(date=date), "schedule")
        if data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_stamp(data)))
    return json.loads(path.read_text()) if path.exists() else None


STANDINGS_URL = "https://api-web.nhle.com/v1/standings/{date}"


def download_standings(date, path):
    """The league standings as of `date` (YYYY-MM-DD): games played, points, points percentage per team.
    Skipped if already saved, so the NHL is called once per date."""
    path = Path(path)
    if not path.exists():
        data = fetch_url(STANDINGS_URL.format(date=date), "standings")
        if data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_stamp(data)))
    return json.loads(path.read_text()) if path.exists() else None


def standings_table(data):
    """One row per team from a saved standings file: team, games_played, points, point_pct."""
    rows = [{"team": t["teamAbbrev"]["default"], "games_played": t["gamesPlayed"], "points": t["points"],
             "point_pct": t["pointPctg"]} for t in data.get("standings", [])]
    return pd.DataFrame(rows)


CLUB_SCHEDULE_URL = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
MP_TO_NHL = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}     # MoneyPuck's old codes -> NHL's


def nhl_code(team, season):
    """The NHL API's code for a MoneyPuck team code in a season (Phoenix was PHX through 2013-14)."""
    team = MP_TO_NHL.get(team, team)
    return "PHX" if team == "ARI" and season <= 2013 else team


def download_club_schedules(team_seasons, raw_dir, pause=0.3):
    """
    Every game (with final score) of each (team, season) in team_seasons, one file per team-season:
    raw_dir/{season}_{TEAM}.json. season = start year. Files already saved are skipped, so it can be resumed.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [(t, s) for t, s in team_seasons if not (raw_dir / f"{s}_{t}.json").exists()]
    print(f"{len(team_seasons) - len(todo)} already saved, {len(todo)} to download "
          f"(about {len(todo) * (pause + 0.3) / 60:.0f} minutes)")
    failed = []
    for i, (team, season) in enumerate(todo, 1):
        data = fetch_url(CLUB_SCHEDULE_URL.format(team=team, season=f"{season}{season + 1}"), f"{team} {season}")
        if data:
            (raw_dir / f"{season}_{team}.json").write_text(json.dumps(_stamp(data)))
        else:
            failed.append((team, season))
        if i % 50 == 0 or i == len(todo):
            print(f"  {i} / {len(todo)}")
        time.sleep(pause)
    print(f"{len(failed)} failed (run again to retry): {failed[:10]}" if failed else "Done, no failures.")
    return failed


def schedule_games(raw_dir):
    """
    Regular-season games from the saved club schedules, one row per game (each game is in both teams' files):
    game_id, season (start year), date, home, away, home_goals, away_goals, last_period (REG, OT or SO).
    """
    rows = {}
    for f in sorted(Path(raw_dir).glob("*.json")):
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                continue
            rows[g["id"]] = {"game_id": g["id"], "season": int(str(g["season"])[:4]), "date": g["gameDate"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "home_goals": g["homeTeam"]["score"], "away_goals": g["awayTeam"]["score"],
                             "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType")}
    return pd.DataFrame(rows.values()).sort_values(["season", "date", "game_id"]).reset_index(drop=True)


MINOR_LEAGUES ={"AHL", "ECHL"}                     # NHL teams' farm leagues
EUROPE_LEAGUES = {"KHL", "VHL", "MHL", "SHL", "Liiga", "NL", "DEL", "HockeyAllsvenskan", "Czechia", "ICEHL", "Mestis",
                  "SL", "Slovakia", "J20 Nationell", "U20 Nationell", "U20 SM-liiga", "U20 SM-sarja", "Czechia2", "DEL2",
                  "Norway", "Denmark", "EIHL", "Ligue Magnus", "Czech U20", "Slovakia U20"}


RELOCATED = {"ARI": "UTA"}                          # draft rights of a relocated team moved with it


def _fix_text(text):
    """Some club names come back from the API double-encoded ('MalmÃ¶' for 'Malmö'); undo that when it applies."""
    if not isinstance(text, str):
        return text
    for codec in ("cp1252", "latin-1"):              # the two ways the bytes get misread
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def build_current_tables(players_dir, rosters_dir, season=2026):
    """
    Read notebook 22's snapshot. Returns three tables:
    - players: one row per profile (team on his profile, active flag, draft details, career NHL games, download time)
    - rows: his seasonTotals rows for `season` (regular season, every league, in the profile's order)
    - rosters: who is on each team's active roster today, and when that was downloaded
    """
    players, rows = [], []
    for f in sorted(Path(players_dir).glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("playerId"):
            continue
        draft = d.get("draftDetails") or {}
        career = ((d.get("careerTotals") or {}).get("regularSeason") or {})
        players.append({
            "playerId": d["playerId"], "name": f"{d['firstName']['default']} {d['lastName']['default']}",
            "position": d.get("position"), "birth_date": d.get("birthDate"), "is_active": bool(d.get("isActive")),
            "profile_team": d.get("currentTeamAbbrev"), "profile_team_name": (d.get("fullTeamName") or {}).get("default"),
            "draft_year": draft.get("year"), "draft_team": draft.get("teamAbbrev"), "draft_pick": draft.get("overallPick"),
            "nhl_gp_career": career.get("gamesPlayed", 0) or 0, "profile_as_of": d.get("_downloaded")})
        for i, s in enumerate(d.get("seasonTotals", [])):
            if s.get("season") == season * 10001 + 1 and s.get("gameTypeId") == 2:      # 2026 -> 20262027
                rows.append({"playerId": d["playerId"], "order": i, "league": s.get("leagueAbbrev"),
                             "club": _fix_text((s.get("teamName") or {}).get("default")), "gp": s.get("gamesPlayed") or 0})
    rosters = []
    for f in sorted(Path(rosters_dir).glob("*.json")):
        d = json.loads(f.read_text())
        for k in ["forwards", "defensemen", "goalies"]:
            rosters += [{"playerId": p["id"], "roster_team": f.stem, "roster_as_of": d.get("_downloaded")} for p in d.get(k, [])]
    return pd.DataFrame(players), pd.DataFrame(rows), pd.DataFrame(rosters)


def nhl_team_names(players):
    """Full NHL team name -> abbreviation, read from the profiles themselves (e.g. 'Detroit Red Wings' -> 'DET')."""
    p = players.dropna(subset=["profile_team", "profile_team_name"])
    return p.groupby("profile_team_name")["profile_team"].agg(lambda x: x.mode().iloc[0]).to_dict()


def affiliate_table(players, rows, min_players=3, min_share=0.6):
    """
    Each AHL/ECHL club's NHL parent, from the data: the most common NHL team on the profiles of its 2026-27
    players who have one (players on NHL contracts). A club needs min_players such players and a min_share
    majority; otherwise its parent is unknown (fill it in by hand if needed).
    """
    r = rows[rows["league"].isin(MINOR_LEAGUES)].merge(players[["playerId", "profile_team"]], on="playerId")
    out = []
    for (league, club), d in r.groupby(["league", "club"]):
        signed = d.dropna(subset=["profile_team"]).drop_duplicates("playerId")
        counts = signed["profile_team"].value_counts()
        top = counts.index[0] if len(counts) else None
        share = counts.iloc[0] / counts.sum() if len(counts) else 0.0
        ok = len(signed) >= min_players and share >= min_share
        out.append({"league": league, "club": club, "players": d["playerId"].nunique(), "nhl_signed": len(signed),
                    "top_parent": top, "share": share, "parent": top if ok else None,
                    "other_parents": ", ".join(f"{t} {n}" for t, n in counts.iloc[1:].items())})
    return pd.DataFrame(out).sort_values(["league", "club"]).reset_index(drop=True)


def cba_rights(draft_year, league, season=2026):
    """
    Are a draftee's unsigned rights plausibly still held in `season` (2026 = 2026-27), by the league he plays in?
    From memory of CBA Article 8.6 (2012 CBA, extended by the 2020 and 2025 MOUs); approximate, so used as a
    check only:
    - college (NCAA): held while he's in college (up to August 15 after his class graduates)
    - Europe (incl. KHL and European juniors): four years, until June 1 of draft year + 4
    - anything else (CHL, USHL, other junior): two years, until June 1 of draft year + 2
    Returns (held, rule text).
    """
    if pd.isna(draft_year):
        return False, "undrafted"
    y = int(draft_year)
    if league == "NCAA":
        return True, "college: held while in the NCAA"
    if league in EUROPE_LEAGUES:
        return y + 4 > season, f"Europe: until June 1, {y + 4}"
    return y + 2 > season, f"junior/other: until June 1, {y + 2}"


def assign_teams(players, rows, rosters, affiliates, team_names, trades, overrides, opening_night, first_nhl_team):
    """
    Opening-night team and status for every player (notebook 22's rules, in order).

    team_today: on an active roster today -> that team; else the team on his profile (the NHL's record of who
    holds him, signed or not); else his 2026-27 AHL/ECHL club's NHL parent; else none.
    team_opening_night: team_today, then
      (rule 2) a player whose first 2026-27 NHL game was for another team starts there (first_nhl_team),
      (trade log) a logged move dated after opening night is undone (he starts on from_team),
      (overrides) roster_overrides.csv wins.
    status: what kind of member of the organization he is (see the notebook for the list).
    Returns the table plus the trade-log check.
    """
    p = players.merge(rosters, on="playerId", how="left")
    last = rows.sort_values("order").groupby("playerId").tail(1).set_index("playerId")       # latest club this season
    minor = rows[rows["league"].isin(MINOR_LEAGUES)].sort_values("order").groupby("playerId").tail(1).set_index("playerId")
    has_nhl26 = set(rows.loc[rows["league"] == "NHL", "playerId"])
    parent = affiliates.set_index(["league", "club"])["parent"]

    p["league_now"] = p["playerId"].map(last["league"])
    p["club_now"] = p["playerId"].map(last["club"])
    p["minor_parent"] = [parent.get((l, c)) if isinstance(l, str) else None
                         for l, c in zip(p["playerId"].map(minor["league"]), p["playerId"].map(minor["club"]))]

    # team today
    p["team_today"], p["rule"] = None, None
    on_roster = p["roster_team"].notna()
    p.loc[on_roster, "team_today"] = p.loc[on_roster, "roster_team"]
    p.loc[on_roster, "rule"] = "active roster today"
    has_prof = ~on_roster & p["profile_team"].notna()
    p.loc[has_prof, "team_today"] = p.loc[has_prof, "profile_team"]
    p.loc[has_prof, "rule"] = "team on his NHL profile"
    via_minor = ~on_roster & ~has_prof & p["minor_parent"].notna()
    p.loc[via_minor, "team_today"] = p.loc[via_minor, "minor_parent"]
    p.loc[via_minor, "rule"] = "AHL/ECHL club's NHL parent"

    # status
    in_minor_now = p["league_now"].isin(MINOR_LEAGUES)
    nhl_vet = (p["nhl_gp_career"] > 0) | p["playerId"].isin(has_nhl26)
    st = np.select(
        [on_roster,
         has_prof & in_minor_now,
         has_prof & p["league_now"].notna() & ~p["league_now"].isin(MINOR_LEAGUES | {"NHL"}),
         has_prof & nhl_vet,
         has_prof,
         via_minor],
        ["NHL roster",
         "AHL/ECHL (NHL contract)",
         "rights held, playing elsewhere",
         "NHL org, off active roster",
         "rights held, no 2026-27 games yet",
         "AHL/ECHL (AHL contract)"],
        default="")
    p["status"] = st

    # Nobody claims him: CBA check for unsigned draftees (never played in the NHL), otherwise not with an NHL organization
    rest = (p["status"] == "") & (p["nhl_gp_career"] == 0)
    cba = [cba_rights(y, l) for y, l in zip(p["draft_year"], p["league_now"])]
    p["cba_held"] = [c[0] for c in cba]
    p["cba_rule"] = [c[1] for c in cba]
    held = rest & p["cba_held"] & p["league_now"].notna()
    p.loc[held, "team_today"] = p.loc[held, "draft_team"].replace(RELOCATED)
    p.loc[held, "status"] = "rights held, profile blank (CBA)"
    p.loc[held, "rule"] = "draft team, CBA rights still held (profile blank)"
    expired = rest & ~held & p["draft_year"].notna() & p["league_now"].notna()          # (rest: never played in the NHL)
    p.loc[expired, "status"] = "rights likely expired"
    p.loc[expired, "rule"] = "CBA rights expired (profile blank)"
    none = p["status"] == ""
    p.loc[none, "status"] = "not with an NHL organization"
    p.loc[none, "rule"] = np.where(p.loc[none, "league_now"].notna(), "no NHL team, playing elsewhere", "no 2026-27 games, no NHL team")

    # Opening night: undo in-season moves with games, then logged trades, then overrides
    p["team_opening_night"] = p["team_today"]
    moved = p["playerId"].map(first_nhl_team)
    rule2 = moved.notna() & (moved != p["team_today"])
    p.loc[rule2, "team_opening_night"] = moved[rule2]
    p.loc[rule2, "rule"] = "first 2026-27 NHL team (in-season move)"

    checks = []
    for _, t in trades.iterrows():
        i = p.index[p["playerId"] == t["playerId"]]
        today = p.loc[i, "team_today"].iloc[0] if len(i) else None
        checks.append({"trade_id": t["trade_id"], "name": t["name"], "from_team": t["from_team"], "to_team": t["to_team"],
                       "team_today": today, "matches": today == t["to_team"]})
        if len(i) and pd.Timestamp(t["date"]) > pd.Timestamp(opening_night):
            p.loc[i, "team_opening_night"] = t["from_team"]
            p.loc[i, "rule"] = f"trade log {t['trade_id']} undone ({t['date']})"

    for _, o in overrides.iterrows():
        i = p.index[p["playerId"] == o["playerId"]]
        if pd.notna(o.get("team")):
            p.loc[i, "team_opening_night"] = o["team"]
        if pd.notna(o.get("status")):
            p.loc[i, "status"] = o["status"]
        p.loc[i, "rule"] = "override: " + str(o.get("note", ""))
    return p, pd.DataFrame(checks)


def first_nhl_teams(game_log_dir, season_start=None):
    """Each player's team in his first 2026-27 NHL regular-season game, with the date of his first game for
    each later team (from the saved game logs)."""
    first, moves = {}, []
    for f in sorted(Path(game_log_dir).glob("*.json")):
        games = sorted(json.loads(f.read_text()).get("gameLog", []), key=lambda g: g["gameDate"])
        if not games:
            continue
        pid = int(f.stem)
        first[pid] = games[0]["teamAbbrev"]
        seen = [games[0]["teamAbbrev"]]
        for g in games[1:]:
            if g["teamAbbrev"] != seen[-1]:
                moves.append({"playerId": pid, "from_team": seen[-1], "to_team": g["teamAbbrev"], "first_game": g["gameDate"]})
                seen.append(g["teamAbbrev"])
    return pd.Series(first, dtype=object), pd.DataFrame(moves, columns=["playerId", "from_team", "to_team", "first_game"])


# ---------------------------------------------------------------------------
# The current season's full schedule (notebook 26): every game, played or not
# ---------------------------------------------------------------------------
def download_season_schedule(teams, raw_dir, season=2026, refresh=False, pause=0.3):
    """
    Each team's full regular-season schedule for one season (raw_dir/{TEAM}.json): dates, home/away, and the
    final score of games already played. refresh=False skips teams already saved (no network call);
    refresh=True downloads them again to pick up new results.
    Keep raw_dir separate from the past seasons' folder: schedule_games reads every file in its folder.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in teams if refresh or not (raw_dir / f"{t}.json").exists()]
    failed = []
    for team in todo:
        data = fetch_url(CLUB_SCHEDULE_URL.format(team=team, season=f"{season}{season + 1}"), f"{team} {season}")
        if data:
            (raw_dir / f"{team}.json").write_text(json.dumps(_stamp(data)))
        else:
            failed.append(team)
        time.sleep(pause)
    print(f"{len(teams) - len(todo)} already saved, {len(todo)} downloaded"
          + (f"; failed (run again): {failed}" if failed else ""))
    return failed


def season_schedule(raw_dir):
    """
    Every regular-season game in the saved files of download_season_schedule, one row per game (each game is in
    both teams' files): game_id, date, home, away, played (bool), home_goals, away_goals (NaN if not played),
    last_period (REG, OT or SO; None if not played). Also returns when the files were downloaded (the oldest stamp).
    """
    rows, stamps = {}, []
    for f in sorted(Path(raw_dir).glob("*.json")):
        data = json.loads(f.read_text())
        stamps.append(data.get("_downloaded"))
        for g in data.get("games", []):
            if g.get("gameType") != 2:
                continue
            played = g.get("gameState") in ("OFF", "FINAL")
            rows[g["id"]] = {"game_id": g["id"], "date": g["gameDate"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"], "played": played,
                             "home_goals": g["homeTeam"].get("score") if played else np.nan,
                             "away_goals": g["awayTeam"].get("score") if played else np.nan,
                             "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType") if played else None}
    games = pd.DataFrame(rows.values()).sort_values(["date", "game_id"]).reset_index(drop=True)
    return games, min(s for s in stamps if s)


def standings_detail(data):
    """One row per team from a saved standings file, with what the playoff race needs: division, conference,
    games played, wins, OT losses, points, regulation wins (RW) and regulation + overtime wins (ROW)."""
    rows = [{"team": t["teamAbbrev"]["default"], "division": t["divisionName"], "conference": t["conferenceName"],
             "gp": t["gamesPlayed"], "wins": t["wins"], "otl": t["otLosses"], "points": t["points"],
             "rw": t["regulationWins"], "row": t["regulationPlusOtWins"]} for t in data.get("standings", [])]
    return pd.DataFrame(rows).sort_values(["conference", "division", "team"]).reset_index(drop=True)
