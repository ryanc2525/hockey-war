"""
The public Forecast snapshot: one small folder per weekly forecast entry, app_data/forecast/<evidence date>/,
tracked in git so the live site can show it.

Ryan's rule: derived Forecast outputs (projections, OVR, team odds) and real NHL game stats may be public; raw
MoneyPuck columns (xG, shots, Corsi, on-ice stats, TOI splits, ...) and anything in data/ may not. So every file
here is built only from the private forecast log entry (data/forecast_log/<date>/, read, never changed) and the
rating table (notebook 20), and every column must be on an explicit allowlist. Anything else stops the export.

pandas only (no scipy), so an app page can import the allowlists later.

    import public_forecast as PF
    PF.export_public(DATA / "forecast_log" / "2026-11-15", DATA / "forecast_table_2026-27.csv", ROOT / "app_data" / "forecast")
"""
import io
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. The allowlists (CLAUDE.md rule 4 lists the same names)
# ---------------------------------------------------------------------------
RANGE = ["", "_p10", "_p90"]                     # average, 10th and 90th percentile over 10,000 simulated seasons
PLAYER_COLS = (["playerId", "name", "team", "position", "age", "ovr", "gp", "g", "a"]
               + [f"rest_{s}{q}" for s in "gap" for q in RANGE]
               + [f"season_{s}{q}" for s in "gap" for q in RANGE]
               + ["pre_p60", "upd_p60"])
TEAM_COLS = ["team", "real_gp", "real_points", "points", "points_p10", "points_p90", "playoff_pct", "division_pct",
             "presidents_pct", "team_bonus_pre", "team_bonus_upd"]
META_KEYS = ["evidence_through", "run_date", "backfilled", "sit_out", "trades_applied", "settings", "git_commit",
             "exported", "ovr_note"]
# A second guard: names that sound like raw stats are refused even if someone adds them to an allowlist by mistake
FORBIDDEN_WORDS = ["xg", "xgoals", "corsi", "fenwick", "shot", "on_ice", "onice", "war", "gsax", "toi", "minutes",
                   "hits", "block", "takeaway", "giveaway", "moneypuck"]
FILES = ["players.csv", "teams.csv", "meta.json"]
EXACT = {"float_precision": "round_trip"}        # read numbers back exactly as they were written


def check_public(names, allowed, what):
    """Raise ValueError if any name isn't on the allowlist, or sounds like a raw stat. names: columns or keys."""
    names = list(names)
    extra = [n for n in names if n not in allowed]
    risky = [n for n in names if any(w in n.lower() for w in FORBIDDEN_WORDS)]
    if extra or risky:
        raise ValueError(f"{what}: not allowed in the public snapshot: "
                         + "; ".join(x for x in [f"not on the allowlist {extra}" if extra else "",
                                                 f"raw-stat words {risky}" if risky else ""] if x))
    return True


# ---------------------------------------------------------------------------
# 2. Building the three files from one log entry
# ---------------------------------------------------------------------------
def _real_counts(entry_dir, through):
    """Real games, goals and assists so far per player from the entry's NHL boxscore evidence (a game counts if he
    got ice time). Unlike the entry's own columns, this includes goalies."""
    rg = pd.read_csv(Path(entry_dir) / "evidence" / "real_games.csv", **EXACT)
    rg = rg[(rg["date"] <= through) & (rg["toi"] > 0)]
    return rg.groupby("playerId").agg(gp=("game_id", "size"), g=("goals", "sum"), a=("assists", "sum"))


def build_public(entry_dir, rating_table):
    """(players, teams, meta) for one entry. Reads only; never writes."""
    entry_dir = Path(entry_dir)
    m = json.loads((entry_dir / "metadata.json").read_text())
    p = pd.read_csv(entry_dir / "players.csv", **EXACT)
    t = pd.read_csv(entry_dir / "teams.csv", **EXACT)
    ft = pd.read_csv(rating_table).set_index("playerId")
    real = _real_counts(entry_dir, m["evidence_through"]).reindex(p["playerId"]).fillna(0).astype(int)

    out = pd.DataFrame({"playerId": p["playerId"], "name": p["name"],
                        "team": p["team_now"].fillna(p["team"]),          # sitting out: his roster team
                        "position": p["playerId"].map(ft["position"]), "age": p["playerId"].map(ft["age"]),
                        "ovr": p["playerId"].map(ft["ovr"]),
                        "gp": real["gp"].to_numpy(), "g": real["g"].to_numpy(), "a": real["a"].to_numpy()})
    for s in "gap":
        for q in RANGE:
            out[f"rest_{s}{q}"] = p[f"rest_{s}{q}"]
    for s in "ga":                                    # full season = real + simulated rest (season_lines' rule)
        for q in RANGE:
            out[f"season_{s}{q}"] = out[s] + p[f"rest_{s}{q}"]
    for q in RANGE:
        out[f"season_p{q}"] = p[f"season_p{q}"]
    out["pre_p60"] = p["pre_g60"] + p["pre_a60"]
    out["upd_p60"] = p["upd_g60"] + p["upd_a60"]
    out = out[PLAYER_COLS]

    teams = t[TEAM_COLS].copy()

    names = p.set_index("playerId")["name"]
    sit = [{"playerId": int(s["playerId"]), "name": s["name"]} if isinstance(s, dict)
           else {"playerId": int(s), "name": names.get(int(s), "")} for s in m["sit_out"]]
    st = m["settings"]
    meta = {"evidence_through": m["evidence_through"], "run_date": m["created"], "backfilled": bool(m["backfilled"]),
            "sit_out": sit, "trades_applied": list(m["trades_applied"]),
            "settings": {"version": st["version"], "skater_update": st["skaters"], "team_bonus_update": st["teams"],
                         "goalie_update": st["goalies"], "seasons_simulated": st["n_sims"], "seed": st["scenario_seed"],
                         "guesses_per_player": st["n_guess"]},
            "git_commit": m["git_commit"], "exported": date.today().isoformat(),
            "ovr_note": "OVR is the preseason rating from notebook 20; it doesn't change week to week. "
                        "team_bonus is in normal goals per game."}
    return out, teams, meta


# ---------------------------------------------------------------------------
# 3. The check: the public numbers are the entry's numbers
# ---------------------------------------------------------------------------
def _same(a, b):
    """Bit-for-bit equal, with blanks equal to blanks."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    both_na = a.isna() & b.isna()
    return bool(((a == b) | both_na).all())


def check_matches_entry(players, teams, entry_dir):
    """Raise ValueError unless every public number equals the entry: copied columns bit for bit, built columns
    equal to their formula on the entry's numbers, and the entry's own identity season_p = G + A + rest_p."""
    entry_dir = Path(entry_dir)
    p = pd.read_csv(entry_dir / "players.csv", **EXACT)
    t = pd.read_csv(entry_dir / "teams.csv", **EXACT)
    bad = []
    if list(players["playerId"]) != list(p["playerId"]) or list(teams["team"]) != list(t["team"]):
        bad.append("rows differ from the entry")
    for s in "gap":
        for q in RANGE:
            if not _same(players[f"rest_{s}{q}"], p[f"rest_{s}{q}"]):
                bad.append(f"rest_{s}{q}")
    for q in RANGE:
        if not _same(players[f"season_p{q}"], p[f"season_p{q}"]):
            bad.append(f"season_p{q}")
        for s in "ga":
            if not _same(players[f"season_{s}{q}"], players[s] + p[f"rest_{s}{q}"]):
                bad.append(f"season_{s}{q}")
    if not _same(players["pre_p60"], p["pre_g60"] + p["pre_a60"]) or not _same(players["upd_p60"], p["upd_g60"] + p["upd_a60"]):
        bad.append("points per 60")
    sk = (p["group"] != "G").to_numpy()               # skaters: boxscore counts = the entry's counts
    for mine, theirs in [("gp", "gp"), ("g", "G"), ("a", "A")]:
        if not (players[mine].to_numpy()[sk] == p[theirs].to_numpy()[sk]).all():
            bad.append(f"{mine} vs. the entry's {theirs}")
    gap = float((p["G"] + p["A"] + p["rest_p"] - p["season_p"]).abs().max())
    if gap > 1e-12:
        bad.append(f"season_p identity off by {gap:.1e}")
    for c in TEAM_COLS:
        if not _same(teams[c], t[c]):
            bad.append(f"teams {c}")
    if bad:
        raise ValueError("The public snapshot doesn't match the entry: " + ", ".join(bad))
    return True


# ---------------------------------------------------------------------------
# 4. The export: checked first, written once, never overwritten
# ---------------------------------------------------------------------------
def export_public(entry_dir, rating_table, out_root):
    """
    Build the snapshot for one entry, check it (allowlist, then the written-and-read-back numbers against the
    entry), and save out_root/<evidence date>/ (players.csv, teams.csv, meta.json). Everything is checked in memory
    before a single file is written, then written to a temporary folder and renamed in one step.
    Refuses (returns None) if the folder exists. Raises, writing nothing, if a check fails.
    """
    entry_dir, out_root = Path(entry_dir), Path(out_root)
    through = json.loads((entry_dir / "metadata.json").read_text())["evidence_through"]
    final = out_root / through
    if final.exists():
        print(f"The public snapshot for {through} already exists ({final}); nothing was written.")
        return None
    players, teams, meta = build_public(entry_dir, rating_table)
    check_public(players.columns, PLAYER_COLS, "players.csv")
    check_public(teams.columns, TEAM_COLS, "teams.csv")
    check_public(meta.keys(), META_KEYS, "meta.json")
    text = {"players.csv": players.to_csv(index=False), "teams.csv": teams.to_csv(index=False),
            "meta.json": json.dumps(meta, indent=1) + "\n"}
    back_p = pd.read_csv(io.StringIO(text["players.csv"]), **EXACT)    # exactly what the files will hold
    back_t = pd.read_csv(io.StringIO(text["teams.csv"]), **EXACT)
    check_public(back_p.columns, PLAYER_COLS, "players.csv")
    check_public(back_t.columns, TEAM_COLS, "teams.csv")
    check_matches_entry(back_p, back_t, entry_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f".tmp_{through}_{os.getpid()}"
    if tmp.exists():
        raise FileExistsError(f"{tmp} is left over from an earlier run; look at it before exporting again")
    tmp.mkdir()
    for name in FILES:
        (tmp / name).write_text(text[name])
    os.rename(tmp, final)
    kb = {name: (final / name).stat().st_size / 1000 for name in FILES}
    print(f"Saved the public snapshot {final}: " + ", ".join(f"{n} {v:.0f} KB" for n, v in kb.items())
          + f" ({sum(kb.values()):.0f} KB in all; {len(players):,} players, {len(teams)} teams)")
    return final


def snapshots(root):
    """Saved public snapshot folders (evidence dates), oldest first."""
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.exists() else []
