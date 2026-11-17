"""
Season-end OVR progression (notebook 31): after a simulated season, every player's rating for the next season.

The idea in one line: the new rating is the same projection run one season later, with the simulated season added
to each player's history. Nothing here is a new model; every step is the one that made today's ratings:
  - NHL skaters: notebook 18 (forecast.project with the player-specific prior, development.nb18_skater_model);
  - NHL goalies: notebook 21 (forecast.project with the goalie prior, development.nb21_goalie_model);
  - skater prospects and new players: notebooks 19 and 22 (development.project_skater_prospects);
  - goalie prospects and new goalies: notebook 21's prior (development.goalie_prior_war_60);
  - OVR: notebook 20's rule (ratings.py, the locked scale; goalies on their own anchors).
The fitted pieces (priors, centering shifts, the aging curve, the NHLe fill) are built once from real history and
then stay FROZEN, like the OVR scale: refitting them on simulated seasons would let the whole league drift.

    import progression as PG
    E = PG.load_engine()                                   # once, about a minute
    check = PG.reproduce_today(E)                          # today's 2026-27 projections and OVRs, exactly
    s = FR.story(inp, "DET", seed=0)                       # a full-season story (notebook 30)
    t = PG.story_progression(E, s)                         # every player's 2027-28 OVR, with the two parts
    m = PG.progression_many(E, inp, "DET", n_sims=10000)   # expected change, chance of +3 / -3

Seasons are start years (2026 = 2026-27). Pre-NHL inputs (NHLe, pre-NHL save%) stop at 2025-26: the simulator
plays NHL games only, so a player with no NHL minutes gets no new season of any kind. Ryan's rule for prospects with
no NHL minutes: OVR before + the real average change of a prospect year at his age and position (expected
development, not simulated; _develop). The engine's own number stays in engine_ovr / engine_ovr_after.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import development as DV
import forecast as FC
import ratings as RT
from edge_data import build_bio_table, build_draft_table
from nhle import load_profiles, career_totals, EXCLUDED

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIM_SEASON = 2026                                  # the simulated season (2026-27)
NEXT = SIM_SEASON + 1                              # the season the new ratings are for (2027-28)
LAST_REAL = 2025                                   # the last finished real season (pre-NHL data stops here)
MIN_SEASON_MINUTES = {"F": 200, "D": 200, "G": 500}      # real or simulated season minutes that count (experience)
PROSPECT_EXCLUDED = set(EXCLUDED) | {"Prospects Challenge", "International-Jr"}   # notebook 19
AGE_GROUPS = ["under 23", "23-26", "27-30", "31+"]
PROSPECT_AGE_GROUPS = ["18-19", "20-21", "22-23", "24+"]
NHL_SOURCES = {"NHL skater (nb18)", "NHL goalie (nb21)"}
FLAG_NO_MINUTES = "no NHL minutes: aging and regression only (no AHL or junior stats simulated)"
FLAG_PROSPECT = ("prospect, no NHL minutes: expected development (the real average change for his age and position), "
                 "not simulated")
PROSPECT_DEV_SEASONS = [2024, 2025]                # real prospect years the expected development is measured on
FLAG_LEFT = "left the projection: no NHL season in the last three"
FLAG_REAL_ONLY = ("not in a lineup now: his real 2026-27 games so far aren't counted (no real WAR so far); "
                  "aging and regression only")


def age_group(age):
    return pd.cut(pd.Series(age, dtype=float), [-np.inf, 22.5, 26.5, 30.5, np.inf], labels=AGE_GROUPS)


def prospect_age_group(age):
    return pd.cut(pd.Series(age, dtype=float), [-np.inf, 19.5, 21.5, 23.5, np.inf], labels=PROSPECT_AGE_GROUPS)


# ---------------------------------------------------------------------------
# 1. The engine: every input today's ratings were built from, read once
# ---------------------------------------------------------------------------
def load_engine(data_dir=DATA_DIR, label="2026-27"):
    """Everything project_season needs, built once from real history. Reads only; writes nothing."""
    d = Path(data_dir)
    history = FC.history_table(pd.read_csv(d / "war_by_season_2008_2025.csv"), pd.read_csv(d / "goalie_war_2008_2025.csv"))
    bios = build_bio_table(d / "bio_raw", 2026)
    profiles = load_profiles(d / "bio_raw")
    careers = pd.read_csv(d / "career_totals.csv")
    nf, nm = pd.read_csv(d / "nhle_factors.csv"), pd.read_csv(d / "nhle_age_multipliers.csv")
    dev = pd.read_csv(d / "dev_ranges.csv")
    luck = dev[dev["kind"] == "luck_fit"].set_index("group")[["a", "b"]]
    rookie_range = dev[dev["kind"] == "rookie"]

    pros = pd.read_csv(d / f"prospects_{label}.csv")
    newp = pd.read_csv(d / f"new_players_{label}.csv")
    p_prof, p_bios = load_profiles(d / "prospect_raw"), build_bio_table(d / "prospect_raw", 2026)
    n_prof = load_profiles(d / "current_raw" / "players")
    n_bios = build_bio_table(d / "current_raw" / "players", 2026)
    n_prof, n_bios = n_prof[n_prof["playerId"].isin(newp["playerId"])], n_bios[n_bios["playerId"].isin(newp["playerId"])]
    bios_all = pd.concat([bios, p_bios, n_bios]).drop_duplicates("playerId")
    profiles_all = pd.concat([profiles, p_prof, n_prof], ignore_index=True)

    sk = DV.nb18_skater_model(history, bios, profiles, careers, nf, nm)
    sk19 = DV.nb19_models(history, bios, profiles, careers, nf, nm)
    gseasons = DV.load_goalie_seasons(d / "bio_raw", d / "prospect_raw")
    gs_new = DV.load_goalie_seasons(d / "current_raw" / "players")
    gs_new = gs_new[gs_new["playerId"].isin(newp["playerId"])]
    g_model = DV.nb21_goalie_model(history, bios, profiles, gseasons)

    # Who is on the prospect path today (notebook 19's prospects and notebook 22's new players), and with which inputs
    current = FC.current_projections(d, label)
    path = pd.DataFrame({"playerId": current["playerId"], "group": current["group"], "source": current["source"]})
    on_p = ~path["source"].isin(NHL_SOURCES)
    path["new"] = path["playerId"].isin(newp["playerId"])

    # NHLe for players outside careers.csv (prospects, new players), the way notebook 19 builds it
    ids_p = set(path.loc[on_p, "playerId"])
    E = {"history": history, "bios": bios, "bios_all": bios_all, "profiles_all": profiles_all, "careers": careers,
         "nf": nf, "nm": nm, "luck": luck, "rookie_range": rookie_range, "sk": sk, "sk19": sk19, "g_model": g_model,
         "gseasons": gseasons, "gseasons_new": pd.concat([gseasons, gs_new], ignore_index=True),
         "path": path, "current": current, "data_dir": d}
    E["nhle_sk"] = pd.concat([sk["nhle"], _prospect_nhle(E, ids_p, LAST_REAL + 1)]).drop_duplicates("playerId")
    sk_p = path[on_p & (path["group"] != "G") & ~path["new"]]
    _, E["fill25"] = DV.project_skater_prospects(sk_p[["playerId", "group"]], history, bios_all,
                                                 _profiles_for(E, sk_p["playerId"], LAST_REAL + 1), nf, nm, sk19,
                                                 rookie_range, luck)
    rt = pd.read_csv(d / f"ratings_{label}.csv").set_index("playerId")
    E["before"] = pd.DataFrame({"name": rt["name"], "team": rt["team"], "position": rt["position"], "group": rt["group"],
                                "age": rt["age"], "prospect": rt["prospect"], "proj_war_60": rt["proj_war_60"],
                                "rating_war_60": rt["rating_war_60"], "ovr": rt["ovr"].astype(int),
                                "ovr_exact": RT.ovr_exact(rt["rating_war_60"], rt["group"])})
    # Ryan's rule for prospects with no NHL minutes (notebook 31): the real average change of a prospect year
    E["expected_dev"] = expected_development(real_prospect_changes(E, PROSPECT_DEV_SEASONS))
    E["prospect_rule"] = "expected"
    return E


def _profiles_for(E, ids, before):
    """Profile rows of these players, seasons before `before` only (what was known going into that season)."""
    p = E["profiles_all"]
    return p[p["playerId"].isin(set(ids)) & (p["season"] < before)]


def _prospect_nhle(E, ids, before):
    """Notebook 19's NHLe (rule last2, its league renames) for these players, from seasons before `before`."""
    c = career_totals(_profiles_for(E, ids, before))
    if c.empty:
        return pd.DataFrame(columns=["playerId", "nhle_ppg"])
    c["league"] = [DV.prospect_league(l, s) for l, s in zip(c["league"], c["season"])]
    cols = ["name", "position", "group", "birth_date", "age"]
    c = (c.groupby(["playerId", "season", "league"], as_index=False)
         .agg(**{k: (k, "first") for k in cols}, **{k: (k, "sum") for k in ["gp", "goals", "assists", "points"]}))
    return DV.best_nhle(c, E["nf"], E["nm"], rule="last2")


# ---------------------------------------------------------------------------
# 2. One projection for one target season (real history, or real + a simulated season)
# ---------------------------------------------------------------------------
def _nhl_skaters(E, h, target, curve, priors=None):
    cand = DV.candidates(h, [target])
    if priors is None:
        feat = DV.skater_features(cand, h, E["bios_all"], E["profiles_all"], E["nhle_sk"])
        priors = feat[["playerId", "season"]].assign(
            prior_gar_60=DV.apply_prior(feat, E["sk"]["final"]).to_numpy() + feat["group"].map(E["sk"]["shift"]).fillna(0).to_numpy())
    p = FC.project(h[h["group"] != "G"], E["bios_all"], target, DV.SK_WEIGHTS, DV.SK_REG, DV.AGING, curve,
                   priors=priors, prior_reg=DV.NB18_WEIGHT)
    return p, priors


def _nhl_goalies(E, h, target, curve, priors=None):
    gm = E["g_model"]
    if priors is None:
        gs = E["gseasons"][E["gseasons"]["season"] <= LAST_REAL]
        cand = DV.goalie_candidates(h, [target])
        feat = DV.goalie_features(cand, h, E["bios_all"], E["profiles_all"], gs, DV.first_nhl_goalie_season(h, gs),
                                  shrink_shots=DV.NB21_K)
        feat["age"] = feat["age"].clip(lower=gm["age_floor"])
        priors = feat[["playerId", "season"]].assign(
            prior_gar_60=DV.apply_prior(feat.assign(group="G"), gm["final"]).to_numpy() + gm["shift"])
    p = FC.project(h[h["group"] == "G"], E["bios_all"], target, DV.G_WEIGHTS, DV.G_REG, DV.AGING, curve,
                   priors=priors, prior_reg=DV.NB21_WEIGHT)
    return p, priors


def prospect_path(E, players, h, target, through=None):
    """
    Notebook 19/21/22's projection for players with no NHL season in the window. players: playerId, group, new.
    through: the last pre-NHL season used (default: the season before target, never past 2025-26).
    Returns playerId, proj_war_60, d_rank_prior (D skaters).
    """
    through = min(target - 1, LAST_REAL) if through is None else through
    out = []
    sk = players[players["group"] != "G"]
    if len(sk):
        o, _ = DV.project_skater_prospects(sk[["playerId", "group"]], h, E["bios_all"],
                                           _profiles_for(E, sk["playerId"], through + 1), E["nf"], E["nm"], E["sk19"],
                                           E["rookie_range"], E["luck"], fill25=E["fill25"], season=target)
        out.append(o[["playerId", "proj_war_60", "d_rank_prior"]])
    gk = players[players["group"] == "G"]
    for is_new, d in gk.groupby(gk["new"]):
        gs = E["gseasons_new" if is_new else "gseasons"]
        gs = gs[gs["season"] <= through]
        f = DV.goalie_features(d[["playerId"]].assign(season=target), h, E["bios_all"], E["profiles_all"], gs,
                               DV.first_nhl_goalie_season(h, gs), shrink_shots=DV.NB21_K)
        out.append(pd.DataFrame({"playerId": f["playerId"], "proj_war_60": DV.goalie_prior_war_60(f, E["g_model"]).to_numpy()}))
    if not out:
        return pd.DataFrame(columns=["playerId", "proj_war_60", "d_rank_prior"])
    return pd.concat(out, ignore_index=True)


def _rating(table):
    """Notebook 20's rule: D on the prospect path are rated on d_rank_prior; everyone else on the projection."""
    t = table.copy()
    rank_d = (t["path"] == "prospect") & (t["group"] == "D") & t["d_rank_prior"].notna()
    t["rating_war_60"] = np.where(rank_d, t["d_rank_prior"], t["proj_war_60"])
    t["ovr_exact"] = RT.ovr_exact(t["rating_war_60"], t["group"])
    t["ovr"] = RT.ovr(t["rating_war_60"], t["group"])
    return t


def project_season(E, target, extra=None, curve=None, priors=None, players=None, prospects=True):
    """
    Every player's projection and OVR going into `target`, by the same rules as today:
      - NHL path: an NHL season (real or simulated) in the three seasons before target (notebooks 18 and 21);
      - prospect path: today's prospects and new players with no NHL season in that window (notebooks 19/21/22),
        same pre-NHL stats (nothing after 2025-26 is known).
    extra: simulated season rows to add to the history (season_rows). curve / priors: test options (a different
    aging curve; frozen prior rows {"F/D": ..., "G": ...}). players: the pool to report (default today's).
    Returns one row per player: playerId, group, path, proj_war_60, d_rank_prior, rating_war_60, ovr_exact, ovr.
    """
    h = E["history"] if extra is None else pd.concat([E["history"], extra[E["history"].columns]], ignore_index=True)
    curve = E["sk"]["curve"] if curve is None else curve
    priors = priors or {}
    sk, sk_pr = _nhl_skaters(E, h, target, curve, priors.get("skaters"))
    gk, gk_pr = _nhl_goalies(E, h, target, curve, priors.get("goalies"))
    nhl = pd.concat([sk, gk])
    out = pd.DataFrame({"playerId": nhl.index, "group": nhl["group"].to_numpy(), "path": "NHL",
                        "proj_war_60": nhl["proj_war_60"].to_numpy(), "d_rank_prior": np.nan})
    pool = E["path"] if players is None else players
    if prospects:
        todo = pool[~pool["playerId"].isin(out["playerId"]) & ~pool["source"].isin(NHL_SOURCES)]
        pp = prospect_path(E, todo, h, target)
        out = pd.concat([out, pp.merge(todo[["playerId", "group"]], on="playerId").assign(path="prospect")],
                        ignore_index=True)
    out = _rating(out)
    out["engine_ovr"] = out["ovr"]                     # the engine's own number (prospects: no new season)
    out["engine_rating_war_60"] = out["rating_war_60"]
    if prospects and target == NEXT and E.get("prospect_rule") == "expected":
        out = _develop(E, out)
    out.attrs["priors"] = {"skaters": sk_pr, "goalies": gk_pr}
    return out


def war_60_from_ovr(ovr_exact, groups):
    """The inverse of ratings.ovr_exact: the rating WAR/60 that gives this exact OVR (same anchors)."""
    o, groups = np.asarray(ovr_exact, dtype=float), np.asarray(groups)
    ws = np.where(groups == "G", np.interp(o, RT.GOALIE_ANCHOR_OVR, RT.GOALIE_ANCHOR_WAR),
                  np.interp(o, RT.ANCHOR_OVR, RT.ANCHOR_WAR))
    return ws * 60 / np.array([RT.STD_MINUTES[g] for g in groups])


def _develop(E, out):
    """
    Ryan's rule (notebook 31): a prospect with no NHL minutes in the simulated season (still on the prospect path)
    gets his OVR before + the real average change of a prospect year at his age group and position
    (E["expected_dev"], measured on PROSPECT_DEV_SEASONS), flagged as expected development, not simulated.
    His rating WAR/60 follows the OVR (inverse of the locked scale); forwards and goalies are rated on their
    projection, so it moves too; defensemen are rated on d_rank_prior, which moves instead (their projection stays
    the position average, as today). The engine's own number stays in engine_ovr / engine_rating_war_60.
    """
    t = out.copy()
    b = E["before"]
    rows = (t["path"] == "prospect") & t["playerId"].isin(b.index)
    ids = t.loc[rows, "playerId"]
    exp = E["expected_dev"]["expected_change"]
    key = list(zip(b.loc[ids, "group"], prospect_age_group(b.loc[ids, "age"].to_numpy()).astype(str)))
    add = pd.Series([exp.get(k, np.nan) for k in key], index=t.index[rows])
    ok = add.notna()
    idx = add.index[ok]
    target = b.loc[t.loc[idx, "playerId"], "ovr_exact"].to_numpy() + add[ok].to_numpy()
    w = war_60_from_ovr(target, t.loc[idx, "group"].to_numpy())
    is_d = (t.loc[idx, "group"] == "D").to_numpy() & t.loc[idx, "d_rank_prior"].notna().to_numpy()
    t.loc[idx[is_d], "d_rank_prior"] = w[is_d]
    t.loc[idx[~is_d], "proj_war_60"] = w[~is_d]
    t["developed"] = t.index.isin(idx)
    return _rating(t)


def reproduce_today(E):
    """project_season(2026) on real history vs. today's files: the largest WAR/60 gaps and the OVRs that differ."""
    p = project_season(E, SIM_SEASON).set_index("playerId")
    cur = E["current"].set_index("playerId")
    b = E["before"]
    ids = cur.index
    return {"players": len(ids), "missing": sorted(set(ids) - set(p.index)),
            "extra": sorted(set(p.index) - set(ids)),
            "max_proj_gap": float((p["proj_war_60"].reindex(ids) - cur["proj_war_60"]).abs().max()),
            "max_rating_gap": float((p["rating_war_60"].reindex(b.index) - b["rating_war_60"]).abs().max()),
            "ovr_differ": int((p["ovr"].reindex(b.index).astype(int) != b["ovr"]).sum()),
            "by_source": (p["proj_war_60"].reindex(ids) - cur["proj_war_60"]).abs().groupby(cur["source"]).max(),
            "table": p}


# ---------------------------------------------------------------------------
# 3. A simulated season as history rows
# ---------------------------------------------------------------------------
def season_minutes(res):
    """
    Each dressed player's 2026-27 minutes in a run (fixed by the lineups, the same in every simulated season):
    real minutes before the start + simulated games x minutes per game (skaters) or 60 x starts (goalies).
    Returns playerId -> minutes, plus the real and simulated parts.
    """
    lu = res["lineups"]
    d = lu[lu["dressed"]].set_index("playerId")
    sim = pd.Series(0.0, index=d.index)
    sk = res["skaters"].set_index("playerId")
    sim.loc[sk.index] = sk["sim_gp"].to_numpy() * d.loc[sk.index, "minutes_per_game"].to_numpy()
    gl = res["goalies"].groupby("playerId")["sim_gp"].sum()
    sim.loc[gl.index] = 60.0 * gl.to_numpy()
    rg = res["bundle_used"].get("real_games")
    real = pd.Series(0.0, index=d.index)
    if rg is not None:
        r = rg[(rg["date"] < res["start_date"]) & (rg["toi"] > 0)].groupby("playerId")["toi"].sum()
        real = r.reindex(d.index).fillna(0.0)
    return pd.DataFrame({"real_minutes": real, "sim_minutes": sim, "minutes": real + sim})


def season_rows(res, war60, minutes, E):
    """
    History rows for the simulated season: one per dressed player, season = 2026, toi = his minutes,
    gar = WAR/60 x minutes / 60 x goals per win. war60: playerId -> season WAR/60 (the story's season_war60).
    """
    lu = res["lineups"]
    d = lu[lu["dressed"]].drop_duplicates("playerId").set_index("playerId")
    ids = d.index
    toi = minutes["minutes"].reindex(ids).to_numpy()
    w = pd.Series(war60).reindex(ids).to_numpy()
    out = pd.DataFrame({"playerId": ids, "season": SIM_SEASON, "name": d["name"].to_numpy(), "team": d["team"].to_numpy(),
                        "position": d["position"].to_numpy(), "group": d["group"].to_numpy(), "toi": toi,
                        "war": w * toi / 60})
    out["gar"] = out["war"] * FC.GOALS_PER_WIN
    return out[E["history"].columns]


def _draws(res, k=0):
    """Dressed players' season WAR/60 in simulated season k: projection + z x draw size (notebook 30's draws)."""
    lu = res["lineups"]
    rows = np.flatnonzero(lu["dressed"].to_numpy())
    pos = np.searchsorted(res["z_ids"], lu["playerId"].to_numpy()[rows])
    size = np.asarray(res["draw_size"])[rows]
    proj = lu["proj_war_60"].to_numpy()[rows]
    return lu["playerId"].to_numpy()[rows], proj, size, pos


# ---------------------------------------------------------------------------
# 4. One story: the progression table
# ---------------------------------------------------------------------------
def _story_parts(E, res, war60):
    """Before, at projection (the aging part) and after, for one set of season WAR/60s."""
    mins = season_minutes(res)
    ids, proj, size, pos = _draws(res)
    at = project_season(E, NEXT, season_rows(res, pd.Series(proj, index=ids), mins, E))
    after = project_season(E, NEXT, season_rows(res, war60, mins, E))
    return at, after, mins


def story_progression(E, story):
    """
    Every player's 2027-28 OVR after one simulated season (a full-season franchise.story). Columns:
    OVR before (today's 2026-27 rating) and after, the whole-number change, and the two parts in exact OVR:
      - aging_part: re-projected with his 2026-27 season exactly at his projection (age +1, experience +1, his
        oldest season dropping out of the three-season window, and the prior one year on);
      - season_part: the rest (how his simulated season differed from his projection).
    Plus his season line (season to date), minutes, season WAR/60 and a flag for players with no NHL minutes.
    """
    res = story["run"]
    if not res["range"]["full_season"]:
        raise ValueError("progression needs a full season: play the story to the end of the regular season")
    dr = story["draws"].set_index("playerId")
    war60 = dr.loc[dr["dressed"], "season_war60"]
    at, after, mins = _story_parts(E, res, war60)
    return _table(E, at, after, mins, war60, story, real_minutes_all(res))


def real_minutes_all(res):
    """Every player's real 2026-27 NHL minutes before the run's start (boxscores; 0 at opening night)."""
    rg = res["bundle_used"].get("real_games")
    if rg is None:
        return pd.Series(dtype=float)
    return rg[(rg["date"] < res["start_date"]) & (rg["toi"] > 0)].groupby("playerId")["toi"].sum()


def _table(E, at, after, mins, war60, story=None, real_all=None):
    b = E["before"]
    a = after.set_index("playerId")
    m = at.set_index("playerId")
    ids = b.index
    t = pd.DataFrame({"name": b["name"], "team": b["team"], "position": b["position"], "group": b["group"],
                      "age": b["age"], "ovr_before": b["ovr"], "ovr_after": a["ovr"].reindex(ids),
                      "path_before": np.where(E["path"].set_index("playerId")["source"].reindex(ids).isin(NHL_SOURCES),
                                              "NHL", "prospect"),
                      "path_after": a["path"].reindex(ids)})
    t["change"] = t["ovr_after"] - t["ovr_before"]
    t["exact_change"] = a["ovr_exact"].reindex(ids) - b["ovr_exact"]
    t["aging_part"] = m["ovr_exact"].reindex(ids) - b["ovr_exact"]
    t["season_part"] = a["ovr_exact"].reindex(ids) - m["ovr_exact"].reindex(ids)
    t["minutes"] = mins["minutes"].reindex(ids)
    t["real_minutes"] = mins["real_minutes"].reindex(ids)
    t["proj_war_60"] = b["proj_war_60"]
    t["season_war_60"] = pd.Series(war60).reindex(ids)
    t["new_proj_war_60"] = a["proj_war_60"].reindex(ids)
    t["engine_ovr_after"] = a["engine_ovr"].reindex(ids)          # the engine's own number (prospects: no new season)
    t["engine_change"] = t["engine_ovr_after"] - t["ovr_before"]
    played = t["minutes"].notna()
    prospect_now = ~played & (t["path_after"] == "prospect")
    had_real = (pd.Series(real_all, dtype=float).reindex(ids).fillna(0) > 0) if real_all is not None else pd.Series(False, index=ids)
    t["flag"] = np.select([t["ovr_after"].isna(), prospect_now & had_real, prospect_now, ~played & had_real, ~played],
                          [FLAG_LEFT, FLAG_PROSPECT + "; his real 2026-27 games so far aren't counted", FLAG_PROSPECT,
                           FLAG_REAL_ONLY, FLAG_NO_MINUTES], "")
    if story is not None:
        sk = story["season_skaters"].set_index("playerId")
        g = story["season_goalies"].set_index("playerId")
        line = pd.Series("", index=ids, dtype=object)
        s_ids = sk.index.intersection(ids)
        line.loc[s_ids] = [f"{int(r.gp)} GP, {int(r.g)}-{int(r.a)}-{int(r.p)}" for r in sk.loc[s_ids].itertuples()]
        g_ids = g.index.intersection(ids)
        line.loc[g_ids] = [f"{int(r.gp)} GP, {int(r.w)}-{int(r.l)}-{int(r.otl)}, {r.gaa:.2f} GAA" if r.gp > 0 else ""
                           for r in g.loc[g_ids].itertuples()]
        t["season_line"] = line
    t["ovr_after"] = t["ovr_after"].astype("Int64")
    t["change"] = t["change"].astype("Int64")
    t["engine_ovr_after"] = t["engine_ovr_after"].astype("Int64")
    t["engine_change"] = t["engine_change"].astype("Int64")
    return t.rename_axis("playerId").reset_index()


def risers_fallers(table, n=10, played_only=True):
    """The n biggest risers and fallers (whole-number change, then exact change), players with NHL minutes."""
    t = table[table["minutes"].notna()] if played_only else table
    t = t[t["change"].notna()]
    up = t.sort_values(["change", "exact_change"], ascending=False).head(n)
    down = t.sort_values(["change", "exact_change"], ascending=True).head(n)
    return up.reset_index(drop=True), down.reset_index(drop=True)


def team_roster(table, team):
    """A team's players (today's team), highest new OVR first."""
    return table[table["team"] == team].sort_values(["ovr_after", "exact_change"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Many seasons: the straight-line shortcut
# ---------------------------------------------------------------------------
STEP = 0.01                                       # WAR/60 step for the slope


def ovr_exact_array(war_60, groups):
    """ratings.ovr_exact for a seasons x players array (groups: one per column). Same anchors, same formula."""
    w = np.asarray(war_60, dtype=float)
    groups = np.asarray(groups)
    minutes = np.array([RT.STD_MINUTES[g] for g in groups])
    ws = w * minutes / 60
    out = np.interp(ws, RT.ANCHOR_WAR, RT.ANCHOR_OVR)
    is_g = groups == "G"
    out[..., is_g] = np.interp(ws[..., is_g], RT.GOALIE_ANCHOR_WAR, RT.GOALIE_ANCHOR_OVR)
    return out


def linear_maps(E, res):
    """
    Lineups are fixed for the season, so each player's minutes, experience and prior inputs are fixed too, and his
    new projection is a straight line in his season WAR/60. Two re-projections give every player's line:
    at = the rating WAR/60 with the season exactly at his projection, slope = change per 1 WAR/60 of season.
    """
    mins = season_minutes(res)
    ids, proj, size, pos = _draws(res)
    at = project_season(E, NEXT, season_rows(res, pd.Series(proj, index=ids), mins, E)).set_index("playerId")
    up = project_season(E, NEXT, season_rows(res, pd.Series(proj + STEP, index=ids), mins, E)).set_index("playerId")
    slope = (up["rating_war_60"] - at["rating_war_60"]) / STEP
    return {"at": at, "slope": slope, "mins": mins, "ids": ids, "proj": proj, "size": size, "pos": pos}


def apply_maps(maps, war60):
    """New rating WAR/60 for one set of season WAR/60s (Series by playerId), from the straight lines."""
    at, slope = maps["at"], maps["slope"]
    dev = (pd.Series(war60).reindex(at.index) - pd.Series(maps["proj"], index=maps["ids"]).reindex(at.index)).fillna(0)
    return at["rating_war_60"] + slope.fillna(0) * dev


def progression_many(E, inputs, team, n_sims=10000, seed=0, sit_out=(), trade_ids=None, overrides=None,
                     goalie_share=None, minutes_rule="moved", chunk=2000):
    """
    n_sims full seasons with the franchise choices (notebook 30's play): every player's expected OVR change and the
    chance of +3 or more and -3 or worse, from the straight-line maps (checked against the direct way on a story).
    """
    import franchise as FR
    res = FR.play(inputs, team, None, None, trade_ids, sit_out, overrides, goalie_share, minutes_rule, n_sims, seed,
                  story=False, keep_z=True)
    maps = linear_maps(E, res)
    b = E["before"]
    at = maps["at"].reindex(b.index)
    slope = maps["slope"].reindex(b.index).fillna(0).to_numpy()
    grp = b["group"]
    col = pd.Series(np.arange(len(b)), index=b.index)
    rows = col.reindex(maps["ids"]).to_numpy()
    ok = ~np.isnan(rows)
    rows = rows[ok].astype(int)
    z = res["z"][:, maps["pos"][ok]] * maps["size"][ok]               # WAR/60 change of each dressed player
    base = at["rating_war_60"].to_numpy()
    before = b["ovr"].to_numpy()
    sums = {k: np.zeros(len(b)) for k in ["change", "up3", "down3", "exact"]}
    after_all = []
    for c0 in range(0, n_sims, chunk):
        w = np.tile(base, (min(chunk, n_sims - c0), 1))
        w[:, rows] += slope[rows] * z[c0:c0 + chunk]
        exact = ovr_exact_array(w, grp.to_numpy())
        ovr = np.floor(exact + 0.5)
        ch = ovr - before
        sums["change"] += ch.sum(0)
        sums["up3"] += (ch >= 3).sum(0)
        sums["down3"] += (ch <= -3).sum(0)
        sums["exact"] += (exact - b["ovr_exact"].to_numpy()).sum(0)
        after_all.append((ovr - before).astype(np.float32))           # NaN: left the projection
    changes = np.vstack(after_all)                                     # n_sims x players, whole-number changes
    out = pd.DataFrame({"name": b["name"], "team": b["team"], "position": b["position"], "group": grp, "age": b["age"],
                        "ovr_before": b["ovr"], "ovr_at_projection": at["ovr"], "expected_change": sums["change"] / n_sims,
                        "expected_exact_change": sums["exact"] / n_sims, "pct_up3": sums["up3"] / n_sims * 100,
                        "pct_down3": sums["down3"] / n_sims * 100,
                        "ovr_after_p10": before + np.percentile(changes, 10, axis=0),
                        "ovr_after_p90": before + np.percentile(changes, 90, axis=0),
                        "dressed": b.index.isin(maps["ids"]), "minutes": maps["mins"]["minutes"].reindex(b.index).to_numpy()})
    gone = at["rating_war_60"].isna().to_numpy()                       # left the projection: no new rating
    out.loc[gone, ["expected_change", "expected_exact_change", "pct_up3", "pct_down3"]] = np.nan
    out = out.rename_axis("playerId").reset_index()
    return {"table": out, "changes": changes, "maps": maps, "run": res}


# ---------------------------------------------------------------------------
# 6. Calibration against real history
# ---------------------------------------------------------------------------
def real_ratings(E, targets):
    """The engine's NHL-path rating going into each target season, on real history only (the same frozen models).
    {target: table}. For target 2026 this is today's NHL ratings exactly."""
    return {t: project_season(E, t, prospects=False).set_index("playerId") for t in targets}


def real_changes(E, rated, seasons):
    """
    Real year-to-year OVR changes: rating going into season t vs. going into t+1, for players rated both times.
    minutes = his real NHL minutes in season t (0 if none); age = his age in season t.
    """
    h = E["history"]
    rows = []
    for t in seasons:
        a, b = rated[t], rated[t + 1]
        ids = a.index.intersection(b.index)
        mins = h[h["season"] == t].groupby("playerId")["toi"].sum().reindex(ids).fillna(0.0)
        age = FC.season_age(FC._birth_dates(E["bios_all"], ids), np.full(len(ids), t))
        rows.append(pd.DataFrame({"playerId": ids, "season": t, "group": a.loc[ids, "group"].to_numpy(), "age": age,
                                  "minutes": mins.to_numpy(), "ovr_before": a.loc[ids, "ovr"].astype(int).to_numpy(),
                                  "ovr_after": b.loc[ids, "ovr"].astype(int).to_numpy(),
                                  "exact_change": (b.loc[ids, "ovr_exact"] - a.loc[ids, "ovr_exact"]).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["change"] = out["ovr_after"] - out["ovr_before"]
    out["played"] = out["minutes"] >= out["group"].map(MIN_SEASON_MINUTES)
    out["age_group"] = age_group(out["age"]).to_numpy()
    return out


def real_parts(E, rated, seasons):
    """
    Real changes split like the simulated ones, for players who played in season t: aging part = re-projected with
    his season t set exactly at his projection going into t (real minutes), season part = the rest. Also his real
    minus projected WAR/60 in t. Shows where a real-vs-simulated gap in the average change comes from.
    """
    h = E["history"]
    rows = []
    for t in seasons:
        a, b = rated[t], rated[t + 1]
        hm = h.copy()
        sel = (hm["season"] == t) & hm["playerId"].isin(a.index)
        p = hm.loc[sel, "playerId"].map(a["proj_war_60"])
        hm.loc[sel, "war"] = p * hm.loc[sel, "toi"] / 60
        hm.loc[sel, "gar"] = hm.loc[sel, "war"] * FC.GOALS_PER_WIN
        at = project_season(dict(E, history=hm), t + 1, prospects=False).set_index("playerId")
        ht = h[h["season"] == t].groupby("playerId")[["war", "toi"]].sum()
        ids = a.index.intersection(b.index).intersection(at.index)
        mins = ht["toi"].reindex(ids).fillna(0)
        ids = ids[(mins >= a.loc[ids, "group"].map(MIN_SEASON_MINUTES)).to_numpy()]
        rows.append(pd.DataFrame({"playerId": ids, "season": t, "group": a.loc[ids, "group"].to_numpy(),
                                  "minutes": ht.loc[ids, "toi"].to_numpy(),
                                  "aging_part": (at.loc[ids, "ovr_exact"] - a.loc[ids, "ovr_exact"]).to_numpy(),
                                  "season_part": (b.loc[ids, "ovr_exact"] - at.loc[ids, "ovr_exact"]).to_numpy(),
                                  "season_vs_projection": (ht.loc[ids, "war"] / ht.loc[ids, "toi"] * 60
                                                           - a.loc[ids, "proj_war_60"]).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def change_stats(change, exact=None, weight=None):
    """Mean, SD, % at +3 or more and % at -3 or worse of whole-number changes (and the SD of the exact change)."""
    c = np.asarray(change, dtype=float)
    out = {"n": len(c), "mean": c.mean(), "sd": c.std(), "pct_up3": (c >= 3).mean() * 100, "pct_down3": (c <= -3).mean() * 100}
    if exact is not None:
        out["sd_exact"] = np.asarray(exact, dtype=float).std()
    return pd.Series(out)


def compare_table(real, sim_changes, sim_info, by):
    """Real next to simulated, by `by` (a column of both: group, age_group, or both). sim_changes: n_sims x players
    (whole-number changes), sim_info: one row per player with the `by` columns."""
    rows = []
    keys = sorted(set(map(tuple, real[by].astype(str).to_numpy())) | set(map(tuple, sim_info[by].astype(str).to_numpy())))
    for k in keys:
        r = real[(real[by].astype(str).to_numpy() == np.array(k)).all(axis=1)]
        cols = np.flatnonzero((sim_info[by].astype(str).to_numpy() == np.array(k)).all(axis=1))
        s = change_stats(sim_changes[:, cols].ravel()) if len(cols) else pd.Series(dtype=float)
        rr = change_stats(r["change"]) if len(r) else pd.Series(dtype=float)
        row = dict(zip(by, k))
        for name, st in [("real", rr), ("sim", s)]:
            for c in ["n", "mean", "sd", "pct_up3", "pct_down3"]:
                row[f"{name}_{c}"] = st.get(c, np.nan)
        if len(r):
            per = r.groupby("season")["change"].agg(["mean", "std"])
            row["real_mean_by_season"] = f"{per['mean'].min():+.2f} to {per['mean'].max():+.2f}"
            row["real_sd_by_season"] = f"{per['std'].min():.2f} to {per['std'].max():.2f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["real_n"] = out["real_n"].astype("Int64")
    out["sim_players"] = [int(((sim_info[by].astype(str).to_numpy() == np.array(k)).all(axis=1)).sum()) for k in keys]
    return out.drop(columns="sim_n")


# ---------------------------------------------------------------------------
# 7. Prospects: real year-to-year changes with their real new pre-NHL season
# ---------------------------------------------------------------------------
def real_prospect_changes(E, seasons):
    """
    For each season t: notebook 19's prospect list going into t (development.prospect_table, draft classes on file:
    2019 on), those with no NHL minutes in t, rated going into t and going into t+1 two ways:
      - real: with their real season-t pre-NHL season (what actually happened);
      - engine: with no new season (what the progression engine does for a prospect who isn't dressed).
    Returns one row per prospect-season.
    """
    d = E["data_dir"]
    drafts = build_draft_table(d / "draft_raw")
    h = E["history"]
    rows = []
    for t in seasons:
        projected = set(DV.candidates(h, [t])["playerId"]) | set(DV.goalie_candidates(h, [t])["playerId"])
        prof = E["profiles_all"][E["profiles_all"]["season"] <= t - 1]
        pros, _ = DV.prospect_table(drafts[drafts["draft_year"] <= t], prof, projected, season=t - 1,
                                    excluded=PROSPECT_EXCLUDED)
        pros = pros[~pros["playerId"].isin(h.loc[h["season"] == t, "playerId"])]
        players = pros[["playerId", "group"]].assign(new=False)
        r0 = _rating(prospect_path(E, players, h, t).merge(players[["playerId", "group"]], on="playerId").assign(path="prospect"))
        r1 = _rating(prospect_path(E, players, h, t + 1).merge(players[["playerId", "group"]], on="playerId").assign(path="prospect"))
        r2 = _rating(prospect_path(E, players, h, t + 1, through=t - 1).merge(players[["playerId", "group"]], on="playerId").assign(path="prospect"))
        m = (r0.set_index("playerId")[["group", "ovr", "ovr_exact"]]
             .join(r1.set_index("playerId")[["ovr", "ovr_exact"]], rsuffix="_real")
             .join(r2.set_index("playerId")[["ovr", "ovr_exact"]], rsuffix="_engine"))
        m["season"] = t
        m["age"] = pros.set_index("playerId")["age"].reindex(m.index)
        m["draft_year"] = pros.set_index("playerId")["draft_year"].reindex(m.index)
        rows.append(m.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["change_real"] = (out["ovr_real"] - out["ovr"]).astype(int)
    out["change_engine"] = (out["ovr_engine"] - out["ovr"]).astype(int)
    out["exact_real"] = out["ovr_exact_real"] - out["ovr_exact"]
    out["exact_engine"] = out["ovr_exact_engine"] - out["ovr_exact"]
    out["age_group"] = prospect_age_group(out["age"]).to_numpy()
    return out


def expected_development(prospects):
    """The option (not applied): the real average exact OVR change of a prospect year, by age group and position."""
    out = prospects.assign(age_group=prospects["age_group"].astype(str)).groupby(["group", "age_group"])["exact_real"]
    return out.agg(["mean", "size"]).rename(columns={"mean": "expected_change", "size": "prospect_seasons"})


# ---------------------------------------------------------------------------
# 8. The zero-change check and the window piece
# ---------------------------------------------------------------------------
def zero_change_check(E, res):
    """
    With aging switched off (a zero aging curve) and the prior held at its 2026-27 value, a 2026-27 season exactly
    at his projection must leave a player's projection unchanged, as long as no season drops out of his three-season
    window (no 2023-24 NHL season). Players who did play in 2023-24 change: that's the window piece.
    Returns one row per NHL-path player: lost_2023 (his 2023-24 season drops out), dressed, the change in WAR/60 and
    in exact OVR.
    """
    zero = E["sk"]["curve"].assign(delta=0.0)
    p26 = project_season(E, SIM_SEASON, curve=zero, prospects=False)
    pri = {k: v.assign(season=NEXT) for k, v in p26.attrs["priors"].items()}
    p26 = p26.set_index("playerId")
    mins = season_minutes(res)
    ids, proj, size, pos = _draws(res)
    war60 = p26["proj_war_60"].reindex(ids).fillna(pd.Series(proj, index=ids))
    p27 = project_season(E, NEXT, season_rows(res, war60, mins, E), curve=zero, priors=pri,
                         prospects=False).set_index("playerId")
    both = p26.index.intersection(p27.index)
    h = E["history"]
    lost = set(h.loc[h["season"] == SIM_SEASON - 3, "playerId"])
    out = pd.DataFrame({"group": p26.loc[both, "group"], "lost_2023": both.isin(lost), "dressed": both.isin(ids),
                        "change_war_60": p27.loc[both, "proj_war_60"] - p26.loc[both, "proj_war_60"],
                        "change_ovr": p27.loc[both, "ovr_exact"] - p26.loc[both, "ovr_exact"]})
    return out.rename_axis("playerId").reset_index()
