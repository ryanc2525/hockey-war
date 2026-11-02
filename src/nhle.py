"""
NHL equivalency (NHLe): how many NHL points per game a player's scoring in another league is worth.

    from nhle import load_profiles, career_totals, next_season_pairs, method_a, same_season_links, method_b
    raw = load_profiles(Path("../data/bio_raw"))      # every season row in every NHL player profile
    careers = career_totals(raw)                       # one row per skater, season, league
    a = method_a(next_season_pairs(careers))           # factors from "league L, then the NHL next season"
    b = method_b(same_season_links(careers))           # factors from players in two leagues in one season

A factor of 0.40 means 1.00 point per game in that league is worth about 0.40 points per game in the NHL.
Only points exist outside the NHL (no TOI or xG), and the profiles only include players who reached the
NHL, so every factor describes NHL-bound players.
season = start year (2023 = 2023-24). Age in a season = age on February 1 of its end year.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from forecast import season_age

CURRENT_SEASON = 2026                    # 2026-27 is still being played: drop it and anything later
FIRST_SEASON, LAST_SEASON = 2005, 2025   # league seasons used for the factors
SKIP_SEASONS = {2012, 2020}              # 2012-13 lockout (NHL players in Europe), 2020-21 COVID loans
MIN_GP_PAIR = 20                         # games in both seasons for a next-season pair (method A)
MIN_DIRECT_PAIRS = 30                    # pairs for a direct method A factor
MIN_GP_SAME = 5                          # games in both leagues for a same-season pair (method B)
MIN_LINK = 8                             # players for a link to be used in a chain
MAX_STEPS = 3                            # longest chain allowed
AGE_BUCKETS = ["<=18", "19", "20", "21", "22-24", "25+"]

# Same league, older name -> the name we use. Before 2015 the profiles store NCAA players under their
# conference, so every conference maps to NCAA.
LEAGUE_MAP = {
    "Sweden": "SHL",
    "Finland": "Liiga",
    "Swiss": "NL", "NLA": "NL",
    "Rus-KHL": "KHL", "Russia": "KHL",     # Russia = Russian Superleague (2005-07), the KHL's predecessor
    "WCHA": "NCAA", "H-East": "NCAA", "CCHA": "NCAA", "ECAC": "NCAA", "NCHC": "NCAA",
    "Big Ten": "NCAA", "CHA": "NCAA",
    "Sweden-2": "HockeyAllsvenskan", "Allsvenskan": "HockeyAllsvenskan",
    "Germany": "DEL",
    "CzRep": "Czechia", "Czech": "Czechia",
    "Austria": "ICEHL", "EBEL": "ICEHL",
    "Swe-Jr.": "J20 Nationell", "SuperElit": "J20 Nationell", "J20 SuperElit": "J20 Nationell",
    "Russia-Jr.": "MHL",
    "Finland-2": "Mestis",
    "Swiss-2": "SL", "NLB": "SL",
}

# Not a league season: international tournaments, short events, and US NTDP rows that mix USHL games
# with exhibitions (the same players also have a separate USHL row, which we keep).
EXCLUDED = {
    # international and national-team events
    "WJC-A", "WJC-20", "WJC-18", "WJ18-A", "WJ18-B", "WJC-B", "WJC-18 D1A", "WJC-20 D1A", "WHC-17",
    "WC", "WC-A", "WC D1A", "OG", "Olympics", "OGQ", "4 Nations", "WCup", "W-Cup", "EHT", "Nat-Tm",
    "International", "YOG", "CWG", "EYOF", "EJC-A", "U17-Dev", "U18-Dev", "WJAC-19",
    "Hlinka Gretzky Cup", "Ivan Hlinka Memorial", "OGC-16", "QGC-16", "WCCC-16", "JCWC",
    # club tournaments and short events
    "Champions HL", "Spengler Cup", "Continental Cup", "EuroHL", "Memorial Cup", "M-Cup", "OHL Cup",
    "Brick Invitational", "WSI U12", "WSI U13", "WSI U14", "WSI U15", "TV-Pucken", "QC Int PW",
    "John Reid Memorial", "Exhib.",
    # US National Team Development Program composite rows
    "U-17", "U-18", "USA-S15", "USA-S16", "USA-S17", "USDP", "NTDP",
}

# hockeystats.com/methodology/nhle (2005-2020 data), for comparison only
PUBLISHED = {
    "KHL": 0.772, "SHL": 0.566, "NL": 0.459, "Liiga": 0.441, "AHL": 0.389, "DEL": 0.352,
    "HockeyAllsvenskan": 0.351, "NCAA": 0.194, "OHL": 0.144, "USHL": 0.143, "MHL": 0.143,
    "WHL": 0.141, "QMJHL": 0.113,
}


# ---------------------------------------------------------------------------
# 1. Career table
# ---------------------------------------------------------------------------
def load_profiles(bio_dir):
    """One row per seasonTotals entry in every profile (all leagues, regular season and playoffs)."""
    rows = []
    for path in sorted(bio_dir.glob("*.json")):
        d = json.loads(path.read_text())
        name = f"{d['firstName']['default']} {d['lastName']['default']}"
        for s in d.get("seasonTotals", []):
            rows.append({
                "playerId": d["playerId"], "name": name, "position": d.get("position"),
                "birth_date": d.get("birthDate"), "season": s["season"] // 10000,   # 20232024 -> 2023
                "league": s.get("leagueAbbrev"), "gameTypeId": s.get("gameTypeId"),
                "gp": s.get("gamesPlayed"), "goals": s.get("goals"), "assists": s.get("assists"),
                "points": s.get("points"),
            })
    return pd.DataFrame(rows)


def _skater_regular_season(raw):
    """Regular-season skater rows from finished seasons, with points recorded."""
    keep = (raw["gameTypeId"] == 2) & (raw["position"] != "G") & (raw["season"] < CURRENT_SEASON)
    return raw[keep & raw["points"].notna() & raw["gp"].notna()]


def career_totals(raw):
    """One row per skater, season and league: games and points summed over teams, plus group and age."""
    r = _skater_regular_season(raw)
    r = r[~r["league"].isin(EXCLUDED)].copy()
    r["league"] = r["league"].replace(LEAGUE_MAP)
    c = (r.groupby(["playerId", "season", "league"], as_index=False)
          .agg(name=("name", "first"), position=("position", "first"), birth_date=("birth_date", "first"),
               gp=("gp", "sum"), goals=("goals", "sum"), assists=("assists", "sum"), points=("points", "sum")))
    c["group"] = np.where(c["position"] == "D", "D", "F")
    c["age"] = season_age(c["birth_date"], c["season"])
    cols = ["playerId", "name", "position", "group", "birth_date", "season", "age", "league",
            "gp", "goals", "assists", "points"]
    return c[cols].sort_values(["playerId", "season", "league"]).reset_index(drop=True)


def league_inventory(raw):
    """Every raw league name: players, rows, seasons, median games per player-season, and what we do with it."""
    r = _skater_regular_season(raw)
    per = r.groupby(["league", "playerId", "season"])["gp"].sum().groupby("league").median()
    inv = r.groupby("league").agg(players=("playerId", "nunique"), rows=("playerId", "size"),
                                  first=("season", "min"), last=("season", "max"))
    inv["median_gp"] = per
    inv["excluded"] = inv.index.isin(EXCLUDED)
    inv["maps_to"] = [LEAGUE_MAP.get(lg, "") for lg in inv.index]
    return inv.sort_values("players", ascending=False)


def age_bucket(age):
    """<=18, 19, 20, 21, 22-24, 25+ (NaN stays NaN)."""
    a = pd.Series(age, dtype=float)
    return pd.cut(a, [-np.inf, 18, 19, 20, 21, 24, np.inf], labels=AGE_BUCKETS).astype(object)


def in_window(season, first=FIRST_SEASON, last=LAST_SEASON):
    """True for league seasons used in the factors."""
    s = pd.Series(season)
    return (s.between(first, last) & ~s.isin(SKIP_SEASONS)).to_numpy()


def ppg_ratio(gp_from, pts_from, gp_to, pts_to):
    """(sum points / sum games) in the 'to' league over the same in the 'from' league."""
    return (pts_to.sum() / gp_to.sum()) / (pts_from.sum() / gp_from.sum())


# ---------------------------------------------------------------------------
# 2. Chains: reach the NHL through other leagues
# ---------------------------------------------------------------------------
def chain_factors(links, anchors, max_steps=MAX_STEPS):
    """
    For each league without an anchor, the shortest chains of links that reach an anchor league.

    links: rows with league_from, league_to, factor, pairs, median_age.
    anchors: {league: (factor, pairs, median_age, steps it counts as)}. Method A anchors are the direct
    factors (each counts as 1 step); method B's only anchor is the NHL itself (0 steps, factor 1).
    A path's factor is the product of its links times the anchor's factor. When several shortest paths
    exist, they are averaged, each weighted by its smallest player count (a chain is only as reliable as
    its weakest link).
    """
    out_links = defaultdict(list)
    for r in links.itertuples(index=False):
        out_links[r.league_from].append(r)
    rows = []
    for league in sorted(set(links["league_from"]) - set(anchors)):
        found, frontier = [], [(league, [], 1.0)]          # (current league, links so far, factor so far)
        while frontier and not found:
            nxt = []
            for here, path, f in frontier:
                for link in out_links[here]:
                    visited = {league} | {p.league_to for p in path}
                    if link.league_to in visited:
                        continue
                    new_path = path + [link]
                    if link.league_to in anchors:
                        a_f, a_n, a_age, a_steps = anchors[link.league_to]
                        if len(new_path) + a_steps <= max_steps:
                            found.append((new_path, f * link.factor * a_f, a_n, a_age, a_steps))
                    elif len(new_path) + 1 <= max_steps:
                        nxt.append((link.league_to, new_path, f * link.factor))
            frontier = nxt
        if not found:
            continue
        weights = np.array([min([p.pairs for p in path] + [a_n]) for path, _, a_n, _, _ in found], dtype=float)
        factors = np.array([f for _, f, _, _, _ in found])
        texts = []
        for (path, f, a_n, a_age, a_steps), w in zip(found, weights):
            steps = [f"{p.league_from}>{p.league_to} ({p.pairs:.0f}, age {p.median_age:.0f})" for p in path]
            if a_steps:
                steps.append(f"{path[-1].league_to} direct ({a_n:.0f}, age {a_age:.0f})")
            texts.append(f"{' x '.join(steps)} = {f:.3f} [weight {w:.0f}]")
        rows.append({"league": league, "factor": np.average(factors, weights=weights),
                     "pairs": weights.sum(), "steps": len(found[0][0]) + found[0][4],
                     "n_paths": len(found), "paths": " | ".join(texts)})
    return pd.DataFrame(rows, columns=["league", "factor", "pairs", "steps", "n_paths", "paths"])


# ---------------------------------------------------------------------------
# 3. Method A: league L in season t, then season t+1
# ---------------------------------------------------------------------------
def next_season_pairs(careers, first=FIRST_SEASON, last=LAST_SEASON, max_to_season=None, min_gp=MIN_GP_PAIR):
    """
    Every move from league L (season t) to league M (season t+1) with min_gp games in both, including M = NHL.
    t must be in the window; neither t nor t+1 can be a skipped season. max_to_season limits t+1
    (used to train the backtest on early years only).
    """
    c = careers[careers["gp"] >= min_gp]
    frm = c[in_window(c["season"], first, last) & ~c["season"].add(1).isin(SKIP_SEASONS).to_numpy()]
    to = c[["playerId", "season", "league", "gp", "points"]].copy()
    to["season"] -= 1                                    # line up season t+1 with season t
    p = frm.merge(to, on=["playerId", "season"], suffixes=("_from", "_to"))
    p = p[p["league_from"] != p["league_to"]]           # staying in the same league tells us nothing
    if max_to_season is not None:
        p = p[p["season"] + 1 <= max_to_season]
    return p[["playerId", "name", "group", "season", "age", "league_from", "gp_from", "points_from",
              "league_to", "gp_to", "points_to"]].reset_index(drop=True)


def _link_table(pairs):
    """One row per (league_from, league_to): pairs, factor and median age at the 'from' season.
    Links where the 'from' players scored no points at all (factor = infinity) are dropped."""
    g = pairs.groupby(["league_from", "league_to"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = g.apply(lambda x: pd.Series({
            "pairs": len(x),
            "factor": ppg_ratio(x["gp_from"], x["points_from"], x["gp_to"], x["points_to"]),
            "median_age": x["age"].median()}), include_groups=False)
    t = t[np.isfinite(t["factor"])]
    return t.reset_index()


def direct_factors(pairs, min_pairs=MIN_DIRECT_PAIRS):
    """Method A direct factors: league L in season t -> NHL in t+1, for leagues with min_pairs pairs."""
    d = _link_table(pairs[pairs["league_to"] == "NHL"])
    return d[d["pairs"] >= min_pairs].drop(columns="league_to").reset_index(drop=True)


def method_a(pairs, min_pairs=MIN_DIRECT_PAIRS, min_link=MIN_LINK, max_steps=MAX_STEPS):
    """Direct factors where a league has min_pairs pairs, chained factors only for the rest."""
    direct = direct_factors(pairs, min_pairs)
    links = _link_table(pairs[pairs["league_to"] != "NHL"])
    links = links[links["pairs"] >= min_link]
    anchors = {r.league_from: (r.factor, r.pairs, r.median_age, 1) for r in direct.itertuples()}
    chained = chain_factors(links, anchors, max_steps)
    direct = direct.rename(columns={"league_from": "league"}).assign(steps=1, n_paths=1, paths="direct")
    out = pd.concat([direct.assign(path_type="direct"), chained.assign(path_type="chained")],
                    ignore_index=True)
    out = out[out["league"] != "NHL"]
    return out[["league", "factor", "pairs", "path_type", "steps", "n_paths", "paths"]] \
        .sort_values("factor", ascending=False).reset_index(drop=True)


def direct_breakdown(pairs, min_pairs=MIN_DIRECT_PAIRS):
    """Direct method A factors for forwards, defensemen and everyone (pairs in brackets)."""
    nhl = pairs[pairs["league_to"] == "NHL"]
    keep = nhl["league_from"].value_counts()
    nhl = nhl[nhl["league_from"].isin(keep[keep >= min_pairs].index)]
    rows = {}
    for label, part in [("F", nhl[nhl["group"] == "F"]), ("D", nhl[nhl["group"] == "D"]), ("all", nhl)]:
        t = _link_table(part).set_index("league_from")
        rows[f"{label} factor"] = t["factor"]
        rows[f"{label} pairs"] = t["pairs"]
    return pd.DataFrame(rows).sort_values("all factor", ascending=False)


def age_breakdown(pairs, min_pairs=MIN_DIRECT_PAIRS, min_cell=20):
    """Direct method A factors by age at season t; cells with fewer than min_cell pairs are blank."""
    nhl = pairs[pairs["league_to"] == "NHL"].assign(bucket=lambda x: age_bucket(x["age"]).to_numpy())
    keep = nhl["league_from"].value_counts()
    nhl = nhl[nhl["league_from"].isin(keep[keep >= min_pairs].index)]
    t = (nhl.groupby(["league_from", "bucket"])
            .apply(lambda x: pd.Series({"pairs": len(x), "factor": ppg_ratio(
                x["gp_from"], x["points_from"], x["gp_to"], x["points_to"])}), include_groups=False)
            .reset_index())
    t.loc[t["pairs"] < min_cell, "factor"] = np.nan
    wide = t.pivot(index="league_from", columns="bucket", values="factor")
    return wide.reindex(columns=[b for b in AGE_BUCKETS if b in wide.columns])


# ---------------------------------------------------------------------------
# 4. Method B: two leagues in the same season
# ---------------------------------------------------------------------------
def same_season_links(careers, first=FIRST_SEASON, last=LAST_SEASON, min_gp=MIN_GP_SAME, min_players=MIN_LINK):
    """Players with min_gp games in two leagues in one season: one link per ordered league pair."""
    c = careers[(careers["gp"] >= min_gp) & in_window(careers["season"], first, last)]
    m = c.merge(c[["playerId", "season", "league", "gp", "points"]], on=["playerId", "season"],
                suffixes=("_from", "_to"))
    m = m[m["league_from"] != m["league_to"]]
    links = _link_table(m)
    return links[links["pairs"] >= min_players].reset_index(drop=True)


def method_b(links, max_steps=MAX_STEPS):
    """Factor to the NHL for every league that can reach it; direct = has its own link to the NHL."""
    t = chain_factors(links, {"NHL": (1.0, np.inf, np.nan, 0)}, max_steps)
    t["path_type"] = np.where(t["steps"] == 1, "direct", "chained")
    t["paths"] = t["paths"].str.replace(r" x NHL direct \(inf, age nan\)", "", regex=True)
    return t[["league", "factor", "pairs", "path_type", "steps", "n_paths", "paths"]] \
        .sort_values("factor", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Comparison and backtest
# ---------------------------------------------------------------------------
def compare_published(a, b, flag=0.30):
    """A, B and published side by side, with % differences and a flag when more than `flag` off."""
    t = pd.DataFrame({"published": pd.Series(PUBLISHED)})
    t["A"] = a.set_index("league")["factor"]
    t["A type"] = a.set_index("league")["path_type"]
    t["B"] = b.set_index("league")["factor"]
    t["B type"] = b.set_index("league")["path_type"]
    for m in ["A", "B"]:
        t[f"{m} vs pub"] = t[m] / t["published"] - 1
        t[f"{m} flag"] = np.where(t[f"{m} vs pub"].abs() > flag, "FLAG", "")
    return t.sort_values("published", ascending=False)


def rookie_rows(careers, min_gp=20, max_gp=None, min_prior_gp=MIN_GP_PAIR):
    """
    Each skater's first NHL season with min_gp+ games (and at most max_gp), paired with the season right
    before it: the non-NHL league where he played the most games (min_prior_gp+). The prior season must
    be inside the factor window (2005-06 to 2025-26, not 2012-13 or 2020-21).
    """
    nhl = careers[(careers["league"] == "NHL") & (careers["gp"] >= min_gp)]
    first = nhl.sort_values("season").groupby("playerId").head(1)
    if max_gp is not None:
        first = first[first["gp"] <= max_gp]
    prior = careers[(careers["league"] != "NHL") & (careers["gp"] >= min_prior_gp)]
    prior = prior.sort_values(["gp", "points"], ascending=False).groupby(["playerId", "season"]).head(1)
    prior = prior.assign(season=prior["season"] + 1)     # line up with the NHL season after it
    r = first.merge(prior[["playerId", "season", "league", "gp", "points", "age"]],
                    on=["playerId", "season"], suffixes=("", "_prior"))
    r = r[in_window(r["season"] - 1)]                    # prior season must be a league season we use
    return pd.DataFrame({
        "playerId": r["playerId"], "name": r["name"], "group": r["group"], "season": r["season"],
        "nhl_gp": r["gp"], "nhl_pts": r["points"], "nhl_ppg": r["points"] / r["gp"],
        "prior_league": r["league_prior"], "prior_gp": r["gp_prior"], "prior_pts": r["points_prior"],
        "prior_ppg": r["points_prior"] / r["gp_prior"], "prior_age": r["age_prior"],
    }).reset_index(drop=True)


def predict(rookies, factors, multipliers=None):
    """Predicted NHL points per game: prior league ppg x factor (x age multiplier if given)."""
    pred = rookies["prior_ppg"] * rookies["prior_league"].map(factors)
    if multipliers is not None:
        pred = pred * age_bucket(rookies["prior_age"]).map(multipliers).fillna(1.0).to_numpy()
    return pred


def age_multipliers(rookies, pred):
    """One number per age bucket: sum of actual NHL points / sum of predicted NHL points (GP-weighted)."""
    ok = pred.notna()
    r = rookies[ok].assign(pred_pts=(pred[ok] * rookies.loc[ok, "nhl_gp"]),
                           bucket=age_bucket(rookies.loc[ok, "prior_age"]).to_numpy())
    g = r.groupby("bucket")
    return (g["nhl_pts"].sum() / g["pred_pts"].sum()).reindex(AGE_BUCKETS)


def gp_rmse(actual, pred, gp):
    """Games-weighted root mean squared error of points per game."""
    ok = pred.notna() & actual.notna()
    return float(np.sqrt(np.average((actual[ok] - pred[ok]) ** 2, weights=gp[ok])))


def gp_bias(actual, pred, gp):
    """Games-weighted average of predicted minus actual (positive = over-predicts)."""
    ok = pred.notna() & actual.notna()
    return float(np.average(pred[ok] - actual[ok], weights=gp[ok]))


def ordering_checks(f):
    """Rough hockey ordering of the factors in dict f; 'n/a' when a league is missing."""
    def ok(cond, *leagues):
        return "n/a" if any(lg not in f or pd.isna(f[lg]) for lg in leagues) else ("yes" if cond() else "no")
    chl = ["OHL", "WHL", "QMJHL"]
    return pd.DataFrame([
        ("KHL above SHL, Liiga and NL", ok(lambda: f["KHL"] > max(f["SHL"], f["Liiga"], f["NL"]), "KHL", "SHL", "Liiga", "NL")),
        ("SHL, Liiga and NL above AHL", ok(lambda: min(f["SHL"], f["Liiga"], f["NL"]) > f["AHL"], "SHL", "Liiga", "NL", "AHL")),
        ("AHL above NCAA", ok(lambda: f["AHL"] > f["NCAA"], "AHL", "NCAA")),
        ("NCAA above OHL, WHL, QMJHL", ok(lambda: f["NCAA"] > max(f[x] for x in chl), "NCAA", *chl)),
        ("NCAA above USHL", ok(lambda: f["NCAA"] > f["USHL"], "NCAA", "USHL")),
        ("OHL, WHL, QMJHL within 25% of each other", ok(lambda: max(f[x] for x in chl) / min(f[x] for x in chl) < 1.25, *chl)),
        ("QMJHL lowest of the three CHL leagues", ok(lambda: f["QMJHL"] < min(f["OHL"], f["WHL"]), *chl)),
    ], columns=["check", "result"])
