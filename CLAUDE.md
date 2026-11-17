# Hockey WAR — context for Claude

## Project
Repo: github.com/ryanc2525/hockey-war. Single-season WAR model for NHL skaters (v0.4)
plus Baseball Savant-style player cards ("Almanac"). A projection side ("Forecast") comes later.
- src/war_model.py: WAR model
- src/player_card.py: percentiles and draw_card (returns a matplotlib figure)
- src/edge_data.py: NHL EDGE and player profile downloads
- data/cards_<label>.csv: full card table per season (gitignored). data/cards_2025-26.csv is the
  fixed reference for notebook 10's regression check; never overwrite it.
- notebooks/01–10: how it was built, step by step
- app.py: Streamlit app, two pages (st.navigation; st.set_page_config only here). Never recomputes WAR
  or calls an NHL API.
  - app_pages/almanac.py (default page): search a player, pick a season, see his card. Reads app_data/ only
    (2021-22 to 2025-26).
  - app_pages/forecast.py: tabs Players (OVR view + leaderboard), Teams, Simulator. Reads
    data/forecast_table_2026-27.csv (notebook 20, gitignored), so it's local only; without the file it shows a
    message and stops. HOCKEY_FORECAST_TABLE overrides the path. The Simulator tab is app_pages/simulator.py
    (a module now, not a page: show_simulator()).
- forecast.current_projections(): the 2026-27 projection for every player (nb18 skaters, nb21 goalies and
  goalie prospects, nb19 skater prospects, nb22 new players). ratings.py: OVR scale, skater and goalie
  anchors, LOCKED.
- Notebook 22: 2026-27 opening-night teams and statuses (data/rosters_2026-27.csv) from an NHL API snapshot in
  data/current_raw/ (gitignored). Hand-kept, never overwritten: data/trades_2026-27.csv (in-season trades; the
  baseline undoes them, the simulator applies them) and data/roster_overrides.csv (overrides win).
  Notebook 20 must run after 22. The 2026-27 schedule has 84 games.
- src/lineups.py + notebook 23 (simulator step 1): who dresses (12 F with 4 natural centers, 6 D, 2 G) and
  minutes by slot. Coach score = w x usage + (1 - w) x projection; the backtest picked w, the rookie rule and
  the usage lookback per group (settings live in the notebook). Writes data/lineups_2026-27.csv (baseline: no
  trades applied, disputes play) and adds the in-season trades T2-T6 to data/trades_2026-27.csv if missing.
  Notebook 23 runs after 22 and reads data/current_raw/standings_2026-11-06.json (context only).
- src/simulate.py + notebook 24 (simulator step 2): game model (Poisson "normal" goals + a late pulled-goalie
  table + OT/SO tilt), vectorized notebook 17 draws (pools centered), seeded season engine. Backtest vs. real
  standings: alone it does NOT beat "last season pulled 1/3 to average"; a 40/60 blend with it does. lineups.py
  options (defaults = notebook 23): pick_lineup(tiered=True) for call-up tiers, assign_minutes(own_weight=...)
  for blended minutes. Past results: data/schedule_raw/ (NHL API club schedules 2008-09 to 2020-21, gitignored);
  2021-22 on from the shot data.
- Notebook 25 (simulator step 3a): coefficient fits {all seasons, 2011-12 on, recency half-life 4} x team
  ingredient {none, level blend 40/60, residual = last season's real minus simulated points, shrunk 0.4}, every
  season scored on fits from earlier seasons only. STEP 3 DEFAULT (Ryan's choice): all seasons + residual
  (lambda 0.4). The level blend scored slightly better (13.02 vs. 13.23 test, within noise) but keeps only 40% of
  a roster move; the residual keeps it all. simulate.py options (defaults = notebook 24): fit_game_model(weights=...),
  simulate_season(bonus={team: normal goals per game}).
- src/season.py + notebook 26 (simulator step 3): the 2026-27 runner. run_scenario(bundle, start_date, trade_ids,
  sit_out, n_sims, seed): real schedule, real results before start_date locked in, trades default to those dated
  on or before it, lineups fixed at the start date (full health), backup goalie starts back-to-back second
  nights + evenly spread games to 30 of 84, league-wide draws (same player luck across scenarios), team bonus,
  playoffs = top 3 per division + 2 wild cards (tiebreaks points, RW, ROW, random). simulate_games(goalie_idx=...)
  is add-only. Notebook 26 rebuilds notebook 25's settings (asserted) and saves the bundle
  data/simulator_2026-27/ (players.csv, schedule.csv, settings.json; gitignored). 2026-27 schedules live in
  data/schedule_raw/2026-27/ (a subfolder, so notebooks 24-25's schedule_games glob doesn't see them);
  REFRESH = True in notebook 26 re-downloads them (33 NHL calls). "Today" = the schedule files' download stamp.
- src/scoring.py + notebook 27 (stat lines): goals/60 and assists/60 projected like notebook 15 (3 seasons,
  age on, 500 regression minutes, via forecast.project on a rate table with gar = count); rookies/prospects:
  NHLe x 60 / rookie minutes, shrunk 0.5 toward the rookie average. Scoring moves with the notebook 17 draw:
  m = exp(0.18 z) / pool mean (k fit on training seasons for 80% coverage). Team goals split season-level
  (multinomial) by g60 x m x minutes, assists by a60 x m x minutes at the real 0/1/2 mix. Goalies get the
  decision and GA (no empty-net or shootout goals) of their starts; no save %. run_scenario(stat_lines=True)
  uses its own random stream (standings unchanged); season_lines() = real so far (boxscores) + simulated.
  Notebook 27 adds scoring.csv, scoring.json, real_games.csv to the bundle; boxscores in
  data/schedule_raw/2026-27/boxscores/ (only new games download), club totals in data/current_raw/club_stats/
  (cross-check). After a notebook 26 REFRESH, rerun notebook 27.
- src/inseason.py + notebook 28 (in-season learning, level 1): per-player importance weighting (particle filter).
  Prior = 3,000 notebook 17 guesses z per player; evidence = goals/assists so far (Poisson at rate x exp(k z) x
  minutes), goalie GSAx so far (as WAR/60, goalie luck fit), optional skater WAR so far. Two versions: "whole"
  (z carried) and "split" (talent t = z x sqrt(a) updated, fresh luck b / rest minutes for the remaining games).
  DEFAULT: whole (picked on training, a near tie; Ryan kept it). Adopted: skater G/A update + team bonus update;
  goalie update not adopted (hurt team predictions).
  Team bonus += g / (g + g0) x (points gap per game) / beta, g0 measured. Adopted only if it beats preseason
  AND pace on test seasons (backtest after 10/20/40 games, 2013-14 to 2017-18 train, 2018-19 + 2021-22 to
  2025-26 test; goalies 2021-25 only, from the raw shot zip's goalieIdForShot). Skater WAR evidence is never
  the default (not testable). run_scenario(updated={"playerId", "z", "bonus"}) is add-only. Past boxscores
  (first 40 games per team) in data/schedule_raw/boxscores_past/; forecast log in data/forecast_log/
  (updated_projections_<date>.csv, both versions).
- src/refresh.py + notebook 29 (weekly refresh): run once a week. Downloads only new daily scores
  (/v1/score/<date> -> data/schedule_raw/2026-27/days/, a day saved only when all its games are final) and new
  boxscores; checks (every final game has a boxscore, boxscore goals vs. scores, MoneyPuck file dates warn
  after 7 days, roster drift: a different NHL team BLOCKS the save until logged in trades_2026-27.csv (waiver
  claim = one-player row, id W1...) or acknowledged in data/drift_ack_2026-27.csv; no-roster call-ups warn only).
  Reruns notebook 28's update with FROZEN settings (data/forecast_log/settings.json, saved once; g0 and beta at
  full precision) and the reality scenario (start today, trades dated <= evidence date, sit-outs from
  data/sit_out_2026-27.csv). Evidence = through yesterday; HOCKEY_TODAY overrides "today" for tests. Postponed
  games are moved into the rest of the season (84 checked). Saves data/forecast_log/<evidence date>/ (players.csv,
  teams.csv, metadata.json with commit, uncommitted flag, sha256 of src modules used, backfilled flag; evidence/),
  all or nothing, never overwritten. Entry 2026-11-09 = notebook 28's file converted (numbers unchanged, rest-of-
  season and team columns recomputed and listed in metadata). The bundle files are never rewritten.
- Update cache (notebook 29's last cell, refresh.build_update_cache): per entry, data/simulator_2026-27/updated/
  <evidence date>/ (draws.npz = every pool player's 4,000 rest-of-season draws, float64 compressed; bonus.json;
  evidence/ copied from the entry; meta.json with the bundle sha256). Rebuilt from the entry's evidence with the
  frozen settings and saved only if it reproduces players.csv to 1e-9 and teams.csv exactly (default-version
  columns; the skater WAR version isn't cached). Never writes in data/forecast_log/. "whole" draws don't depend
  on minutes, so the app reuses them for any lineup; the team bonus stays the entry's. 2026-11-10 built.
- Public Forecast snapshot (notebook 29's cell 21, src/public_forecast.py): per entry, app_data/forecast/
  <evidence date>/ (players.csv, teams.csv, meta.json; TRACKED, about 195 KB a week). Built only from the log
  entry + data/forecast_table_2026-27.csv (preseason OVR, position, age); never changes the entry. gp/g/a come
  from the entry's evidence/real_games.csv (includes goalies; equal to the entry for skaters). season_g/a = real
  + rest (10th/90th shift by the real count). Teams' preseason vs. updated = team_bonus_pre/upd (goals per
  game). Checked in memory before writing: allowlist + raw-stat-word guard, and every number equal to the entry
  (copied columns bit for bit, built columns to their formula). Never overwritten. 2026-11-09, -10, -15 exported.
- src/franchise.py + notebook 30 (franchise-mode engine, no UI yet): load_inputs (the simulator switch's rules),
  lineup overrides (table team, slot, playerId; slots F1-F12, D1-D6, G1, G2; slot number = rank; a team lists all
  20; optional goalie_share {team: share} rebuilds that team's backup games), game_range (next N games for a team,
  an end date, or the rest of the season; every team plays the window; seeds/odds only for a full season),
  story (one seeded season: game log, record, standings, lines two ways = range only and season to date via
  season_lines, draws table for notebook 31) and projection (10,000). Minutes rule "moved" (Ryan's, default):
  players who changed LINE (F1-F3 = line 1 ... F10-F12) or PAIR (D1-D2, D3-D4, D5-D6), or weren't dressed, get
  100% slot minutes, others keep the 75/25 blend, then F/D totals rescaled; "blend" and "slot" kept for comparison.
  Minutes follow a "minutes rank" (notebook 32): unmoved players keep their baseline rank, moved ones take the
  unit's remaining ranks in slot order, so reordering inside a line or pair changes nobody's minutes (all rules). No chemistry. Errors block (slots, duplicates, not in the team's pool, goalie <->
  skater, share outside 0-1); warnings don't (< 4 centers, out of position, call-ups). Add-only options:
  run_scenario(lineups=, keep_games=, keep_z=, update_shift=) and assign_minutes(own_weight_by_player=); defaults
  checked bit for bit against the committed versions. A story passes its seed as update_shift, and so does the
  Simulator tab for one season with the update on (so "Simulate again" re-rolls updated players; 10,000 unchanged). Reserved path for saved overrides:
  data/franchise_lineups_2026-27.csv (ignored; a data/ subfolder would not be). Notebook 30 writes nothing.
- src/progression.py + notebook 31 (season-end OVR progression): the new rating = the same projection run one season
  later with the simulated 2026-27 season added to history (dressed players: minutes = real so far + simulated,
  WAR/60 = the story's season_war60 for the whole season; real WAR so far isn't used). load_engine builds once
  (about 20 s warm): nb18's prior (development.nb18_skater_model, new) + D shift, nb21's goalie model + shift, nb19's
  models + NHLe fill, the aging curve, all FROZEN. project_season(E, target, extra) = NHL path (season in the last
  three) or prospect path (same pre-NHL stats, nothing after 2025-26; project_skater_prospects(season=) is add-only),
  then nb20's rating rule. reproduce_today: exact for all 2,503 players. story_progression (full-season story only):
  change + aging part (season at projection: age, experience, window roll, prior) + season part; flags for no NHL
  minutes, prospects (no new season), real games only, left the projection. progression_many uses straight-line maps
  (lineups fixed -> rating linear in season WAR/60; checked vs. direct) and franchise.play(keep_z=True) (add-only).
  Calibration: real_ratings/real_changes on real history (same frozen engine), real_prospect_changes (draft lists from
  2019). Prospects with no NHL minutes (Ryan's rule, adopted): OVR before + the real average change of a prospect
  year by age group (18-19, 20-21, 22-23, 24+) and position, from 2024-25 and 2025-26 (E["expected_dev"], built in
  load_engine), flagged "expected development, not simulated"; rating WAR/60 follows via the inverse scale (D: their
  d_rank_prior). The engine's own number stays in engine_ovr_after. Calibration verdict: spread matches real history,
  average about 0.4 OVR too high (season part; not tuned). Reserved path data/progression_2026-27.csv (ignored).
  Notebook 31 writes nothing.
- src/boxscore.py + notebook 32 (box scores, engine only): split_story(story, seed) = per-game player lines and goal
  list for a one-season story: each player's range goals shuffled into his team's games (both totals exact; same
  statistics as goal by goal, fixed lineups), assists 0-2 per goal from notebook 27's mix given the team total,
  never the scorer, two different players (random fill, max-flow fallback, "impossible" handling counted); starting
  goalies' GA (no SO/empty-net) and decision; real games before the start use real_games (no goal list).
  run_scenario keeps home_en/away_en per game with keep_games + stat_lines (add-only). box_score(split, game_id).
  simulate.game_probabilities(edge, settings) (add-only): exact home/away/OT chances (Poisson-built Skellam + late
  table + OT logistic). win_probabilities(inputs, team, ...) = chances per game averaged over the same 10,000 draws
  (+ "at projection"); check_against_runs. calendar(games, team, month, probs); season_games for projection mode.
  No periods, goal times, shots, +/-. Reserved path data/box_scores_2026-27.csv (ignored). Notebook 32 writes nothing.
- app_pages/simulator.py: the Forecast page's Simulator tab (local only): play one season (standings with seeds,
  leaders, team stat lines, Simulate again) or run 10,000 (odds, player ranges). Reads the bundle +
  data/trades_2026-27.csv, imports season.py only after the files check (the live site has no scipy).
  HOCKEY_SIMULATOR_DIR overrides the bundle path. Streamlit doesn't reload this module on edit: restart the app.
  Switch "Use 2026-27 results so far" (newest update cache, refresh.load_update_cache; refused if the bundle
  changed): enabled only for start >= the entry's start (date max = that start), default ON; ON = entry's real
  results + updated draws + updated bonus. OFF at dates <= the bundle's download day = the old code path exactly;
  OFF after it = the entry's real results with preseason ratings. No cache: switch hidden with a note.
- app_data/cards.csv: slim public card table (committed), one row per player per season, written
  by notebook 10 via player_card.export_app_table(by_season=True). Only columns the card draws;
  no raw MoneyPuck situation stats. (Notebook 09 wrote the earlier 2025-26-only version.)
- player_card.build_season_cards: full card table for one season (same steps as notebook 08).
- season = start year (MoneyPuck convention): 2025 = 2025-26. Display season_label, never the integer.
- player_card.HIDDEN_ROWS: card rows that are computed but not drawn (currently oz_pct_ev).
- requirements.txt: what app.py needs for Streamlit Community Cloud, pinned to the hockey env.

## Environment
Mac. Always use the conda env Python: /opt/anaconda3/envs/hockey/bin/python
Run the app: /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py

## Working with Ryan
Ryan is a beginner building a portfolio project. Explain what you're doing and why in
plain language, one step at a time.

## Rules
1. Read README.md and the relevant src/ files first, and explain the plan before writing code.
2. Ask before installing anything.
3. Never delete files, never run git push, never commit. Ryan reviews and commits in VS Code.
4. Never publish raw MoneyPuck data (season or shot-level columns such as xG, shots, Corsi/Fenwick,
   on-ice stats, TOI splits) or anything in data/; data/ stays private (gitignored). Public (tracked)
   data is only app_data/: the slim card table (cards.csv, shot_grid.csv, shot_totals.csv, as before)
   and the Forecast snapshot app_data/forecast/<evidence date>/, built by src/public_forecast.py
   (export_public), whose check fails on any other column. Ryan's decision: derived Forecast outputs
   (projections, OVR, team odds) and real NHL game stats may be public. Forecast allowlist:
   - players.csv: playerId, name, team, position, age, ovr, gp, g, a, rest_g, rest_g_p10, rest_g_p90,
     rest_a, rest_a_p10, rest_a_p90, rest_p, rest_p_p10, rest_p_p90, season_g, season_g_p10,
     season_g_p90, season_a, season_a_p10, season_a_p90, season_p, season_p_p10, season_p_p90,
     pre_p60, upd_p60
   - teams.csv: team, real_gp, real_points, points, points_p10, points_p90, playoff_pct, division_pct,
     presidents_pct, team_bonus_pre, team_bonus_upd
   - meta.json: evidence_through, run_date, backfilled, sit_out, trades_applied, settings, git_commit,
     exported, ovr_note
   Ryan allows private, temporary analysis copies in Claude's cloud workspace (never published).
