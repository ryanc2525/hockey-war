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
- app_pages/simulator.py: the Forecast page's Simulator tab (local only): play one season (standings with seeds,
  leaders, team stat lines, Simulate again) or run 10,000 (odds, player ranges). Reads the bundle +
  data/trades_2026-27.csv, imports season.py only after the files check (the live site has no scipy).
  HOCKEY_SIMULATOR_DIR overrides the bundle path. Streamlit doesn't reload this module on edit: restart the app.
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
4. Never publish or copy raw MoneyPuck data anywhere outside this folder. data/ stays private
   (gitignored); only the slim app_data/ table is approved for the public repo. Ryan allows
   private, temporary analysis copies in Claude's cloud workspace (never published).
