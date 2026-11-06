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
  - app_pages/forecast.py: 2026-27 OVR and projections. Reads data/forecast_table_2026-27.csv (notebook 20,
    gitignored), so it's local only; without the file it shows a message and stops. HOCKEY_FORECAST_TABLE
    overrides the path.
- forecast.current_projections(): the 2026-27 projection for every player (nb18 skaters, nb21 goalies and
  goalie prospects, nb19 skater prospects). ratings.py: OVR scale, skater and goalie anchors, LOCKED.
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
