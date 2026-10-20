# Hockey WAR — context for Claude

## Project
Repo: github.com/ryanc2525/hockey-war. Single-season WAR model for NHL skaters (v0.4)
plus Baseball Savant-style player cards ("Almanac"). A projection side ("Forecast") comes later.
- src/war_model.py: WAR model
- src/player_card.py: percentiles and draw_card (returns a matplotlib figure)
- src/edge_data.py: NHL EDGE and player profile downloads
- data/cards_2025-26.csv: full 2025-26 card table, one row per player, every column draw_card needs
- notebooks/01–08: how it was built, step by step
- app.py: Streamlit app. Search a player and see his 2025-26 card. Reads
  app_data/cards_2025-26.csv only; never recomputes WAR or calls an NHL API.
- app_data/cards_2025-26.csv: slim public card table (committed), written by notebook 09 via
  player_card.export_app_table. Only columns the card draws; no raw MoneyPuck situation stats.
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
   (gitignored); only the slim app_data/ table is approved for the public repo.
