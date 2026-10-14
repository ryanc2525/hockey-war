# Hockey WAR: Wins Above Replacement for NHL Skaters

**Status: in progress (v2 rebuild)**

## The question

Can one number capture how much a hockey player is worth to his team, combining offense, defense, and discipline, and express it in wins?

Baseball has WAR (Wins Above Replacement). This project builds a simplified hockey version from public data, then checks it against established public models and against team results.

## Background

This started as a class project at Michigan State (CMSE 201, Spring 2025). The original version is kept in `archive/` for reference. Looking back at it, it had real problems: made-up weights, a "replacement level" that was really the average of good players, double-counted stats, a couple of calculation bugs, and an output that wasn't actually measured in wins. Version 2 rebuilds the model to fix those issues.

## Plan for v2

1. **Data:** 2025-26 season skater data from MoneyPuck
2. **Components, measured in goals above replacement:**
   - Even-strength offense
   - Even-strength defense, relative to the team (to reduce team-quality bias)
   - Power play
   - Penalties drawn minus taken
   - Faceoffs (centers)
3. **Replacement level:** based on players outside each team's regular lineup by ice time
4. **Convert goals to wins**
5. **Validate:** compare results to a public WAR model, and test whether team WAR totals track with team standings
6. **App:** Streamlit dashboard to look up any player's WAR and its breakdown

## Data

Data comes from [MoneyPuck.com](https://moneypuck.com/data.htm). Raw data is not included in this repo. To run the notebooks, download the season-level skaters CSV from MoneyPuck and place it in `data/`.

**Credit:** All player data is from MoneyPuck.com.

## Tools

Python (pandas, NumPy, matplotlib), Jupyter, Streamlit. Built with AI assistance (Claude).

## Repo structure

```
hockey-war/
├── archive/      original class version (v1)
├── data/         put the MoneyPuck CSV here (not tracked)
├── notebooks/    v2 analysis
└── README.md
```

## Author

Ryan Campbell, Michigan State University '26 (B.S. Psychology, minors in Quantitative Data Analytics and Spanish)
