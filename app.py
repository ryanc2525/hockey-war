"""
Hockey WAR app: two pages.

- Almanac (app_pages/almanac.py): Baseball Savant-style player cards, 2021-22 to 2025-26. The default page,
  served at the site's root URL, so the live site opens on it exactly as before.
- Forecast (app_pages/forecast.py): 2026-27 OVR ratings and projections. It needs
  data/forecast_table_2026-27.csv (notebook 20, gitignored), so it only works locally for now; without the
  file it shows a short message and the Almanac keeps working.

    /opt/anaconda3/envs/hockey/bin/python -m streamlit run app.py

This file only sets up the page and the navigation. Streamlit runs it on every click, then runs the page
that's open.
"""

import streamlit as st

# Called once, here, for the whole app (the page files never call it). "centered" is the Almanac's layout.
st.set_page_config(page_title="Hockey WAR", layout="centered")

pages = [
    st.Page("app_pages/almanac.py", title="Almanac", default=True),   # default = shown at the root URL
    st.Page("app_pages/forecast.py", title="Forecast"),
]
st.navigation(pages, position="top").run()     # a menu across the top; runs the page that's open
