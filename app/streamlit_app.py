from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from similarity import standardise, similar

st.set_page_config(page_title="Player replacement finder", layout="wide")
st.title("Player replacement finder")
st.caption("Find statistically similar players. StatsBomb open data, 2015-16 Premier League, La Liga, Serie A and Ligue 1. By Anthony Adusei-Poku (@anthony07-max)")
prof = pd.read_csv(ROOT / "data" / "processed" / "player_profiles.csv")
pct = pd.read_csv(ROOT / "data" / "processed" / "percentiles.csv")

c1, c2, c3 = st.columns(3)
mm = c1.slider("Minimum minutes", 450, 2500, 900, 50)
df, z = standardise(prof, mm)
names = sorted(df.player)
target = c2.selectbox("Departing player", names, index=names.index("N'Golo Kanté") if "N'Golo Kanté" in names else 0)
pos_cut = c3.slider("Only clubs that finished at or below position", 1, 20, 1, help="A rough budget proxy: lower-placed clubs")
leagues = st.multiselect("Leagues", sorted(df.league.unique()), default=sorted(df.league.unique()))
t, res = similar(target, df, z, n=15, max_league_position=pos_cut, leagues=leagues)
st.subheader(f"{t.player} ({t.team}, {t.pos_group})")
st.dataframe(res[["player", "team", "league", "league_position", "minutes", "similarity"]].round(3), hide_index=True, use_container_width=True)

feats = ["progressive_passes", "progressive_carries", "xt_pass", "dribbles", "np_xg", "pressures", "tackles", "interceptions", "recoveries", "pass_completion", "box_touches"]
if not res.empty:
    pick = st.selectbox("Compare with", res.player.tolist())
    fig = go.Figure()
    for name in [t.player, pick]:
        row = pct[pct.player == name][feats].iloc[0]
        fig.add_trace(go.Scatterpolar(r=row.values.tolist() + [row.values[0]], theta=feats + [feats[0]], fill="toself", name=name))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title="Percentile within position group")
    st.plotly_chart(fig, use_container_width=True)
