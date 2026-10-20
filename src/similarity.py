"""Similarity engine: z-score each feature within position group, then cosine similarity."""
import numpy as np
import pandas as pd

FEATURES = ["passes", "pass_completion", "pass_length", "progressive_passes", "progressive_carries", "final_third_passes",
            "box_passes", "crosses", "xt_pass", "xt_carry", "pass_under_pressure_share", "dribbles", "dribble_success",
            "np_shots", "np_xg", "box_touches", "pressures", "pressures_att_third", "tackles", "interceptions",
            "recoveries", "clearances", "blocks", "aerials_lost", "fouls_won", "turnovers", "avg_x"]

def standardise(df: pd.DataFrame, min_minutes=900) -> pd.DataFrame:
    d = df[(df.minutes >= min_minutes) & (df.pos_group != "Goalkeeper")].copy()
    z = d.groupby("pos_group")[FEATURES].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return d, z.clip(-3, 3)

def similar(name: str, df: pd.DataFrame, z: pd.DataFrame, n=10, same_position=True, max_league_position=None,
            exclude_team=True, leagues=None):
    idx = df.index[df.player.str.contains(name, case=False, regex=False)]
    if len(idx) == 0:
        raise ValueError(f"No player matching {name!r} with enough minutes")
    i = idx[0]
    v = z.loc[i].values
    Z = z.values
    sim = Z @ v / (np.linalg.norm(Z, axis=1) * np.linalg.norm(v) + 1e-9)
    out = df.assign(similarity=sim)
    out = out[out.index != i]
    if same_position: out = out[out.pos_group == df.loc[i, "pos_group"]]
    if exclude_team: out = out[out.team != df.loc[i, "team"]]
    if max_league_position is not None: out = out[out.league_position >= max_league_position]
    if leagues: out = out[out.league.isin(leagues)]
    return df.loc[i], out.sort_values("similarity", ascending=False).head(n)
