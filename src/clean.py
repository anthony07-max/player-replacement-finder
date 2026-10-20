"""Build per-player season profiles (per 90 and rate stats) from compact StatsBomb events."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xt import fit_xt, value_moves

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUE = {"premier_league_2015_16": "Premier League", "la_liga_2015_16": "La Liga",
          "serie_a_2015_16": "Serie A", "ligue_1_2015_16": "Ligue 1"}

def minutes(ev):
    rows = []
    for mid, g in ev.groupby("match_id"):
        end = g.minute.max() + 1; on, pos = {}, {}
        for r in g[g.type == "Starting XI"].itertuples():
            for pid, name, p in json.loads(r.extra):
                on[pid] = [name, r.team, 0, end]; pos[pid] = p
        for r in g[g.type == "Substitution"].itertuples():
            rid, rname = json.loads(r.extra)
            if r.player_id in on: on[r.player_id][3] = r.minute
            on[rid] = [rname, r.team, r.minute, end]; pos.setdefault(rid, r.position)
        for pid, (name, team, a, b) in on.items():
            rows.append({"player_id": pid, "player": name, "team": team, "minutes": max(b - a, 0), "position": pos.get(pid)})
    return pd.DataFrame(rows)

def pos_group(p):
    p = p or ""
    if "Goalkeeper" in p: return "Goalkeeper"
    if "Center Back" in p: return "Centre-back"
    if "Back" in p or "Wing Back" in p: return "Full-back"
    if "Defensive Midfield" in p or p in ("Right Center Midfield", "Left Center Midfield", "Center Midfield"): return "Central midfielder"
    if "Attacking Midfield" in p or "Wing" in p or "Right Midfield" in p or "Left Midfield" in p: return "Winger / attacking mid"
    if "Forward" in p or "Striker" in p: return "Striker"
    return "Other"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pd.read_parquet(f).assign(league=LEAGUE[f.stem]) for f in RAW.glob("*.parquet")], ignore_index=True)
    moves = ev[ev.type.isin(["Pass", "Carry"]) & ev.x.notna() & ev.end_x.notna()].copy()
    moves["success"] = moves.outcome.isna()
    shots = ev[(ev.type == "Shot") & ev.x.notna()].copy(); shots["goal"] = shots.outcome == "Goal"
    grid, _ = fit_xt(moves, shots[shots.subtype != "Penalty"])
    moves["xt"] = value_moves(moves, grid)
    dist = lambda x, y: np.hypot(120 - x, 40 - y)
    moves["progressive"] = moves.success & (dist(moves.end_x, moves.end_y) <= 0.75 * dist(moves.x, moves.y)) & (moves.end_x - moves.x >= 5)
    moves["length"] = np.hypot(moves.end_x - moves.x, moves.end_y - moves.y)
    passes, carries = moves[moves.type == "Pass"], moves[moves.type == "Carry"]
    op = passes[passes.subtype.isna()]  # open-play passes (set pieces have a pass type)

    g = lambda df: df.groupby("player_id")
    f = pd.DataFrame({
        "passes": g(op).size(), "pass_completion": g(op).success.mean(),
        "pass_length": g(op).length.mean(),
        "progressive_passes": g(passes).progressive.sum(), "progressive_carries": g(carries).progressive.sum(),
        "final_third_passes": g(op[op.success & (op.end_x > 80)]).size(),
        "box_passes": g(op[op.success & (op.end_x > 102) & op.end_y.between(18, 62)]).size(),
        "crosses": g(passes[passes.cross]).size(),
        "xt_pass": g(passes).xt.sum(), "xt_carry": g(carries).xt.sum(),
        "pass_under_pressure_share": g(op).under_pressure.mean(),
        "dribbles": g(ev[ev.type == "Dribble"]).size(),
        "dribble_success": g(ev[ev.type == "Dribble"]).outcome.apply(lambda s: (s == "Complete").mean()),
        "np_shots": g(shots[shots.subtype != "Penalty"]).size(), "np_xg": g(shots[shots.subtype != "Penalty"]).xg.sum(),
        "np_goals": g(shots[(shots.subtype != "Penalty") & shots.goal]).size(),
        "box_touches": g(ev[(ev.type == "Ball Receipt*") & (ev.x > 102) & ev.y.between(18, 62)]).size(),
        "pressures": g(ev[ev.type == "Pressure"]).size(),
        "pressures_att_third": g(ev[(ev.type == "Pressure") & (ev.x > 80)]).size(),
        "tackles": g(ev[(ev.type == "Duel") & (ev.subtype == "Tackle")]).size(),
        "interceptions": g(ev[ev.type == "Interception"]).size(),
        "recoveries": g(ev[ev.type == "Ball Recovery"]).size(),
        "clearances": g(ev[ev.type == "Clearance"]).size(), "blocks": g(ev[ev.type == "Block"]).size(),
        "aerials_lost": g(ev[(ev.type == "Duel") & (ev.subtype == "Aerial Lost")]).size(),
        "fouls_won": g(ev[ev.type == "Foul Won"]).size(),
        "turnovers": g(ev[ev.type.isin(["Dispossessed", "Miscontrol"])]).size(),
        "avg_x": g(ev[ev.type.isin(["Pass", "Carry", "Ball Receipt*"])]).x.mean(),
    })
    mins = minutes(ev)
    prof = mins.groupby("player_id").agg(player=("player", "first"), minutes=("minutes", "sum"), apps=("minutes", "size"),
                                         team=("team", lambda s: s.mode().iat[0]),
                                         position=("position", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None))
    team_league = ev.groupby("team").league.first()
    prof["player"] = prof.player.str.replace("''", "'", regex=False)
    prof["league"] = prof.team.map(team_league)
    prof["pos_group"] = prof.position.map(pos_group)
    prof = prof.join(f).fillna(0)
    rate_cols = ["pass_completion", "pass_length", "pass_under_pressure_share", "dribble_success", "avg_x"]
    count_cols = [c for c in f.columns if c not in rate_cols]
    for c in count_cols:
        prof[c] = prof[c] / prof.minutes.clip(lower=1) * 90
    # League table position as a rough budget/standing proxy (real results from the match files)
    tables = []
    for mf in RAW.glob("*_matches.csv"):
        m = pd.read_csv(mf)
        pts = {}
        for h, a, hs, as_ in m[["home_team.home_team_name", "away_team.away_team_name", "home_score", "away_score"]].values:
            pts.setdefault(h, 0); pts.setdefault(a, 0)
            pts[h] += 3 if hs > as_ else 1 if hs == as_ else 0; pts[a] += 3 if as_ > hs else 1 if hs == as_ else 0
        t = pd.Series(pts).sort_values(ascending=False)
        tables.append(pd.DataFrame({"team": t.index, "league_position": range(1, len(t) + 1), "points": t.values}))
    tab = pd.concat(tables).set_index("team")
    prof = prof.join(tab, on="team")
    prof.reset_index().to_csv(OUT / "player_profiles.csv", index=False)
    print(len(prof), "players;", (prof.minutes >= 900).sum(), "with 900+ minutes")
    print(prof.pos_group.value_counts())

if __name__ == "__main__":
    main()
