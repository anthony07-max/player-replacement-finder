"""Role clustering and replacement case studies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from similarity import FEATURES, standardise, similar

ROOT = Path(__file__).resolve().parents[1]
P, C = ROOT / "data" / "processed", ROOT / "charts"
C.mkdir(exist_ok=True)

def radar(ax, labels, series, title):
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    for name, vals, col in series:
        v = list(vals) + [vals[0]]
        ax.plot(ang, v, color=col, lw=2, label=name); ax.fill(ang, v, color=col, alpha=.15)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels([]); ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10); ax.legend(loc="lower center", bbox_to_anchor=(.5, -.25), fontsize=8, ncol=2)

def main():
    prof = pd.read_csv(P / "player_profiles.csv")
    df, z = standardise(prof)
    pct = df.groupby("pos_group")[FEATURES].rank(pct=True) * 100

    # --- Role clustering on z-scores *across* all outfield players (raw features standardised globally) ---
    G = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    pcs = PCA(n_components=8, random_state=0).fit(G)
    X = pcs.transform(G)
    sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)) for k in range(5, 11)}
    k = 8
    km = KMeans(k, n_init=20, random_state=0).fit(X)
    df["cluster"] = km.labels_
    cent = G.groupby(df.cluster).mean()
    names = {}
    for c in range(k):
        top = cent.loc[c].sort_values(ascending=False).index[:3].tolist()
        names[c] = f"C{c}: high " + ", ".join(top)
    df["role"] = df.cluster.map(names)
    roles = df.groupby("role").agg(players=("player", "size"), main_position=("pos_group", lambda s: s.mode().iat[0]),
                                   examples=("player", lambda s: ", ".join(s.iloc[np.argsort(-df.loc[s.index, "minutes"].values)[:4]])))
    roles.to_csv(P / "roles.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(k):
        m = df.cluster == c
        ax.scatter(X[m, 0], X[m, 1], s=10, alpha=.6, label=names[c][:60])
    for nm in ["N'Golo Kanté", "Riyad Mahrez", "Dimitri Payet", "Lionel", "Jamie Vardy", "Virgil van Dijk"]:
        h = df.player.str.contains(nm, regex=False).values
        if h.any(): ax.annotate(df.player[h].iat[0].split()[-1] if nm != "Lionel" else "Messi", X[h][0, :2], fontsize=8, weight="bold")
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=6, loc="best"); ax.set_title("Data-driven player roles (KMeans on 27 per-90 features, 4 leagues 2015-16)")
    fig.tight_layout(); fig.savefig(C / "01_player_roles.png", dpi=150); plt.close()

    # --- Case studies ---
    cases = [("N'Golo Kanté", "Left Leicester for Chelsea, July 2016", None),
             ("Riyad Mahrez", "Leicester's PFA Player of the Year", None),
             ("Dimitri Payet", "Left West Ham for Marseille, Jan 2017", None),
             ("N'Golo Kanté", "Budget version: clubs finishing 10th or lower", 10)]
    out, plots = {}, []
    for name, why, maxpos in cases:
        target, res = similar(name, df, z, n=10, max_league_position=maxpos)
        key = target.player + (" (budget)" if maxpos else "")
        out[key] = {"why": why, "team": target.team, "pos_group": target.pos_group,
                    "matches": res[["player", "team", "league", "league_position", "minutes", "similarity"]].round(3).to_dict("records")}
        if not maxpos: plots.append((target, res.iloc[0]))
        print("\n", key, "-", why); print(res[["player", "team", "league", "league_position", "similarity"]].round(3).to_string(index=False))
    # Real outcome check: Leicester signed Nampalys Mendy (Nice) to replace Kanté
    t, full = similar("N'Golo Kanté", df, z, n=500)
    mendy = full.reset_index(drop=True)
    mrank = mendy.index[mendy.player.str.contains("Mendy", regex=False) & (mendy.team == "OGC Nice")]
    mendy_rank = int(mrank[0] + 1) if len(mrank) else None
    n_cm = int(len(full))
    print("Nampalys Mendy rank among Kanté-similar central midfielders:", mendy_rank, "of", n_cm)

    radar_feats = ["progressive_passes", "progressive_carries", "xt_pass", "dribbles", "np_xg", "pressures", "tackles",
                   "interceptions", "recoveries", "pass_completion", "turnovers", "box_touches"]
    fig, axs = plt.subplots(1, len(plots), figsize=(6 * len(plots), 6), subplot_kw=dict(polar=True))
    for ax, (t, best) in zip(np.atleast_1d(axs), plots):
        radar(ax, radar_feats, [(t.player, pct.loc[t.name, radar_feats].values, "#1d3557"),
                                (f"{best.player} ({best.team})", pct.loc[best.name, radar_feats].values, "#e63946")],
              f"{t.player} vs closest match\n(percentiles within position)")
    fig.tight_layout(); fig.savefig(C / "02_case_study_radars.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    r = full.head(15)[::-1]
    ax.barh(r.player + " (" + r.team + ")", r.similarity, color=["#e63946" if "Mendy" in p else "#457b9d" for p in r.player])
    ax.set_xlim(r.similarity.min() - .05, 1); ax.set_title("Most similar central midfielders to N'Golo Kanté (2015-16)")
    fig.tight_layout(); fig.savefig(C / "03_kante_replacements.png", dpi=150); plt.close()

    df.drop(columns=[]).to_csv(P / "players_scored.csv", index=False)
    pct.assign(player=df.player, team=df.team).to_csv(P / "percentiles.csv", index=False)
    kf = {"players_profiled": int(len(df)), "leagues": sorted(df.league.unique().tolist()), "silhouette": {int(a): round(float(b), 3) for a, b in sil.items()},
          "pca_var_2": float(pcs.explained_variance_ratio_[:2].sum()), "roles": roles.reset_index().to_dict("records"),
          "cases": out, "mendy_rank": mendy_rank, "kante_pool": n_cm}
    (P / "key_findings.json").write_text(json.dumps(kf, indent=2, default=str))
    print(roles)

if __name__ == "__main__":
    main()
