"""Expected Threat (xT), following Karun Singh's value-iteration formulation (2018).

The pitch (120 x 80, StatsBomb units) is split into a grid. For each zone we estimate
  s = P(shoot), m = P(move), g = P(goal | shot), T[z -> z'] = P(move ends in z' | move from z, successful)
and solve xT(z) = s*g + m * sum_z' T[z,z'] * xT(z') by iteration. Failed moves count in the move
probability but transfer no value, so turnover risk is priced in.
"""
import numpy as np
import pandas as pd

L, W = 16, 12  # zones along / across the pitch

def zone(x, y):
    xi = np.clip((np.asarray(x) / 120 * L).astype(int), 0, L - 1)
    yi = np.clip((np.asarray(y) / 80 * W).astype(int), 0, W - 1)
    return xi * W + yi

def fit_xt(moves: pd.DataFrame, shots: pd.DataFrame, n_iter=500, tol=1e-6):
    n = L * W
    zm, zs = zone(moves.x, moves.y), zone(shots.x, shots.y)
    move_cnt = np.bincount(zm, minlength=n).astype(float)
    shot_cnt = np.bincount(zs, minlength=n).astype(float)
    goal_cnt = np.bincount(zs, weights=shots.goal.astype(float), minlength=n)
    tot = move_cnt + shot_cnt
    s = np.divide(shot_cnt, tot, out=np.zeros(n), where=tot > 0)
    m = np.divide(move_cnt, tot, out=np.zeros(n), where=tot > 0)
    g = np.divide(goal_cnt, shot_cnt, out=np.zeros(n), where=shot_cnt > 0)
    ok = moves[moves.success]
    T = np.zeros((n, n))
    np.add.at(T, (zone(ok.x, ok.y), zone(ok.end_x, ok.end_y)), 1)
    T = np.divide(T, move_cnt[:, None], out=np.zeros_like(T), where=move_cnt[:, None] > 0)
    xt, history = np.zeros(n), []
    for _ in range(n_iter):
        new = s * g + m * (T @ xt)
        history.append(new.copy())
        if np.abs(new - xt).max() < tol:
            xt = new; break
        xt = new
    return xt.reshape(L, W), {"s": s.reshape(L, W), "m": m.reshape(L, W), "g": g.reshape(L, W), "iterations": len(history)}

def value_moves(moves: pd.DataFrame, grid: np.ndarray) -> pd.Series:
    """xT added by each successful move = xT(end) - xT(start); failed moves score 0 (turnover handled in the grid)."""
    flat = grid.ravel()
    v = flat[zone(moves.end_x, moves.end_y)] - flat[zone(moves.x, moves.y)]
    return pd.Series(np.where(moves.success, v, 0.0), index=moves.index)
