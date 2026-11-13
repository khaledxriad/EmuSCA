import numpy as np
import lbsca as L

Q = L.Q

def corr_checkpoints(cols, model_getters, grid, true_idx, signed_max=True):
    """Incremental Pearson correlation of trace columns against hypothesis models.
    cols: list of arrays (Nmax,) float32;  model_getters: list of f(lo,hi)->(n,Q) float32.
    Returns, for each checkpoint N in grid, (best_hyp, rank_of_true)."""
    P = len(cols)
    sh = np.zeros((P, Q)); sh2 = np.zeros((P, Q)); shl = np.zeros((P, Q))
    sl = np.zeros(P); sl2 = np.zeros(P)
    out = []
    lo = 0
    for N in grid:
        hi = N
        if hi > lo:
            for p in range(P):
                for a in range(lo, hi, 3000):
                    b = min(hi, a + 3000)
                    H = model_getters[p](a, b)
                    l = cols[p][a:b].astype(np.float64)
                    sh[p] += H.sum(0, dtype=np.float64); sh2[p] += np.einsum('ij,ij->j', H, H, dtype=np.float64)
                    shl[p] += l @ H
                    sl[p] += l.sum(); sl2[p] += (l * l).sum()
            lo = hi
        num = N * shl - sh * sl[:, None]
        den = np.sqrt(np.maximum(N * sh2 - sh ** 2, 1e-12) * np.maximum(N * sl2 - sl ** 2, 1e-12)[:, None])
        rho = num / den
        score = rho.sum(0)
        best = int(np.argmax(score))
        rank = int((score > score[true_idx]).sum())
        out.append((best, rank))
    return out

def table_getter(tab, uidx):
    """tab: int8 [Q(s), Q(u)] ; returns f(lo,hi)->(n,Q) float32."""
    def f(lo, hi):
        return tab[:, uidx[lo:hi]].T.astype(np.float32)
    return f

def rowtable_getter(M, uidx):
    """M: float32 [Q(u), Q(s)]."""
    def f(lo, hi):
        return M[uidx[lo:hi]]
    return f

def residx(s_centered):
    return (np.asarray(s_centered) % Q).astype(np.int64)

def success_curve(results_per_trial, true_idx_list):
    """results_per_trial: list (trials) of list (grid) of (best, rank) ; returns SR per grid."""
    T = len(results_per_trial); G = len(results_per_trial[0])
    sr = np.zeros(G); 
    for r, t in zip(results_per_trial, true_idx_list):
        for g in range(G):
            sr[g] += (r[g][0] == t) or (r[g][0] == (Q - t) % Q)   # success up to the sign ambiguity s <-> -s
    return sr / T

def mtd_from_sr(grid, sr, thr=0.9):
    """Smallest grid N with SR>=thr (log-linear interpolation); inf if never."""
    grid = np.asarray(grid, float)
    for i in range(len(grid)):
        if sr[i] >= thr:
            if i == 0: return grid[0]
            x0, x1 = np.log(grid[i-1]), np.log(grid[i]); y0, y1 = sr[i-1], sr[i]
            if y1 == y0: return grid[i]
            return float(np.exp(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)))
    return float('inf')
