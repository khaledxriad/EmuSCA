"""Leakage-emulation library for Kyber-style arithmetic (q = 3329) on a 32-bit Arm core.
All leakage is emulated: L = HW(x) + beta*HD(x, x_prev) + Gaussian noise."""
import numpy as np

Q = 3329
QINV = 62209            # q^{-1} mod 2^16 (Kyber reference value)

def hw32(x):
    """Hamming weight of the 32-bit two's complement representation."""
    x = np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) & 0xFFFFFFFF) >> 24

def hw16(x):
    return hw32(np.asarray(x, dtype=np.int64) & 0xFFFF)

def center(x):
    """Map residues to the signed range [-(q-1)/2, (q-1)/2]."""
    x = np.asarray(x, dtype=np.int64) % Q
    return np.where(x > (Q - 1) // 2, x - Q, x)

def mont_reduce(a):
    """Signed Montgomery reduction exactly as in the Kyber reference code."""
    a = np.asarray(a, dtype=np.int64)
    t = ((a & 0xFFFF) * QINV) & 0xFFFF
    t = np.where(t >= 0x8000, t - 0x10000, t)           # (int16_t)
    return (a - t * Q) >> 16, t

def fqmul_trace(s, u):
    """Return the four intermediates of fqmul(s, u): s, p, t, res."""
    s = np.asarray(s, dtype=np.int64); u = np.asarray(u, dtype=np.int64)
    p = s * u
    res, t = mont_reduce(p)
    return s, p, t, res

def hw_points(s, u, beta=0.25):
    """Four leakage points (HW + beta*HD to previous intermediate)."""
    x0, x1, x2, x3 = fqmul_trace(s, u)
    h = [hw16(x0), hw32(x1), hw16(x2), hw16(x3)]
    hd = [0 * hw16(x0), hw32(x1 ^ x0), hw16(x2 ^ x1), hw16(x3 ^ x2)]
    return np.stack([h[j] + beta * hd[j] for j in range(4)], axis=-1).astype(np.float32)

_tables = {}
def hw_tables():
    """HW tables T[j][s_idx, u_idx] for j=0..3 (hypothesis s over all residues)."""
    if 'T' not in _tables:
        S = center(np.arange(Q))[:, None]; U = center(np.arange(Q))[None, :]
        x0, x1, x2, x3 = fqmul_trace(S + 0 * U, U + 0 * S)
        _tables['T'] = [hw16(x0).astype(np.int8), hw32(x1).astype(np.int8), hw16(x2).astype(np.int8), hw16(x3).astype(np.int8)]
    return _tables['T']

def second_order_table(point=3):
    """M[u_idx, s_idx] = E_{s0}[ cHW(res(s0,u)) * cHW(res(s-s0,u)) ] via cyclic convolution mod q."""
    key = ('M', point)
    if key not in _tables:
        T = hw_tables()[point].astype(np.float64)          # [s_idx, u_idx]
        mu = T.mean()
        g = (T - mu).T                                     # [u_idx, x]
        G = np.fft.rfft(g, axis=1)
        M = np.fft.irfft(G * G, n=Q, axis=1) / Q
        _tables[key] = (M.astype(np.float32), mu)
    return _tables[key]
