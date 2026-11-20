"""One bar for all of P against a bar per k-means region of P: far (over the bar in both observers) and faithful counts for P2 and POS, K 4, 8 and 16, bars at the 99th and 95th percentile, five seeds.
Usage: python -B perregion_v2.py (prints only; reads sphere/push_emb.npz, push2_emb.npz and pos_emb.npz)"""
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

D = os.environ['DUROOM_OUT'] + '/sphere/'
OBS = [("DINOv3", "emb_dinov3-l"), ("SigLIP", "emb_siglip-so400m")]
K_NEIGH, FID = 15, 0.7785286
P, P2, POS = (np.load(D + f, allow_pickle=True) for f in ("push_emb.npz", "push2_emb.npz", "pos_emb.npz"))


def unit(a):
    a = a.astype(np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def chords(a, b):
    return np.sqrt(np.clip(2.0 - 2.0 * a @ b.T, 0.0, None))


def score_ref(e, artist):
    d = chords(e, e)
    same = artist[:, None] == artist[None, :]
    d[same] = np.inf  # drops the row itself and its artist's rows
    return np.sort(d, axis=1)[:, :K_NEIGH].mean(axis=1)


def score_other(x, e):
    return np.sort(chords(x, e), axis=1)[:, :K_NEIGH].mean(axis=1)


S = {}
for name, key in OBS:
    e, x2, xp = unit(P[key]), unit(P2[key]), unit(POS[key])
    S[name] = dict(e=e, x2=x2, xp=xp, sP=score_ref(e, P["artist"]), s2=score_other(x2, e), sp=score_other(xp, e))
    S[name]["bar"] = np.percentile(S[name]["sP"], 99)
    print("%s: global bar %.6f (median %.4f); over it: P %d of 2048, P2 %d of 1000, POS %d of 512"
          % (name, S[name]["bar"], np.median(S[name]["sP"]), (S[name]["sP"] > S[name]["bar"]).sum(),
             (S[name]["s2"] > S[name]["bar"]).sum(), (S[name]["sp"] > S[name]["bar"]).sum()))
f2, fp = P2["fidelity_clip"] <= FID, POS["fidelity_clip"] <= FID
g2 = np.all([S[n]["s2"] > S[n]["bar"] for n, _ in OBS], axis=0)
gp = np.all([S[n]["sp"] > S[n]["bar"] for n, _ in OBS], axis=0)
print("GLOBAL BAR, far in both: P2 %d of 1000 (far and faithful %d); POS %d of 512 (far and faithful %d)"
      % (g2.sum(), (g2 & f2).sum(), gp.sum(), (gp & fp).sum()))
print("faithful: P2 %d of 1000, POS %d of 512\n" % (f2.sum(), fp.sum()))

FITS = {}
for K in (4, 8, 16):
    for seed in range(5):
        for name, _ in OBS:
            km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(S[name]["e"])
            FITS[K, seed, name] = (km.labels_, km.predict(S[name]["x2"]), km.predict(S[name]["xp"]))
    print("fitted K =", K, flush=True)
for K in (4, 8, 16):
    for pct in (99, 95):
        rows = []
        for seed in range(5):
            far2, farp = [], []
            detail = []
            for name, _ in OBS:
                s = S[name]
                rP, r2, rp = FITS[K, seed, name]
                gbar = np.percentile(s["sP"], pct)
                rbar = np.array([np.percentile(s["sP"][rP == r], pct) for r in range(K)])
                far2.append(s["s2"] > rbar[r2])
                farp.append(s["sp"] > rbar[rp])
                if seed == 0:
                    for r in np.argsort(rbar):
                        detail.append("    %s region: %4d P pictures, median score %.3f, region bar %.3f | over the GLOBAL bar: P %3d "
                                      "(%.1f%%), P2 %2d of %3d, POS %2d of %3d"
                                      % (name, (rP == r).sum(), np.median(s["sP"][rP == r]), rbar[r],
                                         (s["sP"][rP == r] > gbar).sum(), 100.0 * (s["sP"][rP == r] > gbar).mean(),
                                         (s["s2"][r2 == r] > gbar).sum(), (r2 == r).sum(),
                                         (s["sp"][rp == r] > gbar).sum(), (rp == r).sum()))
            a2, ap = np.all(far2, axis=0), np.all(farp, axis=0)
            rows.append((a2.sum(), (a2 & f2).sum(), ap.sum(), (ap & fp).sum()))
            if seed == 0 and pct == 99 and K in (4, 8):
                print("K = %d regions per observer, k-means seed 0, bars at the 99th percentile:" % K)
                print("\n".join(detail))
        gl2 = np.all([S[n]["s2"] > np.percentile(S[n]["sP"], pct) for n, _ in OBS], axis=0)
        glp = np.all([S[n]["sp"] > np.percentile(S[n]["sP"], pct) for n, _ in OBS], axis=0)
        r = np.array(rows)
        print("K=%2d, %dth percentile | GLOBAL bar: P2 far %d (f&f %d), POS far %d (f&f %d) | REGION bars, 5 k-means seeds: "
              "P2 far %s (f&f %s), POS far %s (f&f %s)"
              % (K, pct, gl2.sum(), (gl2 & f2).sum(), glp.sum(), (glp & fp).sum(),
                 list(r[:, 0]), list(r[:, 1]), list(r[:, 2]), list(r[:, 3])))
    print()
sys.stdout.flush()
