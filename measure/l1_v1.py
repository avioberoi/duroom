"""Later works flagged at the line that flags k pre-1900 held-out works, per score (CLIP, DINOv3, SigLIP, min and mean of the standardised two), AUC, and the AND rule at each reader's own percentile, from agree_p1_v1.npz.
Usage: python -B l1_v1.py (prints only)"""
import os
import sys
import numpy as np

F = os.environ['DUROOM_OUT'] + '/tests12/agree/agree_p1_v1.npz'
print('input', F, flush=True)
z = np.load(F, allow_pickle=True)
print('keys', sorted(z.files), flush=True)
g = z['group_0A_1B_2later']; yr = np.asarray(z['year'], np.float64)
B, L = g == 1, g == 2
e1, e2 = L & (yr < 1960), L & (yr >= 1960)
print('sizes B, later, 1900-1959, 1960 on:', int(B.sum()), int(L.sum()), int(e1.sum()), int(e2.sum()))
S = {s: np.asarray(z[f'S1_{s}'], np.float64) for s in ('clip', 'dino', 'siglip')}
NAME = {'clip': 'CLIP', 'dino': 'DINOv3', 'siglip': 'SigLIP', 'min': 'min(zD,zS)', 'mean': 'mean(zD,zS)'}
c = lambda m: int(m.sum())  # noqa: E731
fails = []
def check(name, ok, d=''):
    print(f'CHECK {"PASS" if ok else "FAIL"}: {name} {d}', flush=True)
    if not ok: fails.append(name)

# far counts at the 99th percentile (np.percentile of B on the stored dtype; far = strictly above)
Sst = {s: z[f'S1_{s}'] for s in S}
ln = {s: float(np.percentile(Sst[s][B], 99)) for s in S}
for s in S:
    if f'line_{s}_99' in z.files:
        check(f'line {s} 99 = stored', abs(ln[s] - float(z[f'line_{s}_99'])) < 1e-12, f'{ln[s]!r} vs {float(z[f"line_{s}_99"])!r}')
far = {s: Sst[s] > ln[s] for s in S}
got = {s: (c(far[s] & B), c(far[s] & L)) for s in S}
both = far['dino'] & far['siglip']
print('reproduce 99:', got, 'both', (c(both & B), c(both & L)))

# standardised scores (B's median and MAD, as the loop does with its reference: (x - med) / (MAD + 1e-8), MAD unscaled)
zs = {}
for s in ('dino', 'siglip'):
    med = float(np.median(S[s][B])); mad = float(np.median(np.abs(S[s][B] - med)))
    zs[s] = (S[s] - med) / (mad + 1e-8)
    print(f'{NAME[s]} on B: median {med:.6f} MAD {mad:.6f}; 99th line in z units {(ln[s] - med) / (mad + 1e-8):.4f}')
SC = {'clip': S['clip'], 'dino': S['dino'], 'siglip': S['siglip'], 'min': np.minimum(zs['dino'], zs['siglip']), 'mean': 0.5 * (zs['dino'] + zs['siglip'])}
check('min of z on the AND set: {min >= t} = {zD >= t and zS >= t} at t = 3', np.array_equal(SC['min'] >= 3, (zs['dino'] >= 3) & (zs['siglip'] >= 3)))

def line_k(x, k):
    """(k+1)-th largest B value: strictly above it flags k of B when there is no tie at that value."""
    b = np.sort(x[B])[::-1]
    return float(b[k])

def at(x, t):
    f = x > t
    return c(f & B), c(f & L), c(f & e1), c(f & e2)

def auc(x, m):
    """P(later > B), ties count half (Mann-Whitney via average ranks)."""
    a, b = x[m], x[B]
    v = np.concatenate([a, b]); o = np.argsort(v, kind='mergesort'); vs = v[o]
    bd = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1], True]); st, en = bd[:-1], bd[1:]
    r = np.empty(len(v)); r[o] = np.repeat((st + en - 1) / 2.0 + 1.0, en - st)
    na, nb = len(a), len(b)
    return (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)

print('\n### L1a/b. Line that flags 71 of B; later works flagged there')
print('| score | line | B flagged | later (69,468) | 1900-1959 (35,874) | 1960 on (33,594) | later per B flagged |\n|---|---|---|---|---|---|---|')
for k in ('clip', 'dino', 'siglip', 'min', 'mean'):
    t = line_k(SC[k], 71); r = at(SC[k], t)
    print(f'| {NAME[k]} | {t:.6f} | {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[1] / r[0]:.1f} |')
r = (c(both & B), c(both & L), c(both & e1), c(both & e2))
print(f'| AND (each reader at its own 99th) | - | {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[1] / r[0]:.1f} |')

print('\n### L1c. AUC (P later > held-out earlier, ties half)')
print('| score | later | 1900-1959 | 1960 on |\n|---|---|---|---|')
for k in SC:
    print(f'| {NAME[k]} | {auc(SC[k], L):.4f} | {auc(SC[k], e1):.4f} | {auc(SC[k], e2):.4f} |')

print('\n### L1c. Later works flagged at the line that flags k of B (k = 25, 71, 247, 1,232); cells: B flagged / later / 1900-1959 / 1960 on')
print('| score | k=25 | k=71 | k=247 | k=1,232 |\n|---|---|---|---|---|')
for k in SC:
    cells = []
    for kk in (25, 71, 247, 1232):
        r = at(SC[k], line_k(SC[k], kk)); cells.append(f'{r[0]} / {r[1]:,} / {r[2]:,} / {r[3]:,}')
    print(f'| {NAME[k]} | ' + ' | '.join(cells) + ' |')

print('\n### L1d. AND rule: far in DINOv3 and SigLIP, each at its own p-th percentile line of B; then each score at the line that flags the same number of B')
print('| p | AND: B / later / 1900-1959 / 1960 on | CLIP | DINOv3 | SigLIP | min(zD,zS) | mean(zD,zS) |\n|---|---|---|---|---|---|---|')
for p in (95, 97.5, 99, 99.5):
    a = (Sst['dino'] > float(np.percentile(Sst['dino'][B], p))) & (Sst['siglip'] > float(np.percentile(Sst['siglip'][B], p)))
    nb = c(a & B)
    cells = [f'{nb} / {c(a & L):,} / {c(a & e1):,} / {c(a & e2):,}']
    for k in SC:
        r = at(SC[k], line_k(SC[k], nb)); cells.append(f'{r[0]} / {r[1]:,} / {r[2]:,} / {r[3]:,}')
    print(f'| {p} | ' + ' | '.join(cells) + ' |')
print('\nties at the k-lines: none expected; B counts above show the count reached')
print('FAILS:', fails if fails else 'none')
