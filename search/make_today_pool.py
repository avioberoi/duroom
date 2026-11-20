"""Writes today_cap3.npz (pre2027_cap3.npz minus the positive-control, calibration and start works).
Usage: python make_today_pool.py POOLS_DIR OUT_ROOT"""
import glob
import sys

import numpy as np

STARTS = (85088, 16460, 137342, 94285, 116605)      # rows_full of the five starts (search_v2 --start row:N)
pools, O = sys.argv[1], sys.argv[2]
shards = lambda d: [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f'{O}/{d}/shard[0-9].npz'))]  # noqa: E731
ids_of = lambda d: np.concatenate([s['ids'].astype(int) for s in shards(d)])  # noqa: E731

z = np.load(f'{pools}/pre2027_cap3.npz', allow_pickle=True)
ids = z['ids'].astype(int)
pos, cal, P = ids_of('pos'), ids_of('push2'), ids_of('push')
assert len(pos) == 512 and len(cal) == 1000 and len(P) == 2048 and len(set(P)) == 2048, (len(pos), len(cal), len(P))
st = np.load(f'{pools}/starts_pre1900.npz', allow_pickle=True)
starts = st['ids'][np.isin(st['rows_full'], STARTS)].astype(int)
assert len(starts) == 5
keep = ~np.isin(ids, np.concatenate([pos, cal, starts]))
np.savez(f'{pools}/today_cap3.npz', **{k: z[k][keep] for k in z.files})
n = int(keep.sum())
print(f'pool of every date, cap 3: {len(ids)} works; removed {int((~keep).sum())} (positive control {int(np.isin(ids, pos).sum())}, calibration {int(np.isin(ids, cal).sum())}, '
      f'starts {int(np.isin(ids, starts).sum())}); today pool {n} works, {len(np.unique(z["artist"][keep]))} artists')

pick = np.sort(np.random.default_rng(11).choice(np.arange(n), 2048, replace=False))      # pushforward_sphere.py's draw with --n 2048 --rng 11
yr, pid = z['year'][keep][pick], ids[keep][pick]
print(f'the 2,048 drawn: before 1900 {int((yr < 1900).sum())}, 1900 to 1969 {int(((yr >= 1900) & (yr < 1970)).sum())}, 1970 on {int((yr >= 1970).sum())}; '
      f'also in P {int(np.isin(pid, P).sum())}; artists {len(np.unique(z["artist"][keep][pick]))}; overlap with positive control / calibration / starts '
      f'{int(np.isin(pid, pos).sum())} / {int(np.isin(pid, cal).sum())} / {int(np.isin(pid, starts).sum())}')
