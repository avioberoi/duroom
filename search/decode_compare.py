"""Does the image-variations decoder land nearer its source at guidance 3 or 10? Paired over the same works, bootstrap intervals over works, Wilcoxon signed-rank on the paired differences.
Usage: decode_compare.py OUT_DIR   (the folder holding variations_n256/score.json from decode.py ... n256 and decode_score.py DIR 0)"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

DIR = sys.argv[1]
RAW = {d: json.load(open(f'{DIR}/{d}_n256/score.json')) for d in ('variations',)}
S = {d: RAW[d]['lenses'] for d in RAW}
N_RECORD = RAW['variations']['n_record']
B = 10000
rng = np.random.default_rng(0)


def interval(f, *arrays):
    """Median-type statistic f on works, with a 95% bootstrap interval from resampling works (the same resample for paired arrays)."""
    n = len(arrays[0])
    boot = [f(*(a[i] for a in arrays)) for i in (rng.integers(0, n, n) for _ in range(B))]
    return {'value': float(f(*arrays)), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist()}


result = {'boot': B, 'lenses': {}}
for name in S['variations']:
    w = {d: {p['row']: p for p in S[d][name]['per_work']} for d in S}
    rows = sorted(w['variations'])
    get = lambda d, k: np.array([w[d][r][k] for r in rows], dtype=float)    # noqa: E731
    d10 = get('variations', 'd10')
    two = {d: (get(d, 'd_source') + get(d, 'd_source_seed1')) / 2 for d in S}
    entry = {'n_works': len(rows)}
    burn = lambda k: [rng.integers(0, len(rows), len(rows)) for _ in range(k * B)]    # noqa: E731  the run also bootstrapped a second decoder: 5 intervals before these, 3 after
    burn(5)
    for d in S:
        both = np.r_[get(d, 'd_source'), get(d, 'd_source_seed1')]
        entry[d] = {'d_source_median_two_seeds': interval(np.median, two[d]),
                    'share_nearer_than_own_d10': interval(lambda a, b, c: np.mean(np.r_[a < c, b < c]), get(d, 'd_source'), get(d, 'd_source_seed1'), d10),    # over both seeds' renders
                    'rank_median_seed0': interval(np.median, get(d, 'rank')), 'rank_median_share_of_record_seed0': float(np.median(get(d, 'rank')) / N_RECORD), 'd_seed_median': interval(np.median, get(d, 'd_seed')),
                    'share_rank_within_1pct_both_seeds': float(np.mean(np.r_[get(d, 'rank'), get(d, 'rank_seed1')] < N_RECORD / 100)), 'renders': int(len(both))}
        st = S[d][name]['settings']
        if 'alt' in st:    # the other guidance, seed 0, the same works
            base = {r: v for r, v in zip(st['base']['rows'], st['base']['d_source'])}
            alt = {r: v for r, v in zip(st['alt']['rows'], st['alt']['d_source'])}
            common = sorted(set(base) & set(alt))
            diff = np.array([alt[r] - base[r] for r in common])
            entry[d]['alt_minus_base_d_source'] = {**interval(np.median, diff), 'n': len(common), 'wilcoxon_p': float(wilcoxon(diff).pvalue)}
    burn(3)
    result['lenses'][name] = entry
    print(name, {d: (round(entry[d]['d_source_median_two_seeds']['value'], 3), [round(x, 3) for x in entry[d]['share_nearer_than_own_d10']['ci95']]) for d in S}, flush=True)
json.dump(result, open(f'{DIR}/compare_n256.json', 'w'), indent=1)
print('done', flush=True)
