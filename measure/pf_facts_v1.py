"""Facts per painted picture of the picture list: chord to the shown work, nearest works (start work and near-duplicates masked, and unmasked), nearest work before 1900 and from 1900, and the one-reader label (PRE-NEAR / SINCE / UNDATED-ONLY / NONE), in DINOv3 and SigLIP.
Usage: python -B pf_facts_v1.py OUTPREFIX (reads $DUROOM_OUT/inputs/pictures_v2.json and the runs' exact files) -> OUTPREFIX.txt and OUTPREFIX.json, each refused if it exists."""
import importlib.util
import json
import os
import sys

import numpy as np

B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
T3_FILE = os.path.dirname(os.path.abspath(__file__)) + '/t3_measure_v5.py'
LINE = np.array([0.762848, 0.830485])
PICS_FILE = f'{B}/inputs/pictures_v2.json'


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(out):
    for ext in ('.txt', '.json'):
        assert not os.path.exists(out + ext), f'{out + ext} exists: refusing to overwrite'
    lines = []
    say = lambda s='': (lines.append(s), print(s, flush=True))  # noqa: E731
    t3 = load_mod('t3_measure_v5', T3_FILE)
    PICS = json.load(open(PICS_FILE))
    say(f'pf_facts_v1.py; pictures {PICS_FILE} ({len(PICS)})')
    C = t3.load_corpus('real')
    OBS = t3.OBS
    assert tuple(OBS) == ('dinov3-l', 'siglip-so400m'), OBS
    ids = np.asarray(C['ids']).astype(str)
    row_of = {w: r for r, w in enumerate(ids)}
    def row(wid):
        assert str(wid) in row_of, f'work id {wid} not in the corpus ids'
        return row_of[str(wid)]
    srows = sorted({row(p['start_id']) for p in PICS})
    dups = t3.dup_rows(C, srows)
    ctx = t3.Ctx(C, None, dups)
    PRE = C['dated'] & (C['year'] < 1900)
    POST = C['dated'] & (C['year'] >= 1900)
    say(f'corpus {len(ids)} rows; pre-1900 {int(PRE.sum())}, 1900 or later {int(POST.sum())}, undated {int((~C["dated"]).sum())}; line {LINE.tolist()}')
    ex_cache = {}
    res = []

    def work(r):
        return dict(row=int(r), id=str(ids[r]), artist=str(C['artist'][r]), title=str(C['title'][r]),
                    year=(int(C['year'][r]) if C['dated'][r] else 'undated'))

    for p in PICS:
        tag = p['tag']
        ef = f'{O}/eval/{tag}_emb.npz' if tag.startswith('qgc_') else f'{O}/eval/{tag}_emb_h200_v1.npz'
        if ef not in ex_cache:
            z = np.load(ef, allow_pickle=False)
            ex_cache[ef] = (np.asarray(z['paths']).astype(str), {n: z[f'emb_{n}'] for n in OBS})
        paths, emb = ex_cache[ef]
        nm = f"iter{p['round']:04d}_{p['k']:02d}.png"
        hit = [i for i, q in enumerate(paths) if q.endswith('/' + nm)]
        assert len(hit) == 1, (p['pid'], nm, len(hit))
        i = hit[0]
        s = row(p['start_id'])
        mask = ctx.rows(s)
        w = row(p['W_id'])
        if p['W_row'] >= 0:
            assert w == p['W_row'], (p['pid'], 'shown work row', w, p['W_row'])
        o = dict(pid=p['pid'], tag=tag, exact=ef, image=paths[i], row_in_run=i, figs=p['figs'], start=work(s),
                 masked_rows=int(len(mask)), shown=work(w), shown_name_key=p['W_name'])
        for j, n in enumerate(OBS):
            q = t3.unit(emb[n][i][None, :])[0]
            cd = t3.chord(C['E'][n] @ q)                       # chord to every corpus row
            cm = cd.copy(); cm[mask] = np.inf                  # start work and its near-duplicates left out
            nb = int(np.argmin(cm))
            pre_m, post_m = np.where(PRE, cm, np.inf), np.where(POST, cm, np.inf)
            jp, jq = int(np.argmin(pre_m)), int(np.argmin(post_m))
            o[n] = dict(line=float(LINE[j]), shown_chord=float(cd[w]), shown_within=bool(cd[w] <= LINE[j]),
                        nearest=dict(**work(nb), chord=float(cm[nb])), nearest_is_shown=bool(nb == w),
                        nearest_pre=dict(**work(jp), chord=float(cm[jp]), within=bool(cm[jp] <= LINE[j])),
                        nearest_post=dict(**work(jq), chord=float(cm[jq]), within=bool(cm[jq] <= LINE[j])),
                        unmasked_min_all=float(cd.min()), unmasked_min_pre=float(np.where(PRE, cd, np.inf).min()),
                        unmasked_min_post=float(np.where(POST, cd, np.inf).min()))
        U = {n: o[n] for n in OBS}
        pre_near = any(U[n]['unmasked_min_pre'] <= U[n]['line'] for n in OBS)
        since = (not pre_near) and any(U[n]['unmasked_min_post'] <= U[n]['line'] for n in OBS)
        none = all(U[n]['unmasked_min_all'] > U[n]['line'] for n in OBS)
        o['label'] = 'PRE-NEAR' if pre_near else ('SINCE' if since else ('NONE' if none else 'UNDATED-ONLY'))
        ch = 'dinov3-l' if p['chosen'] == 'dinov3' else 'siglip-so400m'
        o['key_chord'], o['key_observer'] = p['W_chord'], ch
        o['key_chord_diff'] = abs(o[ch]['shown_chord'] - p['W_chord'])
        res.append(o)
        say(f"{p['pid']:11s} {tag:11s} {o['label']:8s} | shown {o['shown']['artist']}, {o['shown']['year']}: "
            f"D {o['dinov3-l']['shown_chord']:.3f} S {o['siglip-so400m']['shown_chord']:.3f} (key {p['W_chord']:.6f} in {ch}, diff {o['key_chord_diff']:.2e}) | "
            f"nearest D {o['dinov3-l']['nearest']['artist']} {o['dinov3-l']['nearest']['year']} {o['dinov3-l']['nearest']['chord']:.3f}"
            f"{' =shown' if o['dinov3-l']['nearest_is_shown'] else ''}; S {o['siglip-so400m']['nearest']['artist']} {o['siglip-so400m']['nearest']['year']} "
            f"{o['siglip-so400m']['nearest']['chord']:.3f}{' =shown' if o['siglip-so400m']['nearest_is_shown'] else ''} | pre within D/S "
            f"{o['dinov3-l']['nearest_pre']['within']}/{o['siglip-so400m']['nearest_pre']['within']}")
    open(out + '.json', 'x').write(json.dumps(res, indent=1))
    open(out + '.txt', 'x').write('\n'.join(lines) + '\n')
    say(f'wrote {out}.json')


if __name__ == '__main__':
    main(sys.argv[1])
