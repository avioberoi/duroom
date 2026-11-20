"""Checks a render after render and embed: the 8 known-answer works against their stored repaints (PNG byte identity counted, chord per encoder, gate), logs, per-mode manifests.
Usage: python -B t2_post.py OUT MODE   (default env; stdout is check.txt, also writes check.json; exit 1 on a failed check)"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
OUT, MODE = sys.argv[1], sys.argv[2]
ENC = ('clip-vit-l-14', 'dinov3-l', 'siglip-so400m')
unit = lambda X: np.asarray(X, np.float64) / np.linalg.norm(np.asarray(X, np.float64), axis=-1, keepdims=True)  # noqa: E731
chord = lambda c: np.sqrt(np.clip(2 - 2 * c, 0, None))  # noqa: E731
pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])  # noqa: E731
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()  # noqa: E731
rep, fails = {'mode': MODE, 'out': OUT}, []


def say(k, v):
    rep[k] = v
    print(f'{k}: {v}', flush=True)


em = np.load(f'{OUT}/emb.npz', allow_pickle=True)
paths = [str(p) for p in em['paths']]
kat = [ln.strip() for ln in open(f'{OUT}/inputs/expected_kat.txt') if ln.strip()]
main = [ln.strip() for ln in open(f'{OUT}/inputs/expected_main.txt') if ln.strip()]
if paths != kat + main:
    fails.append('emb.npz paths are not the expected list')
for e in ENC:
    X = em[f'emb_{e}']
    if X.shape[0] != len(paths) or not np.isfinite(X).all():
        fails.append(f'emb_{e} shape or values')
if 'coherence' not in em.files or len(em['coherence']) != len(paths) or not np.isfinite(em['coherence']).all():
    fails.append('coherence missing or not finite')
say('rows embedded', f'{len(paths)} ({len(kat)} known-answer + {len(main)} main); keys {sorted(em.files)}')

P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
prow = {int(i): k for k, i in enumerate(pid(P['paths']))}
kid = pid(kat)
kr = [prow[int(i)] for i in kid]
same = [md5(p) == md5(f'{O}/push/img/push_{i}.png') for p, i in zip(kat, kid)]
say('known-answer works', kid.tolist())
say('known-answer PNG bytes identical to the stored repaint', f'{sum(same)} of {len(kid)}')
for k, e in enumerate(ENC):
    c = chord((unit(em[f'emb_{e}'][:len(kat)]) * unit(P[f'emb_{e}'][kr])).sum(1))
    say(f'known-answer chord to stored reading, {e} (max, median)', [round(float(c.max()), 6), round(float(np.median(c)), 6)])
    if e == 'clip-vit-l-14' and c.max() > 0.1:
        fails.append('known-answer CLIP chord above 0.1 (CLIP loader differs from the exact one)')
say('known-answer gate |new - stored| max', round(float(np.abs(em['coherence'][:len(kat)] - P['coherence'][kr]).max()), 6))

eo = open(f'{OUT}/logs/embed.out', errors='replace').read()
if 'CLIP loaded' not in eo:
    fails.append('embed log lacks "CLIP loaded" (the exact CLIP loader did not run)')
say('embed log has "CLIP loaded"', 'CLIP loaded' in eo)
for f in ('kat_render.out', 'render.out'):
    t = open(f'{OUT}/logs/{f}', errors='replace').read()
    ok = 'variations backend loaded: norm 17.53, guidance 3.0' in t
    say(f'{f} has "variations backend loaded: norm 17.53, guidance 3.0"', ok)
    if not ok:
        fails.append(f'{f}: decoder settings line missing')
    g = re.findall(r'^NVIDIA ([\w ]+?),', t, re.M)
    say(f'{f} card', sorted(set(g)))
tcl = {os.path.basename(f): open(f, errors='replace').read().count('init.tcl') for f in sorted(glob.glob(f'{OUT}/logs/*.err'))}
say('init.tcl lines per .err', tcl)

if MODE == 'prior':
    new, old = np.load(f'{OUT}/prior/manifest.npz', allow_pickle=True), np.load(f'{O}/baselines/prior/manifest.npz', allow_pickle=True)
    same_cols = {k: bool((np.asarray(new[k]) == np.asarray(old[k])).all()) for k in ('kind', 'tag', 'row', 'seed', 'step')}
    dv = float(np.abs(new['v'].astype(np.float64) - old['v'].astype(np.float64)).max())
    say('manifest columns equal to the stored prior manifest', same_cols)
    say('manifest v: bit-identical, max |diff|', [bool((new['v'] == old['v']).all()), dv])
    if not all(same_cols.values()) or dv > 1e-6:
        fails.append('the 256 requests are not the stored ones')
    if [str(p) for p in new['paths']] != main:
        fails.append('prior manifest paths differ from the expected list')
    ident = sum(md5(p) == md5(f'{O}/baselines/prior/{os.path.basename(p)}') for p in main)
    say('PNG bytes identical to the stored baseline render (same name)', f'{ident} of {len(main)}')
elif MODE == 'noise':
    new, req = np.load(f'{OUT}/noise/manifest.npz', allow_pickle=True), np.load(f'{OUT}/inputs/noise_req.npz', allow_pickle=True)
    ok = (len(new['v']) == 160 and (new['seed'] == req['render_seed']).all() and (new['row'] == np.arange(160)).all()
          and [str(t) for t in new['tag']] == [str(t) for t in req['family']])
    say('manifest rows/seeds/kinds as requested', bool(ok))
    say('manifest v vs the stored CLIP readings, max |diff|', float(np.abs(new['v'].astype(np.float64) - req['v'].astype(np.float64)).max()))
    if not ok or [str(p) for p in new['paths']] != main:
        fails.append('noise manifest differs from the requests or the expected list')
say('FAILED checks', fails if fails else 'none')
json.dump(rep, open(f'{OUT}/check.json', 'x'), indent=1, default=str)
sys.exit(1 if fails else 0)
