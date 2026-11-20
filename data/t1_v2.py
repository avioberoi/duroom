"""What does "unexplored" mean, region by region: candidate scores (H1), region table (H2), repaints (H3) and reference sizes and caps (H4), per observer.
Usage: t1_v2.py --out OUT [--tag T] [--observers dinov3-l,siglip-so400m] [--parts H1,H2,H3,H4] [--split N] [--kseed N] (see --help); writes OUT/run_<SLURM_JOB_ID>_<tag>/"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True, help='root folder; the run folder run_<job>_<tag> is created inside')
ap.add_argument('--tag', default='headline')
ap.add_argument('--retro-dir', default=os.path.dirname(os.path.abspath(__file__)), help='folder of data.py')
ap.add_argument('--obs-dir', default=os.environ['DUROOM_EMB'])
ap.add_argument('--sphere-dir', default=os.environ['DUROOM_OUT'] + '/sphere')
ap.add_argument('--observers', default='dinov3-l,siglip-so400m')
ap.add_argument('--parts', default='H1,H2,H3,H4')
ap.add_argument('--split', type=int, default=1, help='rng seed of the artist halves')
ap.add_argument('--kseed', type=int, default=1, help='rng seed of the k-means start')
ap.add_argument('--ks', default='5,10,15,30', help='neighbour counts for S1 (at most 30)')
ap.add_argument('--Ks', default='16,32,64', help='numbers of regions')
ap.add_argument('--qs', default='90,95,99', help='radius percentiles')
ap.add_argument('--levels', default='90,95,99', help='line percentiles (false-alarm rates 10/5/1 percent on the past)')
ap.add_argument('--ndir', type=int, default=32, help='leftover: directions kept')
ap.add_argument('--fit-cap', type=int, default=20, help='leftover fit sample: at most this many A works per artist (as unexp.py)')
ap.add_argument('--cap-seed', type=int, default=0, help='rng seed of the leftover fit sample')
ap.add_argument('--local-k', type=int, default=15)
ap.add_argument('--dup-cos', type=float, default=0.99)
ap.add_argument('--table-K', type=int, default=32)
ap.add_argument('--table-q', type=int, default=95)
ap.add_argument('--no-master', action='store_true', help='drop works whose artist (after an en/ prefix) starts with master-')
ap.add_argument('--media', default='', help="comma list; default retro.py's MEDIA")
ap.add_argument('--h4-sizes', default='2048,512')
ap.add_argument('--h4-caps', default='1,2,3,5,10,0', help='0 = no cap')
ap.add_argument('--h4-draws', type=int, default=20)
ap.add_argument('--h4-seed', type=int, default=0)
ap.add_argument('--h4-levels', default='95,99')
ap.add_argument('--h4-save-k', type=int, default=15, help='k of the per-work H4 scores saved per draw')
ap.add_argument('--smoke-frac', type=float, default=0.0, help='code check only: keep this random share of usable works (rng 0) plus P')
args = ap.parse_args()
ints = lambda s: tuple(int(x) for x in s.split(',') if x != '')    # noqa: E731
KS, KR, QS, LEVELS = ints(args.ks), ints(args.Ks), ints(args.qs), ints(args.levels)
PARTS = set(args.parts.split(','))
assert PARTS <= {'H1', 'H2', 'H3', 'H4'} and args.table_K in KR and args.table_q in QS and max(KS) <= 30

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1core_v2 as core  # noqa: E402

sys.path.insert(0, args.retro_dir)
from data import ids, labels, lens, year  # noqa: E402

JOB = os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d%H%M%S'))
RUN = f'{args.out}/run_{JOB}_{args.tag}'
os.makedirs(RUN, exist_ok=False)


def log(*a):
    print(*a, flush=True)


def wcsv(df, name):
    with open(f'{RUN}/{name}', 'x') as f:
        df.to_csv(f, index=False)


def wnpz(name, **arrs):
    with open(f'{RUN}/{name}', 'xb') as f:
        np.savez(f, **arrs)


def pct(v, L):
    return float(np.percentile(v, L))


log(f't1_v2.py job {JOB} run {RUN}')
log('args', json.dumps(vars(args)))
log('extracted code:\n' + core.describe())
with open(f'{RUN}/args.json', 'x') as f:
    json.dump({**vars(args), 'job': JOB, 'sources': core.SOURCES, 'extracted': core.EXTRACTED}, f, indent=1)

artist, style, genre = (np.asarray(labels[k]).astype(str) for k in ('artist', 'style', 'genre'))
core.set_labels(artist)
codes = np.unique(artist, return_inverse=True)[1].astype(np.int64)
MEDIA = args.media.split(',') if args.media else core.MEDIA_DEFAULT
media = np.isin(genre, MEDIA)
base = np.array([a[3:] if a.startswith('en/') else a for a in artist])
master = np.char.startswith(base.astype(str), 'master-')
usable = (artist != '') & (year >= 1000) & (year <= 2026)
if args.no_master:
    usable = usable & ~master
row_of = {int(i): r for r, i in enumerate(ids)}
PID = re.compile(r'push_(\d+)\.png$')
if args.smoke_frac:
    inP = np.zeros(len(ids), bool)
    inP[[row_of[int(PID.search(str(p)).group(1))] for p in np.load(f'{args.sphere_dir}/push_emb.npz')['paths']]] = True
    usable = usable & ((np.random.default_rng(0).random(len(ids)) < args.smoke_frac) | inP)
    log(f'SMOKE: usable works subsampled to {int(usable.sum())}')
A, Bq, Lq, Ldrop, R, L = core.split(artist, usable, year, args.split)
n = len(ids)
decade = np.floor(np.nan_to_num(year) / 10) * 10
log(f'corpus rows {n}; usable {int(usable.sum())} (no-master {args.no_master}: master- works {int((master & (artist != "")).sum())}); R {len(R)} works, '
    f'{len(np.unique(artist[R]))} artists; later {len(L)}')
log(f'split {args.split}: A {len(A)} works / {len(np.unique(artist[A]))} artists; B {len(Bq)} / {len(np.unique(artist[Bq]))}; later scored {len(Lq)} '
    f'(artist in B {int(np.isin(artist[Lq], artist[Bq]).sum())}, not in R {int((~np.isin(artist[Lq], artist[R])).sum())}); later DROPPED (artist in A) {len(Ldrop)}')
log(f'media = first genre label in {MEDIA}: A {int(media[A].sum())}, B {int(media[Bq].sum())}, later {int(media[Lq].sum())}')
groups = {'A': A, 'A_media': A[media[A]], 'A_rest': A[~media[A]], 'B': Bq, 'B_media': Bq[media[Bq]], 'B_rest': Bq[~media[Bq]],
          'later': Lq, 'later_1900_1959': Lq[year[Lq] < 1960], 'later_1960_on': Lq[year[Lq] >= 1960],
          'later_media': Lq[media[Lq]], 'later_rest': Lq[~media[Lq]]}
for d in sorted(set(decade[Lq].tolist())):
    groups[f'later_{int(d)}s'] = Lq[decade[Lq] == d]
later_groups = [g for g in groups if g.startswith('later')]
log('groups: ' + ', '.join(f'{g} {len(r)}' for g, r in groups.items()))


def print_h1(shares, aucs, title):
    log(f'\n{title}\n  AUC later vs B (later / 1900-1959 / 1960 on) | share of later over the A-line at 90/95/99 | over the B-line | A over A-line 99, B over B-line 99 (loners: line on the leftover)')
    sh = shares.set_index(['candidate', 'rule', 'line', 'level', 'group'])['share']
    au = aucs.set_index(['candidate', 'rule', 'group'])['auc_vs_B']
    for cand, rule, key in core.h1_candidates(KS, KR, QS):
        a = ' '.join(f'{au.get((cand, rule, g), np.nan):.3f}' for g in ('later', 'later_1900_1959', 'later_1960_on'))
        if cand == 'loner':
            rules = ('excl', 'all')
        else:
            rules = (rule,)
        for ru in rules:
            if cand.startswith('region_'):
                q = int(cand.rsplit('q', 1)[1])
                s = ' '.join(f'{ln}:{sh.get((cand, "-", ln, q, "later"), np.nan):.3f}' for ln in ('def_r>1', 'A', 'B'))
                fa = f'A def {sh.get((cand, "-", "def_r>1", q, "A"), np.nan):.4f} B def {sh.get((cand, "-", "def_r>1", q, "B"), np.nan):.4f}'
                log(f'  {cand:18s} {"-":4s} {a} | later at q{q} {s} | {fa}')
                continue
            sA = ' '.join(f'{sh.get((cand, ru, "A", L, "later"), np.nan):.3f}' for L in LEVELS)
            sB = ' '.join(f'{sh.get((cand, ru, "B", L, "later"), np.nan):.3f}' for L in LEVELS)
            fa = f'{sh.get((cand, ru, "A", 99, "A"), np.nan):.4f} {sh.get((cand, ru, "B", 99, "B"), np.nan):.4f}'
            log(f'  {cand:18s} {ru:4s} {a} | {sA} | {sB} | {fa}')


def top5(v, rows):
    x = v[rows]
    x = x[x != '']
    if len(x) == 0:
        return ''
    vals, cnt = np.unique(x, return_counts=True)
    o = np.argsort(-cnt, kind='stable')[:5]
    return '; '.join(f'{vals[i]}:{cnt[i]}' for i in o)


def region_table(reg, Q, gQ, fit_name, outside_past):
    lab = reg['lab']
    nB, nL = int((gQ == 1).sum()), int((gQ == 2).sum())
    out = []
    for k in range(len(reg['C'])):
        inr = lab == k
        a, b, l = Q[inr & (gQ == 0)], Q[inr & (gQ == 1)], Q[inr & (gQ == 2)]
        ls, bs = len(l) / nL, len(b) / nB
        row = {'fit': fit_name, 'region': k, 'n_fit': int(reg['n_fit'][k]), 'n_A': len(a), 'n_B': len(b), 'n_later': len(l),
               'n_later_1900_1959': int((year[l] < 1960).sum()), 'n_later_1960_on': int((year[l] >= 1960).sum()),
               'artists_A': len(np.unique(artist[a])), 'artists_B': len(np.unique(artist[b])), 'later_share': ls, 'B_share': bs,
               'later_share_over_B_share': ls / bs if bs else np.nan, 'pre1900_share_of_A_plus_later': len(a) / (len(a) + len(l)) if len(a) + len(l) else np.nan,
               f'later_outside_every_past_region_q{args.table_q}': float(outside_past[l].mean()) if len(l) else np.nan}
        for j, q in enumerate(reg['qs']):
            row[f'radius_deg_q{q}'] = reg['rad'][k, j]
            row[f'raw_radius_chord_q{q}'] = reg['rawrad'][k, j]
        row.update({'top5_styles_A': top5(style, a), 'top5_genres_A': top5(genre, a), 'top5_styles_later': top5(style, l), 'top5_genres_later': top5(genre, l)})
        out.append(row)
    return pd.DataFrame(out)


def load_repaints(obs):
    """P, P2, pos, today readings in the observer; ids deduplicated in that order (first kept)."""
    sets = (('P', 'push_emb.npz'), ('P2', 'push2_emb.npz'), ('pos', 'pos_emb.npz'), ('today', 'push_today_emb.npz'))
    wid, lab, rep, seen, dropped = [], [], [], set(), {}
    for name, f in sets:
        z = np.load(f'{args.sphere_dir}/{f}')
        w = np.array([int(PID.search(str(p)).group(1)) for p in z['paths']])
        e = z[f'emb_{obs}'].astype('float64')
        keep = np.array([int(i) not in seen for i in w])
        dropped[name] = int((~keep).sum())
        seen.update(int(i) for i in w)
        wid.append(w[keep]); lab += [name] * int(keep.sum()); rep.append(e[keep])
    wid, lab, rep = np.concatenate(wid), np.array(lab), np.vstack(rep)
    return wid, lab, core.unit_rows(rep), dropped


for obs in args.observers.split(','):
    t0 = time.time()
    log(f'\n==================== {obs} ====================')
    X = lens(f'{args.obs_dir}/{obs}.npz').astype('float64')
    assert np.isfinite(X).all() and (np.linalg.norm(X, axis=1) > 1e-8).all()
    X /= np.linalg.norm(X, axis=1, keepdims=True)    # unit rows first
    log(f'lens {obs}: {X.shape}, loaded in {time.time() - t0:.0f}s')
    if 'H1' in PARTS or 'H2' in PARTS or 'H3' in PARTS:
        S, LON, REG, meta = core.h1_scores(X, A, Bq, Lq, codes, ks=KS, Ks=KR, qs=QS, levels=LEVELS, ndir=args.ndir, fit_cap=args.fit_cap,
                                           cap_seed=args.cap_seed, kseed=args.kseed, local_k=args.local_k, log=log)
        Q = meta['Q']
        gQ = np.concatenate([np.zeros(len(A)), np.ones(len(Bq)), np.full(len(Lq), 2)]).astype(np.int8)
        log(f'  leftover fit sample {len(meta["fit_rows"])} A works (at most {args.fit_cap} per artist, rng {args.cap_seed})')
    if 'H1' in PARTS:
        shares, aucs = core.h1_tables(S, LON, groups, later_groups, KS, KR, QS, LEVELS)
        wcsv(shares, f'{obs}_H1_shares.csv')
        wcsv(aucs, f'{obs}_H1_auc.csv')
        arrays = {'id': ids[Q], 'group_0A_1B_2later': gQ, 'year': year[Q], 'artist': artist[Q], 'media': media[Q], 'genre': genre[Q], 'style': style[Q]}
        arrays.update({k: v[Q] for k, v in S.items()})
        for (line, Lv, rule), (t, cnt) in LON.items():
            arrays[f'loner_count_{line}{Lv}_{rule}'] = cnt[Q]
            arrays[f'loner_cut_{line}{Lv}'] = np.float64(t)
        for Kr, reg in REG.items():
            arrays[f'region_K{Kr}_nearest'] = reg['lab']
            arrays[f'region_K{Kr}_angle_nearest_deg'] = reg['ang_near']
        wnpz(f'{obs}_H1_perwork.npz', **arrays)
        cr = {'qs': np.array(QS), 'Q_ids': ids[Q], 'Q_group_0A_1B_2later': gQ, 'A_ids': ids[A], 'B_ids': ids[Bq], 'later_ids': ids[Lq],
              'A_artist': artist[A], 'B_artist': artist[Bq], 'later_artist': artist[Lq], 'later_dropped_ids': ids[Ldrop],
              'split_seed': np.int64(args.split), 'kmeans_seed': np.int64(args.kseed),
              'note': np.array('centres are unit rows in the space of unit stored vectors minus K{K}_mean, re-normalised; radius_deg[k, j] = '
                               'qs[j]-th percentile of angles (degrees) to centre k of the A works nearest to it; NaN = no A works; '
                               'r_q(x) = min over k with radius > 0 of angle(x, centre k) / radius_deg[k, j]')}
        for Kr, reg in REG.items():
            cr.update({f'K{Kr}_mean': reg['mean'], f'K{Kr}_centres': reg['C'], f'K{Kr}_radius_deg': reg['rad'], f'K{Kr}_raw_centres': reg['rawc'],
                       f'K{Kr}_raw_radius_chord': reg['rawrad'], f'K{Kr}_n_A': reg['n_fit'], f'K{Kr}_A_region': reg['lab_fit'], f'K{Kr}_Q_region': reg['lab']})
        wnpz(f'{obs}_centres_radii.npz', **cr)
        print_h1(shares, aucs, f'{obs} H1 (split {args.split}, k-means seed {args.kseed}); rules: excl = own artist excluded, all = all rows (B and later identical, only A\'s own line moves)')
        thr = args.dup_cos
        pa = core.dup_pairs(X, np.concatenate([Bq, Lq]), A, thr)
        pb = core.dup_pairs(X, Lq, Bq, thr)
        gname = np.full(n, '', dtype=object)
        gname[A], gname[Bq], gname[Lq] = 'A', 'B', 'later'
        dp = pd.DataFrame({'id_scored': np.concatenate([ids[pa[0]], ids[pb[0]]]), 'group_scored': np.concatenate([gname[pa[0]], gname[pb[0]]]),
                           'id_other': np.concatenate([ids[pa[1]], ids[pb[1]]]), 'group_other': np.concatenate([gname[pa[1]], gname[pb[1]]]),
                           'cosine': np.concatenate([pa[2], pb[2]])})
        wcsv(dp, f'{obs}_H1_dup_pairs.csv')
        dup_rows = np.unique(np.concatenate([pa[0], pb[0], pb[1]]))
        keep = np.ones(n, bool)
        keep[dup_rows] = False
        log(f'\n  near-duplicate pairs (cosine > {thr}): B-A {int((gname[pa[0]] == "B").sum())}, later-A {int((gname[pa[0]] == "later").sum())}, later-B {len(pb[0])}; '
            f'scored works in a pair (dropped below, lines unchanged): B {int(np.isin(dup_rows, Bq).sum())}, later {int(np.isin(dup_rows, Lq).sum())}')
        shares_d, aucs_d = core.h1_tables(S, LON, groups, later_groups, KS, KR, QS, LEVELS, keep=keep)
        wcsv(shares_d, f'{obs}_H1_shares_nodup.csv')
        wcsv(aucs_d, f'{obs}_H1_auc_nodup.csv')
        print_h1(shares_d, aucs_d, f'{obs} H1 without the scored works in near-duplicate pairs (lines as above)')
        log(f'  H1 written, {time.time() - t0:.0f}s')
    if 'H2' in PARTS:
        TK, TQ = args.table_K, args.table_q
        outside_past = S[f'region_K{TK}_q{TQ}'] > 1
        F = np.sort(np.concatenate([A, Lq]))
        regF = core.regions(X, F, Q, TK, args.kseed, QS)
        tab = pd.concat([region_table(REG[TK], Q, gQ, 'past_A', outside_past), region_table(regF, Q, gQ, 'A_plus_later', outside_past)])
        wcsv(tab, f'{obs}_H2_regions_K{TK}.csv')
        wnpz(f'{obs}_H2_allfit_K{TK}.npz', mean=regF['mean'], centres=regF['C'], radius_deg=regF['rad'], raw_centres=regF['rawc'],
             raw_radius_chord=regF['rawrad'], n_fit=regF['n_fit'], Q_ids=ids[Q], Q_region=regF['lab'], qs=np.array(QS), fit_ids=ids[F])
        log(f'\n{obs} H2 region table K={TK} (split {args.split}, seed {args.kseed}); outside = beyond every past region at q{TQ}; all-works fit: empty {regF["empty"]}')
        cols = ['fit', 'region', 'n_A', 'n_B', 'n_later', 'n_later_1900_1959', 'n_later_1960_on', 'artists_A', 'artists_B', 'later_share_over_B_share',
                'pre1900_share_of_A_plus_later', f'later_outside_every_past_region_q{TQ}', f'radius_deg_q{TQ}', f'raw_radius_chord_q{TQ}']
        with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200):
            log(tab[cols].round(3).to_string(index=False))
        log(f'  H2 written, {time.time() - t0:.0f}s')
    if 'H3' in PARTS:
        zP = np.load(f'{args.sphere_dir}/push_emb.npz')
        P = np.sort(np.array([row_of[int(PID.search(str(p)).group(1))] for p in zP['paths']]))
        assert len(P) == 2048 and np.isin(P, R).all()
        Q1 = np.unique(np.concatenate([P, Bq, Lq]))
        c1 = core.c1_scores(X, P, Q1, codes, KS)
        log(f'\n{obs} H3 C1: S1 against P\'s {len(P)} originals (stored vectors; P works in A {int(np.isin(P, A).sum())}, in B {int(np.isin(P, Bq).sum())}) '
            f'vs against A; "whole past" = A for the held-out comparison. Own lines: P-line = P\'s own leave-artist-out percentile, A-line = A\'s.')
        from scipy.stats import spearmanr
        rows = []
        for k in KS:
            lineP = {Lv: pct(c1[f'k{k}_excl'][P], Lv) for Lv in LEVELS}
            lineA = {Lv: pct(S[f'S1_k{k}_excl'][A], Lv) for Lv in LEVELS}
            sA = S[f'S1_k{k}_excl']
            for rule in ('excl', 'all'):
                sP = c1[f'k{k}_{rule}']
                BP = {Lv: pct(sP[Bq], Lv) for Lv in LEVELS}
                BA = {Lv: pct(sA[Bq], Lv) for Lv in LEVELS}
                for g in ('B', 'later', 'later_1900_1959', 'later_1960_on'):
                    r = groups[g]
                    rho = float(spearmanr(sP[r], sA[r])[0])
                    for lt, (lp, la) in (('own', (lineP, lineA)), ('B', (BP, BA))):
                        for Lv in LEVELS:
                            fP, fA = sP[r] > lp[Lv], sA[r] > la[Lv]
                            rows.append({'k': k, 'rule': rule, 'group': g, 'n': len(r), 'spearman': rho, 'line': lt, 'level': Lv, 'cut_P': lp[Lv], 'cut_A': la[Lv],
                                         'P_only': int((fP & ~fA).sum()), 'A_only': int((fA & ~fP).sum()), 'both': int((fP & fA).sum()), 'neither': int((~fP & ~fA).sum())})
        c1t = pd.DataFrame(rows)
        wcsv(c1t, f'{obs}_H3_C1.csv')
        g1 = np.zeros(len(Q1), np.int8) - 1
        g1[np.isin(Q1, Bq)], g1[np.isin(Q1, Lq)] = 1, 2
        wnpz(f'{obs}_H3_C1_perwork.npz', id=ids[Q1], group_1B_2later_m1Ponly=g1, inP=np.isin(Q1, P), **{f'S1vsP_{k}': v[Q1] for k, v in c1.items()})
        with pd.option_context('display.width', 250, 'display.max_rows', 400):
            log(c1t[c1t.k == 15].to_string(index=False))
        wid, setlab, REP, ddrop = load_repaints(obs)
        orig = np.array([row_of[int(i)] for i in wid])
        yr = year[orig]
        log(f'\n{obs} H3 C2: repaints {len(wid)} after id dedup (dropped as already seen: {ddrop}); originals in R {int(np.isin(orig, R).sum())}')
        c2 = core.c2_scores(X, R, orig, REP, codes, KS)
        pair = core.chord(np.sum(REP * X[orig], 1))
        O = np.unique(orig)
        rank = core.own_rank(REP, X, orig, O)
        nnA, _, nnE = core.knn(X, O, O, codes)
        mnA, mnE = float(np.median(nnA[:, 0])), float(np.median(nnE[:, 0]))
        subsets = {'P': setlab == 'P', 'P2': setlab == 'P2', 'pos': setlab == 'pos', 'today': setlab == 'today', 'today_pre1900': (setlab == 'today') & (yr < 1900),
                   'today_1900_1969': (setlab == 'today') & (yr >= 1900) & (yr < 1970), 'today_1970_on': (setlab == 'today') & (yr >= 1970), 'all': np.ones(len(wid), bool)}
        rows = []
        for sname, sel in subsets.items():
            base_row = {'set': sname, 'n': int(sel.sum()), 'pair_chord_median': float(np.median(pair[sel])), 'pair_chord_p10': pct(pair[sel], 10), 'pair_chord_p90': pct(pair[sel], 90),
                        'own_original_rank1_share': float((rank[sel] == 1).mean()), 'own_original_rank_median': float(np.median(rank[sel])),
                        'nn_chord_among_originals_median_all': mnA, 'nn_chord_among_originals_median_excl': mnE,
                        'pair_below_nn_median_all': float((pair[sel] < mnA).mean()), 'pair_below_nn_median_excl': float((pair[sel] < mnE).mean())}
            for (rule, k), (so, sr) in c2.items():
                rows.append({**base_row, 'rule': rule, 'k': k, 'S1_repaint_median': float(np.median(sr[sel])), 'S1_original_median': float(np.median(so[sel])),
                             'shift_median_paired': float(np.median(sr[sel] - so[sel])), 'auc_repaint_vs_original': core.auc(sr[sel], so[sel])})
        c2t = pd.DataFrame(rows)
        wcsv(c2t, f'{obs}_H3_C2.csv')
        wnpz(f'{obs}_H3_C2_perwork.npz', id=wid, set=setlab, year=yr, pair_chord=pair, own_original_rank=rank, originals_ids=ids[O],
             nn_chord_original_all=nnA[:, 0], nn_chord_original_excl=nnE[:, 0],
             **{f'S1_{w}_{rule}_k{k}': v for (rule, k), (so, sr) in c2.items() for w, v in (('original', so), ('repaint', sr))})
        with pd.option_context('display.width', 250, 'display.max_columns', 30):
            log(c2t[c2t.k == 15].round(4).to_string(index=False))
        log(f'  H3 written, {time.time() - t0:.0f}s')
    if 'H4' in PARTS:
        sizes, caps, lev4 = ints(args.h4_sizes), ints(args.h4_caps), ints(args.h4_levels)
        Lall = L
        cols = np.concatenate([R, Lall])
        rows, refs, mats = [], [], {'excl': [], 'all': []}
        log(f'\n{obs} H4: R {len(R)} stored originals, later {len(Lall)} (all later works); lines = percentiles of the held-out R works scored '
            f'leave-artist-out; later works scored under both rules; AUC later vs held-out R (leave-artist-out scores)')
        for cap in caps:
            log(f'  cap {cap or "none"}: attainable reference size {core.attainable(R, cap, codes)}')
        for size in sizes:
            for cap in caps:
                att = core.attainable(R, cap, codes)
                if att < size:
                    log(f'  size {size} cap {cap or "none"}: NOT DRAWN, attainable {att} < size')
                    rows.append({'size': size, 'cap': cap, 'draw': -1, 'note': f'infeasible: attainable {att}'})
                    continue
                for d in range(args.h4_draws):
                    rng = np.random.default_rng([args.h4_seed, size, cap, d])
                    ref = core.h4_reference(R, cap, size, rng, codes)
                    held = np.setdiff1d(R, ref)
                    da, _, do = core.knn(X, ref, np.concatenate([held, Lall]), codes)
                    nh = len(held)
                    per_art = np.bincount(codes[ref])
                    per_art = per_art[per_art > 0]
                    info = {'size': size, 'cap': cap, 'draw': d, 'n_artists': len(per_art), 'max_per_artist': int(per_art.max()),
                            'artists_at_cap': int((per_art == cap).sum()) if cap else -1, 'n_heldout': nh, 'n_later': len(Lall)}
                    for k in KS:
                        h_ex, l_ex, l_all = do[:nh, :k].mean(1), do[nh:, :k].mean(1), da[nh:, :k].mean(1)
                        for rule, lv in (('excl', l_ex), ('all', l_all)):
                            row = {**info, 'k': k, 'rule': rule, 'auc_later_vs_heldout': core.auc(lv, h_ex)}
                            for Lv in lev4:
                                c = pct(h_ex, Lv)
                                row[f'line{Lv}'] = c
                                row[f'later_share_over_line{Lv}'] = float((lv > c).mean())
                            rows.append(row)
                    kk = args.h4_save_k
                    pos_h = np.searchsorted(cols[:len(R)], held)
                    for rule, d_ in (('excl', do), ('all', da)):
                        m = np.full(len(cols), np.nan, np.float32)
                        m[pos_h] = d_[:nh, :kk].mean(1)
                        m[len(R):] = d_[nh:, :kk].mean(1)
                        mats[rule].append(m)
                    refs.append(np.pad(ids[ref], (0, max(sizes) - size), constant_values=-1))
                log(f'  size {size} cap {cap or "none"}: {args.h4_draws} draws done, {time.time() - t0:.0f}s')
        h4 = pd.DataFrame(rows)
        wcsv(h4, f'{obs}_H4_draws.csv')
        drawn = h4[h4.draw >= 0]
        agg = {c: ['median', 'min', 'max'] for c in ['auc_later_vs_heldout'] + [f'line{Lv}' for Lv in lev4] + [f'later_share_over_line{Lv}' for Lv in lev4] + ['n_artists', 'artists_at_cap']}
        summ = drawn.groupby(['size', 'cap', 'rule', 'k']).agg(agg)
        summ.columns = [f'{a}_{b}' for a, b in summ.columns]
        summ = summ.reset_index()
        wcsv(summ, f'{obs}_H4_summary.csv')
        sc = drawn[['size', 'cap']].drop_duplicates()
        keys = list(zip(sc['size'].astype(int), sc['cap'].astype(int)))
        wnpz(f'{obs}_H4_perwork_k{args.h4_save_k}.npz', col_ids=ids[cols], col_is_R=np.r_[np.ones(len(R), bool), np.zeros(len(Lall), bool)],
             draw_size_cap=np.array([(s, c, d) for s, c in keys for d in range(args.h4_draws)]), ref_ids=np.array(refs),
             S1_excl=np.array(mats['excl']), S1_all=np.array(mats['all']))
        with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 400):
            log(summ[summ.k == 15].round(4).to_string(index=False))
        log(f'  H4 written, {time.time() - t0:.0f}s')
    del X
log('done')
