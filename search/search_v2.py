"""Committee novelty search over CLIP-coordinate candidates (CMA-ES, random or MAP-Elites): writes archive.npz and the candidate files to the run directory.
Usage: python search_v2.py --space sphere --start seedrow --start-pool NPZ --pushforward NPZ [other flags: see parse_args]"""
import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np
import cma
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (load_config, get_output_dir, load_wikiart, load_observer_corpus,
                   get_reference_image, load_encoder, pick_start,
                   load_coherence_gate, coherence_score)
from novelty_scorer import NoveltyScorer
from generators import generate, set_default_backend


ANGLE_BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 100.0))  # mode random: degrees from x0; draw shares 1/6, 1/6, 4/6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--metric', default='euclidean',
                   choices=['euclidean', 'cosine', 'mahalanobis', 'mahalanobis_local'])
    p.add_argument('--start', default='p90')  # p<N>, centroid, boundary, frontier, row:N, or seedrow (one start per seed from --start-pool, rng(seed + 31337))
    p.add_argument('--aggregation', default='mean', choices=['min', 'mean'])
    p.add_argument('--maxiter', type=int, default=100)
    p.add_argument('--popsize', type=int, default=32)
    p.add_argument('--k', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', default=None, help='Config YAML path')
    p.add_argument('--out-root', default=None)
    p.add_argument('--tag', default='run')
    p.add_argument('--random-fitness', action='store_true')  # tell CMA-ES N(0,1) values instead of the novelty fitness
    p.add_argument('--project-shell', nargs=2, type=float, default=None, metavar=('R1', 'R2'))  # rescale each candidate's length into [R1, R2] before decoding
    p.add_argument('--pushforward', default=None)  # npz whose emb_<observer> rows join every observer's kNN index (euclidean only)
    p.add_argument('--gate-threshold', type=float, default=None)  # coherence threshold instead of the config's
    p.add_argument('--mode', default='cma', choices=['cma', 'random', 'mapelites'])  # random: angle-binned draws around x0, no CMA-ES; mapelites: grid over content similarity and angle
    p.add_argument('--log-candidates', action='store_true')
    p.add_argument('--random-norm', type=float, default=16.242)
    p.add_argument('--norm-from-push', action='store_true')  # z = (raw - median_P) / MAD_P, fixed at the start, instead of the running MAD
    p.add_argument('--admit-q99', action='store_true')  # admit iff raw > P's q99 for every observer
    p.add_argument('--delta', type=float, default=None)  # no admission within DINOv3 L2 distance D of an archived item
    p.add_argument('--band', nargs=2, type=float, default=None, metavar=('LO', 'HI'))  # feasible only if the DINOv3 cosine to the content photo is in [LO, HI]
    p.add_argument('--warmup', type=int, default=0)  # score and tell, but admit nothing, for the first W iterations
    p.add_argument('--no-early-stop', action='store_true')
    p.add_argument('--render-seeded', action='store_true')  # torch.manual_seed(seed * 1000000 + (it + 1) * 1000 + i) before each render
    p.add_argument('--space', default='style', choices=['style', 'sphere'])  # sphere: unit sphere of CLIP ViT-L/14 image embeddings, candidates projected to norm 1
    p.add_argument('--start-pool', default=None)
    p.add_argument('--sigma-cap', type=float, default=None)  # CMA-ES sigma is clipped to F * sigma0 before each ask
    p.add_argument('--angles-from', default=None)  # mode random: draw each requested angle from this npz's 'angle_deg'
    p.add_argument('--fidelity-max', type=float, default=None)
    p.add_argument('--reanchor', type=int, default=None)  # after each tell, set the mean to the unit mean of the M best feasible renders' CLIP vectors
    p.add_argument('--sigma-path', default=None)  # npz with log_sigma; from round 1, before each ask, sigma is set to the logged value of the previous round, then capped as usual
    p.add_argument('--noise-seed', type=int, default=None)  # the decoder noise is seeded from this instead of --seed (default: --seed)
    return p.parse_args()


class MADNormalizer:
    def __init__(self):
        self.values = []

    def update(self, x):
        self.values.append(x)

    def normalize(self, x):
        if len(self.values) < 2:
            return x
        med = np.median(self.values)
        mad = np.median(np.abs(np.array(self.values) - med)) + 1e-8
        return (x - med) / mad


def _savez_atomic(dest, /, **arrays):
    """Write-then-rename, so an interrupt during the write leaves the previous checkpoint whole."""
    tmp = Path(f"{dest}.tmp")    # dest is positional-only: the candidate files have a field named 'path'
    with open(tmp, 'wb') as fh:    # a file object, so numpy does not append .npz
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)


def save_checkpoint(out, archive, log, start_idx, args, observer_names, extra=None, name="archive.npz"):
    emb_keys = [k for k in archive[0] if k.startswith('emb_')]
    extra_keys = sorted({k for a in archive for k in a if k.startswith(('raw_', 'push_share_', 'content_sim', 'angle_deg', 'z_', 'fidelity'))})
    _savez_atomic(out / name,
                  style_embs=np.stack([a['style'] for a in archive]),
                  paths=np.array([a['path'] for a in archive]),
                  novelties=np.array([a['nov'] for a in archive]),
                  coherences=np.array([a['coh'] for a in archive]),
                  start_idx=start_idx, metric=args.metric, aggregation=args.aggregation,
                  observer_names=np.array(observer_names),
                  flags=np.array(repr(vars(args))),
                  **{k: np.stack([a[k] for a in archive]) for k in emb_keys},
                  **{k: np.array([a.get(k, np.nan) for a in archive], dtype=float) for k in extra_keys},
                  **{f"log_{k}": np.array(v) for k, v in log.items()},
                  **(extra or {}))


def push_null(index, P, offset, k, artist=None):
    """Leave-one-out k-NN mean distance of each P row (own row dropped; with artist, same-artist rows too, as a candidate's artist is never in P)."""
    d = np.empty(len(P))
    extra = 0 if artist is None else 64
    for i in range(len(P)):    # one query per call: batched GPU queries are not used
        D, I = index.search(P[i:i + 1], k + 1 + extra)
        if artist is None:
            own = np.flatnonzero(I[0] == offset + i)
            keep = np.delete(np.arange(k + 1), own[0] if len(own) else 0)  # a duplicate row can take its place: drop the first
        else:
            same = [j for j, idx in enumerate(I[0]) if idx == offset + i or (offset <= idx < offset + len(P) and artist[idx - offset] == artist[i])]
            keep = np.delete(np.arange(len(I[0])), same)[:k]
        d[i] = np.mean(np.sqrt(np.maximum(D[0][keep], 0)))
    return d


def main(args):
    np.random.seed(args.seed)
    rng_fit = np.random.default_rng(args.seed + 7919)
    cfg = load_config(args.config)
    backend = cfg.get('generator', {}).get('backend', 'variations')
    set_default_backend(backend)
    if args.out_root:
        out = Path(args.out_root) / f"{args.tag}_committee_{args.metric}_{args.aggregation}_{args.start}"
        out.mkdir(parents=True, exist_ok=True)
    else:
        prefix = cfg['data'].get('run_prefix', '')
        tag = f"{prefix}_" if prefix else ""
        out = get_output_dir(cfg, f"{tag}committee_{args.metric}_{args.aggregation}_{args.start}")
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)

    def project(s):
        s = np.asarray(s, dtype=np.float64)
        if args.space == 'sphere':
            return s / np.linalg.norm(s)
        if args.project_shell is None:
            return s
        n = np.linalg.norm(s)
        return s * (np.clip(n, args.project_shell[0], args.project_shell[1]) / n)

    if args.space == 'sphere':  # the unit sphere of raw CLIP image embeddings; the backend decodes at the config's length
        style_corpus = load_observer_corpus(cfg, 'clip-vit-l-14', normalize=True)
        os.environ.setdefault('VARIATIONS_NORM', str(cfg.get('generator', {}).get('norm', 17.53)))
        os.environ.setdefault('VARIATIONS_GUIDANCE', str(cfg.get('generator', {}).get('guidance', 3)))
        args.random_norm = 1.0
    else:
        style_corpus = load_wikiart(cfg, normalize=False)
    if args.start == 'seedrow':  # one start per seed from the pool
        pool_rows = np.load(args.start_pool, allow_pickle=True)['rows_full']  # rows of the unfiltered observer file, as style_corpus is indexed here
        start_idx = int(pool_rows[np.random.default_rng(args.seed + 31337).integers(len(pool_rows))])
        x0 = style_corpus[start_idx].copy()
    elif args.start.startswith('row:'):
        start_idx = int(args.start[4:])
        x0 = style_corpus[start_idx].copy()
    else:
        start_idx, x0 = pick_start(style_corpus, args.start)
    content = get_reference_image()
    neg_emb, coh_encode, coh_threshold = load_coherence_gate(cfg)
    if args.gate_threshold is not None:
        coh_threshold = args.gate_threshold

    observer_names = cfg['committee']['observers']
    DINO = 'dinov3-l'
    need_null = args.norm_from_push or args.admit_q99 or args.mode == 'mapelites'
    log_all = args.mode != 'cma' or args.log_candidates  # every candidate scored and written (all_draws / all_candidates.npz)
    need_cs = args.band is not None or log_all  # content similarity: band, candidate files, MAP-Elites measure
    if args.mode == 'mapelites' and not args.norm_from_push:
        raise SystemExit('--mode mapelites needs --norm-from-push: an elite is replaced only by a higher score on a fixed scale')
    if need_null and not args.pushforward:
        raise SystemExit('--norm-from-push, --admit-q99 and --mode mapelites need --pushforward')
    if (need_cs or args.delta is not None) and DINO not in observer_names:
        raise SystemExit(f'--band, --delta and modes random and mapelites read the {DINO} observer embedding')
    if args.noise_seed is not None and not args.render_seeded:
        raise SystemExit('--noise-seed needs --render-seeded: without it no render is seeded at all')
    if args.render_seeded and (args.popsize > 1000 or args.maxiter > 998):
        raise SystemExit('--render-seeded: the seed rule is unique only for popsize <= 1000 and maxiter <= 998')
    push = np.load(args.pushforward) if args.pushforward else None
    if push is not None and args.metric != 'euclidean':
        raise SystemExit('--pushforward is implemented for the euclidean metric only')
    n_push = 0
    observers = {}
    norm_extra = {}
    for name in observer_names:
        encode_fn, dim = load_encoder(name)
        obs_corpus = load_observer_corpus(cfg, name)
        scorer = NoveltyScorer(obs_corpus, args.metric, args.k,
                               rank=cfg['whitening']['rank'],
                               n_clusters=cfg['whitening']['n_clusters'])
        null = None
        if push is not None:
            P = np.ascontiguousarray(push[f'emb_{name}'], dtype='float32')
            n_before = scorer.index.ntotal
            scorer.index.add(P)
            n_push = len(P)
            if need_null:
                d = push_null(scorer.index, P, n_before, args.k, push['artist'] if 'artist' in push.files else None)
                med = float(np.median(d))
                null = (med, float(np.median(np.abs(d - med))), float(np.percentile(d, 99)))
                norm_extra.update({f'norm_median_{name}': null[0], f'norm_mad_{name}': null[1], f'norm_q99_{name}': null[2]})
                print(f"P null {name}: median={null[0]:.6f} MAD={null[1]:.6f} q99={null[2]:.6f} (leave-one-out {args.k}-NN "
                      f"mean distance of {len(P)} rows; {n_before} rows were in the index before P; "
                      f"{'same-artist rows excluded' if 'artist' in push.files else 'row itself excluded only'})")
        observers[name] = {'encode': encode_fn, 'dim': dim, 'scorer': scorer, 'mad': MADNormalizer(), 'null': null}

    fid_encode = None
    if args.space == 'sphere':  # render-to-request chord in the coordinate lens, logged per item, never gated
        try:
            fid_encode = observers['clip-vit-l-14']['encode'] if 'clip-vit-l-14' in observers else load_encoder('clip-vit-l-14')[0]
        except Exception as e:  # the render environment may lack the CLIP package and its cache; the exact value is computed offline
            print(f"fidelity encoder unavailable here ({type(e).__name__}: {e}); fidelity_clip is computed offline from embed_images", flush=True)
    if args.fidelity_max is not None and fid_encode is None:
        raise SystemExit('--fidelity-max needs the fidelity encoder: --space sphere with a loadable CLIP encoder')
    if args.reanchor is not None and (args.reanchor < 1 or fid_encode is None or args.mode != 'cma'):
        raise SystemExit('--reanchor needs M >= 1, --space sphere with a loadable CLIP encoder, and --mode cma')

    def fidelity(path, s):
        r = np.asarray(fid_encode(path), dtype=np.float64)
        u = np.asarray(s, dtype=np.float64)
        fidelity.last = (r / np.linalg.norm(r)).astype(np.float32)  # the render's own point on the sphere
        return float(np.sqrt(max(2.0 - 2.0 * float(r @ (u / np.linalg.norm(u))), 0.0)))

    if need_cs:  # embed_images.py 'content': DINOv3 of the content photo, unit, float64; item rows unit in float32
        c = np.asarray(observers[DINO]['encode'](str(content)), dtype=np.float64)
        c_unit = c / np.linalg.norm(c)

    def content_sim(e):
        e = np.asarray(e, dtype=np.float32).reshape(1, -1)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return float((e.astype(np.float64) @ c_unit).astype(np.float32)[0])

    def band_violation(cs):
        return max(args.band[0] - cs, cs - args.band[1], 0.0) if args.band is not None else 0.0

    def near_archived(entry):  # --delta: DINOv3 L2 distance to every archived item, the start included
        A = np.stack([a[f'emb_{DINO}'] for a in archive]).astype(np.float64)
        return float(np.min(np.linalg.norm(A - np.asarray(entry[f'emb_{DINO}'], dtype=np.float64), axis=1))) <= args.delta

    def render(style, path, it, i):
        if args.render_seeded:
            import torch
            torch.manual_seed((args.seed if args.noise_seed is None else args.noise_seed) * 1000000 + (it + 1) * 1000 + i)
        generate(content, style, path)

    print(f"Style corpus: {len(style_corpus)}, Start: {args.start} (idx={start_idx})")
    print(f"Observers: {observer_names}, Aggregation: {args.aggregation}")
    print(f"Coherence gate: threshold={coh_threshold}")
    print(f"Flags: {vars(args)}; pushforward rows per observer: {n_push}")
    if args.render_seeded:
        print(f"Render seed: torch.manual_seed(S * 1000000 + (iteration + 1) * 1000 + index), S = {args.seed if args.noise_seed is None else args.noise_seed} "
              f"({'--seed' if args.noise_seed is None else '--noise-seed'}); the start is iteration -1, index 0")
    if args.mode == 'mapelites' and (args.admit_q99 or args.delta is not None or args.warmup):
        print("mapelites: --admit-q99, --delta and --warmup are ignored (a candidate enters its cell if feasible and the cell is "
              "empty or holds a lower objective)")

    archive = []
    start_path = out / "start.png"
    x0 = project(x0).astype('float32')
    render(x0, start_path, -1, 0)
    start_entry = {'style': x0, 'path': str(start_path), 'nov': 0.0}
    for name, obs in observers.items():
        emb = obs['encode'](start_path)
        if args.mode == 'mapelites':  # static index (P only): an elite's objective never changes
            start_entry[f'raw_{name}'] = float(obs['scorer'].score(emb))
        else:
            obs['scorer'].add(emb)
        start_entry[f'emb_{name}'] = emb
    start_entry['coh'] = coherence_score(coh_encode(start_path), neg_emb)
    if fid_encode is not None:
        start_entry['fidelity_clip'] = fidelity(start_path, x0)
    archive.append(start_entry)
    if need_cs:
        start_entry['content_sim'] = content_sim(start_entry[f'emb_{DINO}'])
    if log_all:
        start_entry.update(angle_deg=0.0, angle_bin=0, it=-1, slot=0, admitted=True, feasible=bool(
            start_entry['coh'] >= coh_threshold and band_violation(start_entry['content_sim']) == 0.0))
    x0u = x0.astype(np.float64) / np.linalg.norm(x0.astype(np.float64))

    def angle_to_x0(s):  # atan2 form: arccos loses precision near 0 degrees
        s = np.asarray(s, dtype=np.float64)
        return float(np.degrees(np.arctan2(np.linalg.norm(s - (s @ x0u) * x0u), s @ x0u)))

    def angle_bin(a):  # 0: [0, 30), 1: [30, 60), 2: [60, 100], 3: beyond 100 degrees
        return 0 if a < 30 else 1 if a < 60 else 2 if a <= 100 else 3

    def draw_random(j):  # draw number j: bin from the schedule, t uniform in it, x = R (cos t x0u + sin t u), u orthogonal to x0
        if angle_pool is not None:  # the requested angle follows a CMA arm's own logged candidate angles
            t = float(rng.choice(angle_pool))
        else:
            lo, hi = ANGLE_BINS[bin_schedule[j]]
            t = rng.uniform(lo, hi)
        g = rng.standard_normal(len(x0))
        u = g - (g @ x0u) * x0u
        u /= np.linalg.norm(u)
        return args.random_norm * (np.cos(np.radians(t)) * x0u + np.sin(np.radians(t)) * u), t

    grid = {}  # MAP-Elites: (cell_m1, cell_m2) -> elite entry

    def cell_of(e):  # m1 content similarity in [0, 1] by 20 cells, m2 angle to x0 in [0, 180] degrees by 10 cells
        return (int(np.clip(np.floor(e['content_sim'] * 20), 0, 19)), int(np.clip(np.floor(e['angle_deg'] / 18.0), 0, 9)))

    sigma0 = cfg['cma']['sigma0']
    rng = np.random.default_rng(args.seed + 104729)  # random and mapelites draws; CMA-ES seeds itself from --seed
    n_draws = args.maxiter * args.popsize
    bin_schedule = rng.permutation(np.repeat([0, 1, 2], [n_draws // 6, n_draws // 6, n_draws - 2 * (n_draws // 6)]))
    angle_pool = None
    if args.angles_from:
        za = np.load(args.angles_from, allow_pickle=True)
        angle_pool = np.asarray(za['angle_deg'], dtype=np.float64)
        angle_pool = angle_pool[np.isfinite(angle_pool) & (angle_pool > 0)]
        print(f"mode random: angles drawn from {args.angles_from}: {len(angle_pool)} candidate angles, median {np.median(angle_pool):.1f} deg")
    n_sigma_capped = 0
    norm_extra.setdefault('n_reanchored', 0)
    es = None if args.mode != 'cma' else cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        'popsize': args.popsize, 'CMA_diagonal': cfg['cma']['diagonal'],
        'maxiter': args.maxiter, 'seed': args.seed,
        'tolfun': cfg['cma']['tolfun'], 'tolx': cfg['cma']['tolx'],
        'verbose': -9,
    })
    sig_path = np.asarray(np.load(args.sigma_path, allow_pickle=True)['log_sigma'], dtype=np.float64) if args.sigma_path else None
    assert sig_path is None or (args.mode == 'cma' and len(sig_path) >= args.maxiter - 1 and np.isfinite(sig_path).all()), '--sigma-path needs --mode cma and a finite log_sigma of at least maxiter - 1 rounds'

    log = {'iter': [], 'nov': [], 'archive': [], 'sigma': [], 'sigma_asked': [], 'coherence': [], 'n_incoherent': [],
           'norm_asked': [], 'norm_decoded': []}
    if args.band is not None:
        log['n_out_of_band'] = []
    if args.delta is not None and args.mode != 'mapelites':
        log['n_delta_blocked'] = []
    if args.mode == 'mapelites':
        log.update(occupied=[], qd_score=[], n_inserted=[])
    threshold_pct = cfg['novelty']['threshold_percentile']
    aggregate = min if args.aggregation == 'min' else np.mean
    es_patience = cfg['cma']['early_stop_patience']
    es_min_accept = cfg['cma']['early_stop_min_acceptance']
    draws = []  # every candidate when log_all, for all_draws.npz (random) or all_candidates.npz (cma, mapelites)

    def zfixed(obs, raw):
        return (raw - obs['null'][0]) / (obs['null'][1] + 1e-8)

    def zscore(obs, raw):  # P's fixed scale (--norm-from-push), else the running MAD
        if args.norm_from_push:
            return zfixed(obs, raw)
        obs['mad'].update(raw)
        return obs['mad'].normalize(raw)

    def sigma_now():  # step size of the draws: CMA-ES's; sigma0 for MAP-Elites mutation; none (nan) for angle draws
        if es is not None:
            return es.sigma
        return sigma0 if args.mode == 'mapelites' else float('nan')

    if args.mode == 'mapelites':  # the start is scored first and seeds its own cell
        for n, o in observers.items():
            start_entry[f'z_{n}'] = zfixed(o, start_entry[f'raw_{n}'])
        start_entry['nov'] = start_entry['z_agg'] = float(aggregate([start_entry[f'z_{n}'] for n in observers]))
        start_entry['passes_q99'] = all(start_entry[f'raw_{n}'] > o['null'][2] for n, o in observers.items())
        start_entry['cell'] = cell_of(start_entry)
        grid[start_entry['cell']] = start_entry

    def row_fields(rows):  # typed per-row fields of the candidate logs (absent without log_all)
        if not log_all:
            return {}
        f = dict(iteration=np.array([r['it'] for r in rows]), index=np.array([r['slot'] for r in rows]),
                 feasible=np.array([r['feasible'] for r in rows]), admitted=np.array([r['admitted'] for r in rows]),
                 angle_bin=np.array([r['angle_bin'] for r in rows]))
        if need_null:
            f['passes_q99'] = np.array([r.get('passes_q99', False) for r in rows])
        return f

    def save():
        rows = archive
        if args.mode == 'mapelites':  # row 0 stays the start (the evaluation chain skips it); then every elite by cell
            elites = [grid[c] for c in sorted(grid)]
            rows = [start_entry] + [e for e in elites if e is not start_entry]
        extra = dict(norm_extra, **row_fields(rows))
        if args.mode == 'mapelites':
            extra.update(cell_m1=np.array([r['cell'][0] for r in rows]), cell_m2=np.array([r['cell'][1] for r in rows]),
                         is_elite=np.array([grid.get(r['cell']) is r for r in rows]))
        save_checkpoint(out, rows, log, start_idx, args, observer_names, extra)
        if log_all:  # archive-format names plus the candidate-file names
            d = [start_entry] + draws
            save_checkpoint(out, d, log, start_idx, args, observer_names, dict(
                norm_extra, style=np.stack([r['style'] for r in d]), path=np.array([r['path'] for r in d]),
                render_clip=np.stack([np.asarray(r.get('render_clip', np.full(len(x0), np.nan)), dtype=np.float32) for r in d]),
                coherence=np.array([r['coh'] for r in d]), **row_fields(d)),
                name="all_draws.npz" if args.mode == 'random' else "all_candidates.npz")

    def _save_on_signal(signum, frame):
        print(f"\nCaught signal {signum}, saving checkpoint...")
        save()
        print(f"Saved {len(archive)} images to {out}")
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, _save_on_signal)
    signal.signal(signal.SIGUSR1, _save_on_signal)

    print(f"CMA-ES: sigma0={sigma0}, pop={args.popsize}, metric={args.metric}, k={args.k}, space={args.space}"
          + (f" (a sample of sigma0 sits about {np.degrees(np.arctan(sigma0 * np.sqrt(len(x0)))):.0f} degrees from the mean)" if args.space == 'sphere' else ''))
    if es is None:
        print(f"Mode {args.mode}: CMA-ES is not used")
    if args.mode == 'random':
        print(f"Angle bins {ANGLE_BINS}: {np.bincount(bin_schedule, minlength=3).tolist()} draws in a seeded random order, "
              f"at norm {args.random_norm}")

    try:
        for it in tqdm(range(args.maxiter), desc="Committee"):
            if es is not None and es.stop():
                print(f"CMA-ES stopped at iter {it}: {es.stop()}")
                break

            thetas = [None] * args.popsize
            if es is not None:
                if sig_path is not None and it > 0:  # replay a logged step size (its sigma after the previous round's tell); the cap below then applies
                    es.sigma = float(sig_path[it - 1])
                if args.sigma_cap is not None and es.sigma > args.sigma_cap * sigma0:
                    es.sigma = args.sigma_cap * sigma0
                    n_sigma_capped += 1
                    norm_extra['n_sigma_capped'] = n_sigma_capped
                log['sigma_asked'].append(float(es.sigma))  # the step size the samples of this round were drawn with
                solutions = es.ask()
            elif args.mode == 'random':
                solutions, thetas = zip(*[draw_random(it * args.popsize + i) for i in range(args.popsize)])
            else:  # mapelites: a uniformly drawn occupied cell's elite plus sigma0 N(0, I), per slot
                keys = sorted(grid)
                solutions = [grid[keys[j]]['style'] + sigma0 * rng.standard_normal(len(x0))
                             for j in rng.integers(len(keys), size=args.popsize)]
            candidates = []  # (told solution, entry, nov, coh, feasible)
            norms_asked, norms_decoded = [], []

            for i, s in enumerate(solutions):
                s_used = project(s)
                norms_asked.append(float(np.linalg.norm(s)))
                norms_decoded.append(float(np.linalg.norm(s_used)))
                path = img_dir / f"iter{it:04d}_{i:02d}.png"
                try:
                    render(np.asarray(s_used, dtype='float32'), path, it, i)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        import torch; torch.cuda.empty_cache()
                        tqdm.write(f"  OOM on iter {it} candidate {i}, skipping")
                        continue
                    raise

                entry = {'style': np.asarray(s_used, dtype='float32'), 'path': str(path)}
                for name, obs in observers.items():
                    entry[f'emb_{name}'] = obs['encode'](path)

                coh = coherence_score(coh_encode(path), neg_emb)
                feasible = coh >= coh_threshold
                if fid_encode is not None:
                    entry['fidelity_clip'] = fidelity(path, s_used)
                    entry['render_clip'] = fidelity.last
                    if args.fidelity_max is not None and entry['fidelity_clip'] > args.fidelity_max:  # the request must be honoured
                        feasible = False
                if need_cs:
                    entry['content_sim'] = content_sim(entry[f'emb_{DINO}'])
                    feasible = feasible and band_violation(entry['content_sim']) == 0.0
                if log_all:
                    entry['angle_deg'] = thetas[i] if thetas[i] is not None else angle_to_x0(s_used)
                    entry.update(angle_bin=angle_bin(entry['angle_deg']), it=it, slot=i, feasible=bool(feasible), admitted=False)

                if feasible or log_all:  # log_all: raw scores of every candidate, for the candidate file
                    raw_scores = {}
                    for name, obs in observers.items():
                        raw = obs['scorer'].score(entry[f'emb_{name}'])
                        entry[f'raw_{name}'] = float(raw)
                        if n_push:
                            q = np.asarray(entry[f'emb_{name}'], dtype='float32').reshape(1, -1)
                            kk = min(args.k, obs['scorer'].index.ntotal)
                            _, I = obs['scorer'].index.search(q, kk)
                            entry[f'push_share_{name}'] = float(np.mean(I[0] < n_push))
                        if feasible:
                            raw_scores[name] = zscore(obs, raw)
                        if log_all:  # an infeasible candidate's z exists only on the fixed scale
                            entry[f'z_{name}'] = raw_scores[name] if feasible else (
                                zfixed(obs, raw) if args.norm_from_push else float('nan'))
                    nov = float(aggregate(list(raw_scores.values()))) if feasible else 0.0
                    if log_all:
                        zs = [entry[f'z_{n}'] for n in observers]
                        entry['z_agg'] = float(aggregate(zs)) if np.all(np.isfinite(zs)) else float('nan')
                        if need_null:
                            entry['passes_q99'] = all(entry[f'raw_{n}'] > o['null'][2] for n, o in observers.items())
                else:
                    nov = 0.0

                entry['nov'] = nov
                entry['coh'] = coh
                candidates.append((s_used, entry, nov, coh, feasible))
                if log_all:
                    draws.append(entry)

            if not candidates:
                tqdm.write(f"  All candidates OOM'd at iter {it}, stopping")
                break

            feasible_fits = [-c[2] for c in candidates if c[4]]
            f_offset = np.median(feasible_fits) if feasible_fits else 0.0
            fitnesses = []
            for s, entry, nov, coh, feasible in candidates:
                if feasible:
                    fitnesses.append(-nov)
                else:  # gate and band violations, each squared (band_violation is 0 without --band); the fidelity excess is added
                    fitnesses.append(f_offset + max(coh_threshold - coh, 0.0) ** 2
                                     + band_violation(entry.get('content_sim', 0.0)) ** 2
                                     + (max(entry.get('fidelity_clip', 0.0) - args.fidelity_max, 0.0) ** 2 if args.fidelity_max is not None else 0.0))
            if args.random_fitness:
                fitnesses = [float(v) for v in rng_fit.standard_normal(len(candidates))]

            if es is not None:
                es.tell([c[0] for c in candidates], fitnesses)
                if args.space == 'sphere':  # keep the search mean on the sphere, so sigma stays an angle
                    es.mean = es.mean / np.linalg.norm(es.mean)
                if args.reanchor:  # the mean moves to where the best feasible renders actually are, not where they were asked for
                    order = sorted(range(len(candidates)), key=lambda i: fitnesses[i])  # told fitness: novelty, or the random draw under --random-fitness
                    best = [candidates[i][1]['render_clip'] for i in order if candidates[i][4] and 'render_clip' in candidates[i][1]][:args.reanchor]
                    if best:
                        m = np.mean(np.asarray(best, dtype=np.float64), axis=0)
                        es.mean = m / np.linalg.norm(m)
                    norm_extra['n_reanchored'] = norm_extra.get('n_reanchored', 0) + (1 if best else 0)

            threshold = np.percentile([a['nov'] for a in archive], threshold_pct) if len(archive) > 1 else 0
            n_blocked = n_inserted = 0
            for s, entry, nov, coh, feasible in candidates:
                if args.mode == 'mapelites':  # one elite per cell; a cell's objective never decreases
                    if feasible:
                        entry['cell'] = cell_of(entry)
                        if entry['cell'] not in grid or nov > grid[entry['cell']]['nov']:
                            grid[entry['cell']] = entry
                            entry['admitted'] = True
                            n_inserted += 1
                    continue
                admit = feasible and it >= args.warmup and (
                    all(entry[f'raw_{n}'] > o['null'][2] for n, o in observers.items()) if args.admit_q99 else nov > threshold)
                if admit and args.delta is not None and near_archived(entry):
                    n_blocked += 1
                    admit = False
                if admit:
                    for name, obs in observers.items():
                        obs['scorer'].add(entry[f'emb_{name}'])
                    archive.append(entry)
                    entry['admitted'] = True

            cohs = [c[3] for c in candidates]
            n_bad = sum(1 for c in candidates if not c[4])
            feasible_novs = [c[2] for c in candidates if c[4]]
            log['iter'].append(it)
            log['nov'].append(max(feasible_novs) if feasible_novs else 0.0)
            log['archive'].append(len(grid) if args.mode == 'mapelites' else len(archive))
            log['sigma'].append(sigma_now())
            log['coherence'].append(np.mean(cohs))
            log['n_incoherent'].append(n_bad)
            log['norm_asked'].append(float(np.median(norms_asked)))
            log['norm_decoded'].append(float(np.median(norms_decoded)))
            if args.band is not None:
                log['n_out_of_band'].append(sum(1 for c in candidates if band_violation(c[1]['content_sim']) > 0))
            if 'n_delta_blocked' in log:
                log['n_delta_blocked'].append(n_blocked)
            if args.mode == 'mapelites':
                log['occupied'].append(len(grid))
                log['qd_score'].append(float(sum(e['nov'] for e in grid.values())))
                log['n_inserted'].append(n_inserted)

            save()
            if it % 10 == 0:
                best_nov = max(feasible_novs) if feasible_novs else 0.0
                tqdm.write(f"[{it:3d}] archive={log['archive'][-1]:4d} nov={best_nov:.4f} "
                           f"sigma={sigma_now():.4f} coh={np.mean(cohs):.3f} bad={n_bad}/{len(candidates)} "
                           f"norm={np.median(norms_asked):.1f}->{np.median(norms_decoded):.1f}")
                extra_logs = {k: log[k][-1] for k in ('n_out_of_band', 'n_delta_blocked', 'occupied', 'qd_score') if k in log}
                if extra_logs:
                    tqdm.write(f"      {extra_logs}")

            if es is not None and not args.no_early_stop and len(log['archive']) >= es_patience:
                recent = log['archive'][-es_patience:]
                additions = [recent[j] - recent[j-1] for j in range(1, len(recent))]
                rate = np.mean(additions)
                if rate < es_min_accept:
                    tqdm.write(f"Early stop at iter {it}: acceptance rate {rate:.2f} < {es_min_accept} "
                               f"over last {es_patience} iters")
                    break

    except (Exception, KeyboardInterrupt) as e:
        print(f"\nInterrupted: {e}")
    finally:
        save()
        print(f"Final: {len(archive)} images, {len(log['iter'])} iters, sigma={sigma_now():.4f}, sigma capped on {n_sigma_capped} iterations")
        if 'n_delta_blocked' in log:
            print(f"Admissions blocked by --delta {args.delta}: {sum(log['n_delta_blocked'])}")
        if args.mode == 'mapelites':
            print(f"MAP-Elites: {len(grid)} occupied cells, QD score (sum of objectives) {sum(e['nov'] for e in grid.values()):.4f}")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main(parse_args())
