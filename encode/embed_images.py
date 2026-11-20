"""Embeds a list of images with the search encoders plus 'coherence' and 'content' (DINOv3 cosine to content_photo.png) into one .npz.
Usage: python embed_images.py --list paths.txt --out out.npz --encoders clip-vit-l-14 dinov3-l ..."""
import argparse
import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402


def load(name):
    # siglip2-so400m loads through encode_things, the rest through utils
    if name != 'siglip2-so400m':
        return utils.load_encoder(name)[0]
    import encode_things as et
    from PIL import Image
    fn, _ = et.load_encoder(name)
    return lambda p: fn(Image.open(p).convert('RGB'))


def unit_rows(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--encoders', nargs='+', required=True)
    a = ap.parse_args()
    paths = [line.strip() for line in open(a.list) if line.strip()]
    res, t0 = {'paths': np.array(paths)}, time.time()
    for name in a.encoders:
        if name == 'coherence':
            neg, enc, thr = utils.load_coherence_gate(utils.load_config())
            res['coherence'] = np.array([utils.coherence_score(enc(p), neg) for p in paths], dtype=np.float32)
            res['coherence_threshold_cfg'] = np.float32(thr)
        elif name == 'content':
            dino = load('dinov3-l')
            if 'emb_dinov3-l' not in res:
                res['emb_dinov3-l'] = unit_rows([dino(p) for p in paths])
            c = np.asarray(dino(str(utils.get_reference_image())), dtype=np.float64)
            res['content_sim'] = (res['emb_dinov3-l'].astype(np.float64) @ (c / np.linalg.norm(c))).astype(np.float32)
        else:
            if f'emb_{name}' not in res:
                fn = load(name)  # encode_things.load_encoder does not cache; a per-image call would reload the model
                res[f'emb_{name}'] = unit_rows([fn(p) for p in paths])
        print(name, 'done', len(paths), 'images', round(time.time() - t0, 1), 's', flush=True)
    np.savez(a.out, **res)
