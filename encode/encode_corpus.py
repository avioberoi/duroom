"""Encodes the WikiArt corpus with one observer encoder and saves the embeddings as .npz.
Usage: python encode_corpus.py --encoder {clip-vit-l-14,dinov3-l,siglip-so400m}"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import load_config, load_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True,
                   choices=['clip-vit-l-14', 'dinov3-l', 'siglip-so400m'])
    return p.parse_args()


def main(args):
    cfg = load_config()
    data = np.load(cfg['data']['wikiart_style'], allow_pickle=True)
    ids = data['ids']
    img_dir = Path(cfg['data']['wikiart_images'])
    print(f"Corpus: {len(ids)} images from {img_dir}")

    encode_fn, dim = load_encoder(args.encoder)
    out_path = Path(cfg['data']['observer_corpus'][args.encoder])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.zeros((len(ids), dim), dtype='float32')
    failed = 0
    for i, img_id in enumerate(tqdm(ids, desc=args.encoder)):
        try:
            embeddings[i] = encode_fn(str(img_dir / f"{img_id}.jpg"))
        except Exception as e:
            if i < 3:
                tqdm.write(f"  Failed {img_id}: {e}")
            failed += 1

    np.savez(out_path, ids=ids, embeddings=embeddings)
    print(f"Saved {len(ids)} embeddings ({failed} failed) to {out_path}")
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")


if __name__ == "__main__":
    main(parse_args())
