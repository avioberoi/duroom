"""Runs embed_images.py on DIR/list.txt -> DIR/emb.npz for each DIR.
Usage: python embed_dirs.py DIR [DIR ...]"""
import os
import subprocess
import sys

E = os.path.join(os.environ['DUROOM_REPO'], 'encode', 'embed_images.py')
for d in sys.argv[1:]:
    print('== embedding', d, flush=True)
    subprocess.run([sys.executable, E, '--list', f'{d}/list.txt', '--out', f'{d}/emb.npz', '--encoders', 'clip-vit-l-14', 'dinov3-l', 'siglip-so400m', 'aimv2-huge'], check=True)
print('all shards embedded', flush=True)
