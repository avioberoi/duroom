"""Raw CLIP ViT-L/14 image vectors of every WikiArt work (OpenAI clip package, fp16 on GPU), saved with the ids.
Usage: clip_raw.py"""
import os

import clip
import numpy as np
import torch
from PIL import Image

ART = os.environ['WIKIART_DIR'] + '/artwork_data'
IMG = os.environ['WIKIART_DIR'] + '/artwork_images'
OUT = os.environ['DUROOM_OUT'] + '/reembed'

torch.set_float32_matmul_precision('high')
stored = np.load(f'{ART}/artwork_style_embeddings.npz')
ids = stored['ids']
rows = np.arange(len(ids))

clip_model, preprocess = clip.load('ViT-L/14', device='cuda')
clip_model.eval()


class Images(torch.utils.data.Dataset):
    def __len__(self):
        return len(rows)

    def __getitem__(self, i):
        return preprocess(Image.open(f'{IMG}/{ids[rows[i]]}.jpg').convert('RGB'))


raw = []
with torch.no_grad():
    for images in torch.utils.data.DataLoader(Images(), batch_size=256, num_workers=8):
        raw.append(clip_model.encode_image(images.cuda().half()).float().cpu().numpy())
raw = np.concatenate(raw)

os.makedirs(OUT, exist_ok=True)
np.savez(f'{OUT}/wikiart_catfm.npz', ids=ids, clip_raw=raw)
print('saved', raw.shape, flush=True)
