"""SD image-variations decoder: a CLIP image embedding scaled to VARIATIONS_NORM replaces the pipeline's image-encoder output; noise comes from torch's global generator, seeded by the caller.
Usage: from generators.variations import generate; generate(content_img_path, emb, output_path)   (content path ignored)"""
import glob
import os

import numpy as np

_pipe = None
_held = {}


def _norm():
    return float(os.environ.get('VARIATIONS_NORM', '17.53'))


def _guidance():
    return float(os.environ.get('VARIATIONS_GUIDANCE', '3'))


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import StableDiffusionImageVariationPipeline
        path = sorted(glob.glob(f"{os.environ['HF_HOME']}/hub/models--lambdalabs--sd-image-variations-diffusers/snapshots/*"))[-1]
        _pipe = StableDiffusionImageVariationPipeline.from_pretrained(path, safety_checker=None, image_encoder=None, feature_extractor=None,
                                                                      local_files_only=True).to('cuda')
        _pipe.set_progress_bar_config(disable=True)
        _pipe._encode_image = lambda image, device, n, cfg: torch.cat([torch.zeros_like(_held['e']), _held['e']]) if cfg else _held['e']
        print(f"variations backend loaded: norm {_norm()}, guidance {_guidance()}", flush=True)
    return _pipe


def generate(content_img_path, emb, output_path):
    import torch
    from PIL import Image
    pipe = _load()
    v = np.asarray(emb, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v) * _norm()    # the search's unit vector at the record's length
    _held['e'] = torch.tensor(v, dtype=torch.float32, device='cuda').view(1, 1, -1)
    with torch.no_grad():
        image = pipe(image=[Image.new('RGB', (224, 224))], guidance_scale=_guidance(), num_inference_steps=50,
                     latents=torch.randn((1, 4, 64, 64), device='cuda')).images[0]    # noise from the global generator, seeded by the caller
    image.save(output_path)
    return output_path
