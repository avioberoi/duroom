"""Embeds an image directory with one vision encoder and saves the embeddings as .npz.
Usage: python encode_things.py --encoder NAME [--img-dir DIR] [--output-dir DIR] [--batch-size N]"""
import argparse
import time
from pathlib import Path
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


ENCODER_REGISTRY = {
    'clip-vit-l-14': {
        'lib': 'open_clip',
        'model': 'ViT-L-14',
        'pretrained': 'openai',
        'dim': 768,
    },
    'clip-bigg-14': {
        'lib': 'open_clip',
        'model': 'ViT-bigG-14',
        'pretrained': 'laion2b_s39b_b160k',
        'dim': 1280,
    },
    'siglip2-so400m': {
        'lib': 'open_clip',
        'model': 'ViT-SO400M-14-SigLIP2',
        'pretrained': 'webli',
        'dim': 1152,
    },
    'siglip2-giant': {
        'lib': 'open_clip',
        'model': 'ViT-gopt-16-SigLIP2-384',
        'pretrained': 'webli',
        'dim': 1536,
    },
    'dfn5b-vit-h': {
        'lib': 'open_clip',
        'model': 'ViT-H-14-378-quickgelu',
        'pretrained': 'dfn5b',
        'dim': 1024,
    },
    'dinov2-l': {
        'lib': 'torch_hub',
        'model': 'dinov2_vitl14',
        'dim': 1024,
    },
    'dinov2-g': {
        'lib': 'torch_hub',
        'model': 'dinov2_vitg14',
        'dim': 1536,
    },
    'aimv2-huge': {
        'lib': 'timm',
        'model': 'aimv2_huge_patch14_224',
        'dim': 1536,
    },
}


def load_encoder(name):
    """Returns (encode_fn, dim); encode_fn maps a PIL image to a unit vector."""
    spec = ENCODER_REGISTRY[name]

    if spec['lib'] == 'open_clip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec['model'], pretrained=spec['pretrained'])
        model = model.eval().cuda()

        @torch.no_grad()
        def encode_fn(img):
            x = preprocess(img).unsqueeze(0).cuda()
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().astype('float32').flatten()
        return encode_fn, spec['dim']

    elif spec['lib'] == 'torch_hub':
        model = torch.hub.load('facebookresearch/dinov2', spec['model']).eval().cuda()
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        @torch.no_grad()
        def encode_fn(img):
            x = preprocess(img).unsqueeze(0).cuda()
            emb = model(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().astype('float32').flatten()
        return encode_fn, spec['dim']

    elif spec['lib'] == 'timm':
        import timm
        model = timm.create_model(spec['model'], pretrained=True, num_classes=0).eval().cuda()
        data_cfg = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)

        @torch.no_grad()
        def encode_fn(img):
            x = preprocess(img).unsqueeze(0).cuda()
            emb = model(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().astype('float32').flatten()
        return encode_fn, spec['dim']


def collect_images(img_dir):
    """Images of a nested (concept/image.jpg) or flat (image.jpg) directory."""
    img_dir = Path(img_dir)
    images = []

    subdirs = [d for d in sorted(img_dir.iterdir()) if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            for img_path in sorted(subdir.glob('*.jpg')):
                concept = subdir.name
                img_id = img_path.stem
                images.append({
                    'path': img_path,
                    'id': f'{concept}/{img_id}',
                    'concept': concept,
                })
            for img_path in sorted(subdir.glob('*.png')):
                concept = subdir.name
                img_id = img_path.stem
                images.append({
                    'path': img_path,
                    'id': f'{concept}/{img_id}',
                    'concept': concept,
                })
    else:
        for img_path in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            images.append({
                'path': img_path,
                'id': img_path.stem,
                'concept': img_path.stem,
            })

    return images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True, choices=list(ENCODER_REGISTRY.keys()))
    p.add_argument('--img-dir', default=os.path.join(os.environ.get('THINGS_DIR', 'things'), 'images_THINGS/object_images'),
                   help='Root image directory')
    p.add_argument('--output-dir', default=os.path.join(os.environ.get('THINGS_DIR', 'things'), 'embeddings'),
                   help='Output directory for embeddings')
    p.add_argument('--batch-size', type=int, default=1, help='Batch size (1 for safety)')
    return p.parse_args()


def main():
    args = parse_args()
    spec = ENCODER_REGISTRY[args.encoder]
    print(f"Encoder: {args.encoder} ({spec['model']}, {spec['dim']}d)")

    images = collect_images(args.img_dir)
    print(f"Found {len(images):,} images in {args.img_dir}")

    print(f"Loading {args.encoder}...")
    t0 = time.time()
    encode_fn, dim = load_encoder(args.encoder)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.zeros((len(images), dim), dtype='float32')
    ids = []
    concepts = []
    failed = 0

    for i, img_info in enumerate(tqdm(images, desc=args.encoder)):
        try:
            img = Image.open(img_info['path']).convert('RGB')
            embeddings[i] = encode_fn(img)
            ids.append(img_info['id'])
            concepts.append(img_info['concept'])
        except Exception as e:
            if failed < 3:
                tqdm.write(f"  FAIL [{img_info['id']}]: {e}")
            failed += 1
            ids.append(img_info['id'])
            concepts.append(img_info['concept'])

    out_path = out_dir / f"{args.encoder}.npz"
    np.savez(out_path,
             embeddings=embeddings,
             ids=np.array(ids),
             concepts=np.array(concepts))

    print(f"\nSaved {len(images):,} embeddings ({failed} failed) to {out_path}")
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Norm stats: mean={np.linalg.norm(embeddings, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(embeddings, axis=1).std():.4f}")


if __name__ == "__main__":
    main()
