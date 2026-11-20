"""Shared helpers: config, corpus loading, kNN novelty, encoder loaders."""
import os
import yaml
import numpy as np
import faiss
from pathlib import Path
import os

def load_config(path=None):
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def get_output_dir(cfg, step):
    out = Path(cfg['data']['output_dir']) / step
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_corpus(cfg=None, normalize=False):
    """Corpus embeddings from cfg['corpus']; uses data.wikiart_style when that block is absent."""
    cfg = cfg or load_config()
    if 'corpus' in cfg and 'embeddings' in cfg['corpus']:
        path = cfg['corpus']['embeddings']
        key = cfg['corpus'].get('embeddings_key', 'embeddings')
    else:
        path = cfg['data']['wikiart_style']
        key = 'embeddings'
    emb = np.load(path)[key].astype('float32')
    return emb / np.linalg.norm(emb, axis=1, keepdims=True) if normalize else emb


def load_wikiart(cfg=None, normalize=False):
    return load_corpus(cfg, normalize)


def load_wikiart_data(cfg=None):
    cfg = cfg or load_config()
    data = np.load(cfg['data']['wikiart_style'])
    return {
        'style': data['embeddings'].astype('float32'),
        'ids': data['ids'],
        'img_dir': Path(cfg['data']['wikiart_images']),
    }

def load_observer_corpus(cfg, encoder_name, normalize=False):
    path = cfg['data']['observer_corpus'][encoder_name]
    emb = np.load(path)['embeddings'].astype('float32')
    return emb / np.linalg.norm(emb, axis=1, keepdims=True) if normalize else emb


def get_image_path(data, idx):
    return data['img_dir'] / f"{data['ids'][idx]}.jpg"

def get_reference_image():
    return Path(os.environ['DUROOM_OUT']) / 'inputs' / 'content_photo.png'


def pick_start(corpus, method='p90'):
    centroid = corpus.mean(0)
    dists = np.linalg.norm(corpus - centroid, axis=1)
    if method == 'centroid':
        idx = np.argmin(dists)
    elif method == 'boundary':
        idx = np.argmax(dists)
    elif method.startswith('p'):
        # 'p90': the point whose distance to the centroid is nearest the 90th percentile
        pct = int(method[1:])
        target = np.percentile(dists, pct)
        idx = np.argmin(np.abs(dists - target))
    elif method == 'frontier':
        # point at the 90th-percentile spread of the second-largest cluster
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=2000)
        labels = kmeans.fit_predict(corpus)
        sizes = np.array([np.sum(labels == k) for k in range(10)])
        frontier_k = int(np.argsort(-sizes)[1])
        mask = labels == frontier_k
        members = np.where(mask)[0]
        intra_dists = np.linalg.norm(corpus[mask] - kmeans.cluster_centers_[frontier_k], axis=1)
        p90_intra = np.percentile(intra_dists, 90)
        idx = int(members[np.argmin(np.abs(intra_dists - p90_intra))])
    else:
        raise ValueError(f"Unknown start method: {method}. "
                         "Available: boundary, centroid, p<N>, frontier")
    return idx, corpus[idx].copy()


def build_index(emb, dim=768):
    """FAISS L2 index; an empty emb gives an empty index of width dim."""
    d = emb.shape[1] if len(emb) > 0 else dim
    idx = faiss.IndexFlatL2(d)
    if len(emb) > 0:
        idx.add(emb.astype('float32'))
    return idx


def knn_novelty(query, index, k=15):
    """Mean k-NN L2 distance; inf if the index is empty."""
    if index.ntotal == 0:
        return float('inf')
    k_actual = min(k, index.ntotal)
    D, _ = index.search(query.reshape(1, -1).astype('float32'), k_actual)
    return float(np.mean(np.sqrt(D[0])))  # sqrt because FAISS returns squared L2


def normalize(emb):
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


_encoders = {}
_neg_emb = None  # negative-prompt embedding of the coherence gate
_compute_neg = False   # True only in neg_emb.py: the SigLIP encoder then computes the negative-prompt embedding (needs its tokenizer: network access)

ENCODER_DIMS = {
    'clip-vit-l-14': 768,
    'clip-bigg-14': 1280,
    'dinov3-l': 1024,
    'dinov2-l': 1024,
    'siglip-so400m': 1152,
    'siglip2-so400m': 1152,
    'aimv2-huge': 1536,
}


def _load_clip_encoder():
    import torch
    from generators.scflow import _load_clip  # needs the OpenAI `clip` package (QuickGELU); no open_clip substitute
    model, preprocess = _load_clip()

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_dinov3_encoder():
    import torch
    import timm
    model = timm.create_model('vit_large_patch16_dinov3', pretrained=True, num_classes=0).eval().cuda()
    data_cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_cfg, is_training=False)

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_siglip_encoder():
    global _neg_emb
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP', pretrained='webli')
    model = model.eval().cuda()
    if _compute_neg:
        tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
        neg_text = tokenizer(["noise", "static", "random pixels", "distorted"]).cuda()
        with torch.no_grad():
            neg = model.encode_text(neg_text).mean(dim=0)
            _neg_emb = (neg / neg.norm()).cpu().numpy().astype('float32')

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_clip_bigg_encoder():
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    model = model.eval().cuda()

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_dinov2_l_encoder():
    import torch
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().cuda()
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_siglip2_so400m_encoder():
    global _neg_emb
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP2', pretrained='webli')
    model = model.eval().cuda()
    if _compute_neg:
        tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP2')
        neg_text = tokenizer(["noise", "static", "random pixels", "distorted"]).cuda()
        with torch.no_grad():
            neg = model.encode_text(neg_text).mean(dim=0)
            _neg_emb = (neg / neg.norm()).cpu().numpy().astype('float32')

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


def _load_aimv2_huge_encoder():
    import torch
    import timm
    model = timm.create_model('aimv2_huge_patch14_224', pretrained=True,
                               num_classes=0).eval().cuda()
    data_cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_cfg, is_training=False)

    def encode_fn(img_path):
        from PIL import Image
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = model(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32').flatten()
    return encode_fn


_ENCODER_LOADERS = {
    'clip-vit-l-14': _load_clip_encoder,
    'clip-bigg-14': _load_clip_bigg_encoder,
    'dinov3-l': _load_dinov3_encoder,
    'dinov2-l': _load_dinov2_l_encoder,
    'siglip-so400m': _load_siglip_encoder,
    'siglip2-so400m': _load_siglip2_so400m_encoder,
    'aimv2-huge': _load_aimv2_huge_encoder,
}


def load_encoder(name):
    """Returns (encode_fn, dim); one instance per encoder name."""
    if name not in ENCODER_DIMS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_DIMS.keys())}")
    if name not in _encoders:
        _encoders[name] = _ENCODER_LOADERS[name]()
        print(f"Loaded encoder: {name} (dim={ENCODER_DIMS[name]})")
    return _encoders[name], ENCODER_DIMS[name]


def load_coherence_gate(cfg=None, compute_neg=False):
    """Returns (neg_emb, encode_fn, threshold); neg_emb is read from $DUROOM_OUT/inputs/neg_emb_siglip.npy unless compute_neg (neg_emb.py)."""
    global _neg_emb, _compute_neg
    cfg = cfg or load_config()
    gate_encoder = cfg.get('coherence', {}).get('encoder', 'siglip-so400m')
    if compute_neg:
        _compute_neg = True
    else:
        neg_file = os.environ['DUROOM_OUT'] + '/inputs/neg_emb_siglip.npy'
        if not os.path.exists(neg_file):
            raise SystemExit(f'{neg_file} is missing: run search/neg_emb.py once (needs network access) before any search or render step')
        _neg_emb = np.load(neg_file)
    encode_fn, _ = load_encoder(gate_encoder)
    return _neg_emb, encode_fn, cfg['coherence']['threshold']


def coherence_score(clip_emb, neg_emb):
    """1 - cosine_sim(image, negative_text). Higher = more coherent."""
    return 1.0 - float(clip_emb @ neg_emb)


def load_clip():
    encode_fn, _ = load_encoder('clip-vit-l-14')
    from generators.scflow import _load_clip
    return _load_clip()


def encode_image(img_path):
    encode_fn, _ = load_encoder('clip-vit-l-14')
    return encode_fn(img_path)
