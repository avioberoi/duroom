"""CLIP ViT-L/14 loader of the OpenAI `clip` package (QuickGELU weights), used by the encoders."""
import clip

_clip = None


def _load_clip():
    global _clip
    if _clip is None:
        _clip = clip.load('ViT-L/14', device='cuda')
        _clip[0].eval()
        print("CLIP loaded")
    return _clip
