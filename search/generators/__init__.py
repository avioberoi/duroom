"""Style-to-image generators with unified interface"""

BACKENDS = {}
DEFAULT_BACKEND = 'variations'


def _lazy_variations():
    from .variations import generate as gen
    BACKENDS['variations'] = gen
    return gen


_LOADERS = {'variations': _lazy_variations}


def generate(content_img, style_emb, output_path, backend=None):
    backend = backend or DEFAULT_BACKEND
    if backend not in BACKENDS:
        if backend in _LOADERS:
            _LOADERS[backend]()
        else:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(_LOADERS.keys())}")
    return BACKENDS[backend](content_img, style_emb, output_path)


def set_default_backend(backend):
    global DEFAULT_BACKEND
    if backend not in _LOADERS:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(_LOADERS.keys())}")
    if backend not in BACKENDS:
        _LOADERS[backend]()
    DEFAULT_BACKEND = backend
