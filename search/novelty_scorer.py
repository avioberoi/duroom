"""Configurable k-NN novelty scorer with optional Mahalanobis whitenings"""
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

try:
    GPU_AVAILABLE = faiss.get_num_gpus() > 0
except AttributeError:
    GPU_AVAILABLE = False
GPU_RES = faiss.StandardGpuResources() if GPU_AVAILABLE else None


def _to_gpu(index):
    if GPU_AVAILABLE:
        return faiss.index_cpu_to_gpu(GPU_RES, 0, index)
    return index


def compute_whitening(X, rank=64):
    """Low-rank whitening: L such that ||Lx - Ly||^2 = Mahalanobis distance."""
    U, S, _ = np.linalg.svd(np.cov(X.T))
    L = np.diag(1.0 / np.sqrt(S[:rank] + 1e-6)) @ U[:, :rank].T
    return L.astype('float32')


def compute_cluster_whitening(corpus, n_clusters=20, rank=32):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(corpus)
    transforms = []
    for k in range(n_clusters):
        mask = labels == k
        X = corpus[mask] if mask.sum() >= rank + 10 else corpus
        transforms.append(compute_whitening(X, rank))
    return labels, kmeans.cluster_centers_.astype('float32'), transforms


class NoveltyScorer:
    """Metrics: euclidean, cosine, mahalanobis, mahalanobis_local."""

    def __init__(self, corpus, metric='euclidean', k=15, rank=64, n_clusters=20):
        self.metric = metric
        self.k = k
        self.d = corpus.shape[1]
        self.L = None
        self.cluster_transforms = None
        self.cluster_index = None
        self.archive_raw = []

        if metric == 'mahalanobis':
            self.L = compute_whitening(corpus, rank)
            self.index = faiss.IndexFlatL2(self.L.shape[0])
        elif metric == 'mahalanobis_local':
            _, centroids, self.cluster_transforms = compute_cluster_whitening(
                corpus, n_clusters, rank=min(rank, 32))
            self.cluster_index = faiss.IndexFlatL2(self.d)
            self.cluster_index.add(centroids)
            self.index = faiss.IndexFlatL2(self.d)
        elif metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.d)
        else:  # euclidean
            self.index = faiss.IndexFlatL2(self.d)

        if metric != 'mahalanobis_local':
            self.index = _to_gpu(self.index)

    def add(self, x):
        x = np.asarray(x).reshape(1, -1).astype('float32')
        if self.metric == 'mahalanobis':
            self.index.add((self.L @ x.T).T)
        elif self.metric == 'mahalanobis_local':
            self.index.add(x)
            self.archive_raw.append(x.copy())
        elif self.metric == 'cosine':
            self.index.add(x / (np.linalg.norm(x) + 1e-8))
        else:
            self.index.add(x)

    def score(self, x):
        x = np.asarray(x).reshape(1, -1).astype('float32')
        k = min(self.k, self.index.ntotal)
        if k == 0:
            return float('inf')

        if self.metric == 'mahalanobis':
            D, _ = self.index.search((self.L @ x.T).T, k)
            return float(np.mean(np.sqrt(D[0])))
        elif self.metric == 'mahalanobis_local':
            return self._score_local(x, k)
        elif self.metric == 'cosine':
            D, _ = self.index.search(x / (np.linalg.norm(x) + 1e-8), k)
            return float(np.mean(1 - D[0]))
        else:
            D, _ = self.index.search(x, k)
            return float(np.mean(np.sqrt(D[0])))

    def _score_local(self, x, k):
        if not self.archive_raw:
            return float('inf')
        _, I = self.cluster_index.search(x, 1)
        L = self.cluster_transforms[int(I[0, 0])]
        query = (L @ x.T).T
        archive_t = np.vstack([(L @ v.T).T for v in self.archive_raw])
        dists = np.linalg.norm(archive_t - query, axis=1)
        return float(np.mean(np.sort(dists)[:min(self.k, len(dists))]))
