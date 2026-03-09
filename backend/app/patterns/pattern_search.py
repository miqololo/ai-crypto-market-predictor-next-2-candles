"""Candle pattern extraction and similarity search using FAISS + fastdtw."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def extract_patterns(
    df: pd.DataFrame,
    pattern_len: int = 20,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Extract sliding-window candle patterns from OHLCV.
    Returns (patterns array, timestamps).
    """
    if features is None:
        features = ["open", "high", "low", "close", "volume"]
    available = [c for c in features if c in df.columns]
    data = df[available].values.astype(np.float32)
    # Normalize per column
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    patterns = []
    timestamps = []
    for i in range(len(data) - pattern_len):
        patterns.append(data[i : i + pattern_len].flatten())
        timestamps.append(df.index[i + pattern_len - 1])
    return np.array(patterns, dtype=np.float32), timestamps


class PatternSearch:
    """FAISS index + DTW for candle pattern similarity search."""

    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss and HAS_FAISS
        self.index = None
        self.patterns: Optional[np.ndarray] = None
        self.timestamps: List[pd.Timestamp] = []
        self.metadata: List[dict] = []
        self.documents: List[str] = []

    def build(
        self,
        patterns: np.ndarray,
        timestamps: Optional[List] = None,
        metadata: Optional[List[dict]] = None,
        documents: Optional[List[str]] = None,
    ):
        """Build FAISS index from patterns."""
        self.patterns = patterns.astype(np.float32)
        self.timestamps = timestamps or []
        self.metadata = metadata or [{}] * len(patterns)
        self.documents = documents or []
        if self.use_faiss:
            dim = patterns.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(patterns)
        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        use_dtw: bool = False,
        n_features: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest patterns.
        use_dtw=True: slower but uses DTW distance (better for time series).
        """
        if use_dtw and self.patterns is not None:
            return self._search_dtw(query, k, n_features)
        if self.use_faiss and self.index is not None:
            _, indices = self.index.search(
                query.reshape(1, -1).astype(np.float32), k
            )
            # FAISS returns L2 distances
            distances = []
            for idx in indices[0]:
                if 0 <= idx < len(self.patterns):
                    d = np.linalg.norm(query - self.patterns[idx])
                    distances.append((int(idx), float(d)))
            return distances
        # Fallback: brute force
        if self.patterns is None:
            return []
        dists = np.linalg.norm(self.patterns - query, axis=1)
        top_k = np.argsort(dists)[:k]
        return [(int(i), float(dists[i])) for i in top_k]

    def _search_dtw(
        self, query: np.ndarray, k: int, n_features: int = 5
    ) -> List[Tuple[int, float]]:
        """DTW-based search (slower, more accurate for time series)."""
        if self.patterns is None:
            return []
        q = query.flatten()
        pattern_len = len(q) // n_features
        q_2d = q.reshape(pattern_len, n_features)
        results = []
        for i, p in enumerate(self.patterns):
            p_2d = p.reshape(pattern_len, n_features)
            dist, _ = fastdtw(q_2d, p_2d, dist=euclidean)
            results.append((i, dist))
        results.sort(key=lambda x: x[1])
        return results[:k]

    def save(self, path: str):
        """Save index, patterns, and timestamps."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.patterns is not None:
            np.save(path / "patterns.npy", self.patterns)
        if self.timestamps:
            ts_str = [str(t) for t in self.timestamps]
            np.save(path / "timestamps.npy", np.array(ts_str, dtype=object), allow_pickle=True)
        if self.documents:
            np.save(path / "documents.npy", np.array(self.documents, dtype=object), allow_pickle=True)
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

    def load(self, path: str):
        """Load index, patterns, and timestamps."""
        path = Path(path)
        patterns_path = path / "patterns.npy"
        if patterns_path.exists():
            self.patterns = np.load(patterns_path)
            self.dim = self.patterns.shape[1]
        ts_path = path / "timestamps.npy"
        if ts_path.exists():
            ts_arr = np.load(ts_path, allow_pickle=True)
            self.timestamps = list(ts_arr) if ts_arr.size else []
        doc_path = path / "documents.npy"
        if doc_path.exists():
            doc_arr = np.load(doc_path, allow_pickle=True)
            self.documents = list(doc_arr) if doc_arr.size else []
        index_path = path / "index.faiss"
        if index_path.exists() and self.use_faiss:
            self.index = faiss.read_index(str(index_path))
        return self
