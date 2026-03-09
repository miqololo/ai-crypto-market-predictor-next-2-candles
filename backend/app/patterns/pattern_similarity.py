"""
PatternSimilaritySearch: find historical moments most similar to a query
using a user-chosen subset of features. Supports cosine similarity and Euclidean distance.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MetricType = Literal["cosine", "euclidean"]
MissingStrategy = Literal["skip", "impute_zero", "impute_mean"]


class PatternSimilaritySearch:
    """
    Search for the most similar historical data points (or short windows)
    using a user-chosen subset of features at query time.
    """

    def __init__(
        self,
        missing_strategy: MissingStrategy = "skip",
    ):
        """
        Args:
            missing_strategy: How to handle NaN in selected features.
                - "skip": drop rows with any NaN in selected features (default)
                - "impute_zero": replace NaN with 0
                - "impute_mean": replace NaN with column mean
        """
        self.missing_strategy = missing_strategy
        self._df: Optional[pd.DataFrame] = None
        self._feature_means: Optional[Dict[str, float]] = {}

    def fit(
        self,
        data: Union[str, Path, pd.DataFrame],
        timestamp_col: Optional[str] = None,
    ) -> "PatternSimilaritySearch":
        """
        Load and prepare data for similarity search.

        Args:
            data: Path to JSON/CSV file or a pandas DataFrame.
                 JSON: supports records, split, index, columns orient.
            timestamp_col: Column name for timestamps. If None, uses index when
                 it's datetime-like, else creates a range index.

        Returns:
            self for method chaining
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix.lower() == ".json":
                self._df = pd.read_json(path, orient="records")
            elif path.suffix.lower() == ".csv":
                self._df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif isinstance(data, pd.DataFrame):
            self._df = data.copy()
        else:
            raise TypeError("data must be str, Path, or pd.DataFrame")

        # Ensure timestamp column
        self._df = self._df.copy()
        if timestamp_col and timestamp_col in self._df.columns:
            self._df["_timestamp"] = pd.to_datetime(self._df[timestamp_col])
        elif "timestamp" in self._df.columns:
            self._df["_timestamp"] = pd.to_datetime(self._df["timestamp"])
        elif isinstance(self._df.index, pd.DatetimeIndex):
            self._df["_timestamp"] = self._df.index
        else:
            self._df["_timestamp"] = pd.RangeIndex(len(self._df))

        self._feature_means = {}
        return self

    def _prepare_features(
        self,
        selected_features: List[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract selected features and apply missing strategy.
        Returns (X array, valid_indices) where valid_indices maps rows in X to self._df.
        """
        available = [c for c in selected_features if c in self._df.columns]
        if not available:
            raise ValueError(
                f"None of the selected features {selected_features} exist in data. "
                f"Available columns: {list(self._df.columns)}"
            )
        if len(available) < len(selected_features):
            missing = set(selected_features) - set(available)
            raise ValueError(
                f"Features not found in data: {missing}. "
                f"Available: {list(self._df.columns)}"
            )

        sub = self._df[available].copy()

        if self.missing_strategy == "skip":
            valid = ~sub.isna().any(axis=1)
            valid_indices = np.where(valid.values)[0]
            sub = sub.loc[valid].fillna(0)
        elif self.missing_strategy == "impute_zero":
            sub = sub.fillna(0)
            valid_indices = np.arange(len(sub))
        else:  # impute_mean
            for col in available:
                mean_val = sub[col].mean()
                self._feature_means[col] = mean_val
                sub[col] = sub[col].fillna(mean_val)
            valid_indices = np.arange(len(sub))

        X = sub[selected_features].values.astype(np.float64)
        return X, valid_indices

    def query(
        self,
        current_values_dict: Dict[str, float],
        selected_features_list: List[str],
        metric: MetricType = "cosine",
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Find top-k historical moments most similar to the query.

        Args:
            current_values_dict: Dict of feature name -> value for the query point.
            selected_features_list: Features to use for similarity (must match keys in dict).
            metric: "cosine" (default) or "euclidean".
            k: Number of results to return.

        Returns:
            DataFrame with columns: timestamp, similarity_score, matched_row_data.
            For cosine: similarity_score in [-1, 1] (higher = more similar).
            For euclidean: similarity_score is negative distance (higher = closer).
        """
        if self._df is None:
            raise RuntimeError("Call fit() before query()")

        # Build query vector in same order as selected features
        query_vec = np.array(
            [current_values_dict.get(f, np.nan) for f in selected_features_list],
            dtype=np.float64,
        )
        if np.any(np.isnan(query_vec)):
            missing = [f for f in selected_features_list if np.isnan(current_values_dict.get(f, np.nan))]
            raise ValueError(f"Query missing values for features: {missing}")

        X, valid_indices = self._prepare_features(selected_features_list)
        if len(X) == 0:
            return pd.DataFrame(columns=["timestamp", "similarity_score", "matched_row_data"])

        if HAS_TORCH:
            scores, indices = self._search_torch(query_vec, X, metric, k)
        else:
            scores, indices = self._search_numpy(query_vec, X, metric, k)

        # Map search indices back to DataFrame rows
        key_cols = [
            c for c in ["open", "high", "low", "close", "volume", "rsi", "macd_hist",
                        "long_short_ratio", "taker_ratio", "funding_rate", "normalize_long_short_ratio",
                        "normalize_ema_9", "normalize_macd_hist", "supertrend_dir"]
            if c in self._df.columns
        ]
        if not key_cols:
            key_cols = [c for c in self._df.columns if c != "_timestamp"]

        result_rows = []
        for i, idx in enumerate(indices):
            orig_idx = valid_indices[idx]
            row = self._df.iloc[orig_idx]
            ts = row["_timestamp"] if "_timestamp" in row.index else self._df.index[orig_idx]
            matched = {c: row[c] for c in key_cols if c in row.index}
            result_rows.append({
                "timestamp": ts,
                "similarity_score": float(scores[i]),
                "matched_row_data": matched,
            })

        return pd.DataFrame(result_rows)

    def _search_torch(
        self,
        query: np.ndarray,
        X: np.ndarray,
        metric: MetricType,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """PyTorch-based search."""
        q = torch.from_numpy(query).float().unsqueeze(0)
        Xt = torch.from_numpy(X).float()

        if metric == "cosine":
            # Cosine similarity: higher is more similar
            q_norm = q / (q.norm(dim=1, keepdim=True) + 1e-10)
            X_norm = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-10)
            sim = (q_norm @ X_norm.T).squeeze(0)
            scores, indices = torch.topk(sim, min(k, len(sim)))
        else:  # euclidean
            # Euclidean distance: lower is more similar
            dist = torch.cdist(q, Xt).squeeze(0)
            scores, indices = torch.topk(-dist, min(k, len(dist)))
            scores = -scores  # convert back to negative distance (higher = closer)

        return scores.numpy(), indices.numpy()

    def _search_numpy(
        self,
        query: np.ndarray,
        X: np.ndarray,
        metric: MetricType,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy-based search (fallback when PyTorch not available)."""
        k = min(k, len(X))

        if metric == "cosine":
            q_norm = query / (np.linalg.norm(query) + 1e-10)
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            sim = X_norm @ q_norm
            indices = np.argsort(-sim)[:k]
            scores = sim[indices]
        else:  # euclidean
            dist = np.linalg.norm(X - query, axis=1)
            indices = np.argsort(dist)[:k]
            scores = -dist[indices]  # negative distance: higher = closer

        return scores, indices

    def query_window(
        self,
        current_window_df: pd.DataFrame,
        selected_features_list: List[str],
        metric: MetricType = "cosine",
        k: int = 10,
        aggregate: Literal["flatten", "mean"] = "mean",
    ) -> pd.DataFrame:
        """
        Find similar short sequences (e.g. last N candles) using selected features.

        Args:
            current_window_df: DataFrame with last N rows (e.g. last 8 candles).
            selected_features_list: Features to use.
            metric: "cosine" or "euclidean".
            k: Number of results.
            aggregate: "flatten" = concatenate all values per feature into one vector,
                       "mean" = average each feature over the window.

        Returns:
            Same format as query().
        """
        if self._df is None:
            raise RuntimeError("Call fit() before query_window()")

        available = [c for c in selected_features_list if c in current_window_df.columns]
        if len(available) < len(selected_features_list):
            raise ValueError(
                f"Features {set(selected_features_list) - set(available)} not in window DataFrame"
            )

        if aggregate == "mean":
            query_vec = current_window_df[selected_features_list].mean(axis=0).values.astype(np.float64)
        else:  # flatten
            query_vec = current_window_df[selected_features_list].values.flatten().astype(np.float64)

        if np.any(np.isnan(query_vec)):
            raise ValueError("Query window contains NaN in selected features")

        window_len = len(current_window_df)
        if window_len > len(self._df):
            return pd.DataFrame(columns=["timestamp", "similarity_score", "matched_row_data"])

        # Prepare data: use impute for windows so we have consecutive rows
        sub = self._df[selected_features_list].copy()
        if self.missing_strategy == "skip":
            sub = sub.fillna(0)  # for windows, impute to preserve sequence
        elif self.missing_strategy == "impute_zero":
            sub = sub.fillna(0)
        else:
            for col in selected_features_list:
                sub[col] = sub[col].fillna(sub[col].mean())

        n = len(sub)
        if n < window_len:
            return pd.DataFrame(columns=["timestamp", "similarity_score", "matched_row_data"])

        # Build sliding windows
        X_list = []
        row_indices = []
        for i in range(n - window_len + 1):
            window = sub.iloc[i : i + window_len]
            if aggregate == "mean":
                X_list.append(window.mean(axis=0).values)
            else:
                X_list.append(window.values.flatten())
            row_indices.append(i + window_len - 1)

        X = np.array(X_list, dtype=np.float64)
        if len(X) == 0:
            return pd.DataFrame(columns=["timestamp", "similarity_score", "matched_row_data"])

        if query_vec.shape[0] != X.shape[1]:
            raise ValueError(
                f"Query dimension {query_vec.shape[0]} != database window dimension {X.shape[1]}"
            )

        if HAS_TORCH:
            scores, idx_in_X = self._search_torch(query_vec, X, metric, min(k, len(X)))
        else:
            scores, idx_in_X = self._search_numpy(query_vec, X, metric, min(k, len(X)))

        key_cols = [
            c for c in ["open", "high", "low", "close", "volume", "rsi", "macd_hist",
                        "long_short_ratio", "taker_ratio", "funding_rate"]
            if c in self._df.columns
        ]
        if not key_cols:
            key_cols = [c for c in self._df.columns if c != "_timestamp"]

        result_rows = []
        for i, idx in enumerate(idx_in_X):
            orig_idx = row_indices[idx]
            row = self._df.iloc[orig_idx]
            ts = row["_timestamp"] if "_timestamp" in row.index else self._df.index[orig_idx]
            matched = {c: row[c] for c in key_cols if c in row.index}
            result_rows.append({
                "timestamp": ts,
                "similarity_score": float(scores[i]),
                "matched_row_data": matched,
            })

        return pd.DataFrame(result_rows)
