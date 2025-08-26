"""
movie_search.py
----------------
Semantic search over movie plots using SentenceTransformers (all-MiniLM-L6-v2).

"""
from __future__ import annotations

import os
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Lazy import guard (gives clear error if not installed)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


# -----------------------------
# Data Loading
# -----------------------------
REQUIRED_COLUMNS = {"movie_id", "title", "plot"}


def load_movies(csv_path: str = "movies.csv") -> pd.DataFrame:
    """
    Load the movies dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Defaults to "movies.csv" in the current directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the columns: ["movie_id", "title", "plot"].

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found.
    ValueError
        If required columns are missing.
    """
    csv_path = str('https://github.com/abhinaybhandekar/AI-Systems-Development--IIIT-Naya-Raipur/blob/main/Assignment-1/movies.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Make sure the dataset file is present in the repository root. "
            "Expected columns: movie_id,title,plot"
        )
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    # Ensure types
    df = df.copy()
    df["movie_id"] = df["movie_id"].astype(str)
    df["title"] = df["title"].astype(str)
    df["plot"] = df["plot"].astype(str)
    return df


# -----------------------------
# Search Engine
# -----------------------------
class MovieSearchEngine:
    """
    A small wrapper around SentenceTransformers for semantic search on movie plots.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        movies: Optional[pd.DataFrame] = None,
        csv_path: str = "movies.csv",
        device: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model to use. The assignment specifies all-MiniLM-L6-v2.
        movies : Optional[pd.DataFrame]
            Pre-loaded DataFrame. If None, will load from `csv_path`.
        csv_path : str
            Used if `movies` is None. Also used as the base for the embedding cache path.
        device : Optional[str]
            Device hint passed to SentenceTransformer (e.g., 'cpu' or 'cuda').
        """
        if movies is None:
            movies = load_movies(csv_path)
        self.movies = movies.reset_index(drop=True)

        if SentenceTransformer is None:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Please install requirements first: "
                "pip install -r requirements.txt"
            )

        # Initialize model lazily to keep constructor lightweight if desired.
        self.model = SentenceTransformer(model_name, device=device)

        # Prepare embeddings
        self._csv_path = csv_path
        self._embeddings = self._load_or_build_embeddings()

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def _cache_path(self) -> str:
        base = pathlib.Path(self._csv_path).with_suffix("")
        return f"{base}.embeddings.npz"

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_path = self._cache_path()
        plots = self.movies["plot"].astype(str).tolist()

        # Try cache
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                emb = data["embeddings"]
                if emb.shape[0] == len(plots):
                    return emb
                else:
                    warnings.warn(
                        "Cached embeddings length mismatch with CSV; rebuilding cache."
                    )
            except Exception:
                warnings.warn("Failed to load cached embeddings; rebuilding.")

        # Build embeddings
        emb = self.model.encode(
            plots,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.savez_compressed(cache_path, embeddings=emb)
        return emb

    def search(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Search for the most semantically similar plots to the query.

        Returns a DataFrame with columns: ["movie_id", "title", "plot", "similarity"],
        sorted by descending similarity. If top_n exceeds the dataset size, all rows
        are returned.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("`top_n` must be a positive integer.")

        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]  # shape: (N,)

        df = self.movies.copy()
        df["similarity"] = sims
        df = df.sort_values("similarity", ascending=False).head(top_n)
        # Ensure consistent column order / dtypes
        df = df.loc[:, ["movie_id", "title", "plot", "similarity"]]
        return df.reset_index(drop=True)


# -----------------------------
# Functional API
# -----------------------------
def search_movies(query: str, top_n: int = 5, *, engine: Optional[MovieSearchEngine] = None) -> pd.DataFrame:
    """
    Functional wrapper that creates (or reuses) a MovieSearchEngine and returns results.
    """
    if engine is None:
        engine = MovieSearchEngine()  # loads from default 'movies.csv'
    return engine.search(query, top_n=top_n)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search on movie plots.")
    parser.add_argument("query", type=str, help="Your search query, e.g., 'spy thriller in Paris'")
    parser.add_argument("--top_n", type=int, default=5, help="Number of results to return")
    parser.add_argument("--csv", type=str, default="movies.csv", help="Path to the movies CSV file")
    args = parser.parse_args()

    # Demonstration run
    engine = MovieSearchEngine(csv_path=args.csv)
    results = engine.search(args.query, top_n=args.top_n)

    # Pretty-print a compact table
    with pd.option_context("display.max_colwidth", 80):
        print(results)
