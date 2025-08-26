"""
movie_search.py
---------------
Implements semantic search on movie plots using SentenceTransformers.

"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once (global)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_movies(query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Perform semantic search over a fixed dataset of movies.

    Parameters
    ----------
    query : str
        Search query (natural language).
    top_n : int
        Number of results to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['title', 'plot', 'similarity'].
    """
    
    df = pd.DataFrame({
        "title": [
            "Spy Movie",
            "Romance in Paris",
            "Action Flick"
        ],
        "plot": [
            "A spy navigates intrigue in Paris to stop a terrorist plot.",
            "A couple falls in love in Paris under romantic circumstances.",
            "A high-octane chase through New York with explosions."
        ]
    })

    # Encode dataset and query
    plot_embeddings = _model.encode(df["plot"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = _model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Compute cosine similarity
    sims = cosine_similarity(query_embedding, plot_embeddings)[0]

    # Attach to DataFrame
    df["similarity"] = sims

    # Sort and return top_n
    result = df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)

    return result
