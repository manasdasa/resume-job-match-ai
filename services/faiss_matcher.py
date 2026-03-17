import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def search_matching_jobs(
    resume_text: str,
    model: SentenceTransformer,
    index: faiss.Index,
    jobs_df: pd.DataFrame,
    top_k: int = 5,
) -> list:
    """
    Encode resume_text with the shared model and run a FAISS nearest-neighbour
    search against the preloaded index.

    Parameters
    ----------
    resume_text : str
        Raw text extracted from the uploaded resume.
    model       : SentenceTransformer
        The single model instance owned by app/main.py.
    index       : faiss.Index
        Pre-built FAISS index loaded at startup.
    jobs_df     : pd.DataFrame
        Jobs dataset loaded at startup (columns: job_title, company, location).
    top_k       : int
        Number of top matches to return.

    Returns
    -------
    list of dicts with keys: title, company, location, score
    """
    embedding = (
        model.encode([resume_text], convert_to_numpy=True)
        .astype("float32")
    )

    distances, indices = index.search(embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        row = jobs_df.iloc[idx]
        results.append({
            "title":    row["job_title"],
            "company":  row["company"],
            "location": row["location"],
            "score":    round(float(score), 4),
        })

    return results
