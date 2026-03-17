import numpy as np
from sentence_transformers import SentenceTransformer


def calculate_match_score(
    resume_text: str,
    job_text: str,
    model: SentenceTransformer,
) -> float:
    """
    Compute a cosine-similarity score between resume and job description.

    Both texts are encoded in a single batched model.encode() call, which
    is faster than two separate encode() calls.

    Parameters
    ----------
    resume_text : str
        Raw text extracted from the uploaded resume.
    job_text    : str
        Job description text pasted by the user.
    model       : SentenceTransformer
        The single model instance owned by app/main.py.

    Returns
    -------
    float  Score in the range 0–100 (higher = better match).
    """
    embeddings = model.encode(
        [resume_text, job_text],
        convert_to_numpy=True,
    )
    resume_emb, job_emb = embeddings[0], embeddings[1]

    norm = np.linalg.norm(resume_emb) * np.linalg.norm(job_emb)
    similarity = float(np.dot(resume_emb, job_emb) / norm) if norm > 0 else 0.0

    return round(similarity * 100, 2)
