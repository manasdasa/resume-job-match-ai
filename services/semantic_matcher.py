from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# These will be initialized once
job_embeddings = None
jobs_df = None


def initialize_jobs(df):
    """
    Precompute job embeddings once when the server starts
    """
    global jobs_df, job_embeddings

    jobs_df = df

    descriptions = df["Job Description"].fillna("").tolist()

    job_embeddings = model.encode(descriptions)


def rank_jobs(resume_text):

    resume_embedding = model.encode([resume_text])

    scores = cosine_similarity(resume_embedding, job_embeddings)[0]

    ranked_indices = scores.argsort()[::-1][:5]

    results = []

    for idx in ranked_indices:
        results.append(jobs_df.iloc[idx])

    return results