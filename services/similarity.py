from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import get_embedding


def calculate_match_score(resume_text, job_text):

    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_text)

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    score = round(similarity * 100, 2)

    return score