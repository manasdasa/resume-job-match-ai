import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("models/jobs_faiss.index")

# Load dataset
jobs_df = pd.read_csv("data/jobs_small.csv")

# Clean missing values so "nan" never appears
jobs_df["job_title"] = jobs_df["job_title"].fillna("Unknown Position")
jobs_df["company"] = jobs_df["company"].fillna("Unknown Company")
jobs_df["location"] = jobs_df["location"].fillna("Unknown Location")
jobs_df["description"] = jobs_df["description"].fillna("")


def search_matching_jobs(resume_text, top_k=5):

    # Convert resume text to embedding
    embedding = model.encode([resume_text], convert_to_numpy=True)

    embedding = embedding.astype("float32")

    # FAISS search
    distances, indices = index.search(embedding, top_k)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        row = jobs_df.iloc[idx]

        results.append({
            "title": row["job_title"],
            "company": row["company"],
            "location": row["location"],
            "score": round(float(score), 4)
        })

    return results