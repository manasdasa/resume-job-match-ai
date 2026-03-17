"""
Run this script ONCE (locally or in CI) before deploying.

It encodes every job description in the dataset and saves the resulting
float32 embedding matrix to data/job_embeddings.npy.

At runtime, app/main.py loads that file and builds the FAISS index in
milliseconds instead of waiting minutes for encoding.

Usage:
    python scripts/precompute_embeddings.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH   = "data/jobs_small.csv"
OUTPUT_PATH = "data/job_embeddings.npy"
MODEL_NAME  = "all-MiniLM-L6-v2"


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["description"] = df["description"].fillna("")
    print(f"  {len(df):,} job records loaded.")

    print(f"Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding job descriptions (this may take a few minutes) ...")
    descriptions = df["description"].tolist()
    embeddings = model.encode(
        descriptions,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    np.save(OUTPUT_PATH, embeddings)

    size_mb = embeddings.nbytes / 1_000_000
    print(f"\nDone.")
    print(f"  Shape  : {embeddings.shape}")
    print(f"  dtype  : {embeddings.dtype}")
    print(f"  Saved  : {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
