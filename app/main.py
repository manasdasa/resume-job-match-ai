"""
FastAPI application entry point.

All heavyweight resources (model, embeddings, FAISS index, DataFrame) are
loaded ONCE inside the lifespan context manager before the server starts
accepting requests.  Every route handler receives these resources through
module-level references — no loading, no I/O, no model inference happens
outside of per-request work.

Startup sequence
----------------
1. Load SentenceTransformer model (~90 MB, downloaded once then cached)
2. Load precomputed embeddings from data/job_embeddings.npy
3. Build in-memory FAISS index from the embeddings  (< 1 ms for 10k vectors)
4. Load jobs DataFrame from data/jobs_small.csv

Per-request work in /analyze
-----------------------------
- PDF → text extraction          (pdfplumber,  ~50–200 ms)
- Resume embedding (1 vector)    (model.encode, ~20–50 ms)
- FAISS nearest-neighbour search (< 1 ms)
- Match score (2-vector batch)   (model.encode, ~20–50 ms)
- Skill extraction + advice      (pure string ops, < 1 ms)
"""

from contextlib import asynccontextmanager

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services.faiss_matcher import search_matching_jobs
from services.resume_advice import generate_resume_advice
from services.resume_parser import extract_resume_text
from services.similarity import calculate_match_score
from services.skill_extractor import extract_skills
from services.skill_gap import find_missing_skills


# ── Constants ─────────────────────────────────────────────────────────────────

SKILLS_LIST = [
    "python",
    "java",
    "sql",
    "aws",
    "docker",
    "machine learning",
    "excel",
]

MODEL_NAME        = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH   = "data/job_embeddings.npy"
JOBS_CSV_PATH     = "data/jobs_small.csv"


# ── Shared state (populated once at startup, read-only during requests) ───────

model:       SentenceTransformer = None
faiss_index: faiss.Index         = None
jobs_df:     pd.DataFrame        = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything inside this block runs before the server accepts its first
    request.  The `yield` separates startup from shutdown logic.
    """
    global model, faiss_index, jobs_df

    print(f"[startup] Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[startup] Loading precomputed embeddings from '{EMBEDDINGS_PATH}' ...")
    embeddings = np.load(EMBEDDINGS_PATH)
    dimension  = embeddings.shape[1]

    print(f"[startup] Building FAISS index ({len(embeddings):,} vectors, dim={dimension}) ...")
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print(f"[startup] Loading jobs dataset from '{JOBS_CSV_PATH}' ...")
    jobs_df = pd.read_csv(JOBS_CSV_PATH)
    jobs_df["job_title"]   = jobs_df["job_title"].fillna("Unknown Position")
    jobs_df["company"]     = jobs_df["company"].fillna("Unknown Company")
    jobs_df["location"]    = jobs_df["location"].fillna("Unknown Location")
    jobs_df["description"] = jobs_df["description"].fillna("")

    print(f"[startup] Ready — {len(jobs_df):,} jobs indexed.")

    yield  # ← server is live from here until shutdown


# ── App ───────────────────────────────────────────────────────────────────────

app       = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    # ── 1. Extract text from uploaded PDF ─────────────────────────────────────
    resume_text = extract_resume_text(resume.file)

    # ── 2. Skill extraction + gap analysis ────────────────────────────────────
    resume_skills = extract_skills(resume_text, SKILLS_LIST)
    job_skills    = extract_skills(job_description, SKILLS_LIST)
    missing       = find_missing_skills(resume_skills, job_skills)
    suggestions   = generate_resume_advice(missing)

    # ── 3. Semantic job matching via FAISS (uses shared model + index) ─────────
    recommended_jobs = search_matching_jobs(
        resume_text, model, faiss_index, jobs_df, top_k=5
    )

    # ── 4. Resume ↔ job-description cosine similarity score ───────────────────
    score = calculate_match_score(resume_text, job_description, model)

    return templates.TemplateResponse(
        "results.html",
        {
            "request":       request,
            "match_score":   score,
            "resume_skills": resume_skills,
            "missing_skills": missing,
            "suggestions":   suggestions,
            "jobs":          recommended_jobs,
        },
    )
