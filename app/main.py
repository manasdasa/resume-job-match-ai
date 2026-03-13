from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import FastAPI, UploadFile, File, Form
from services.resume_parser import extract_resume_text
from services.skill_extractor import extract_skills
from services.skill_gap import find_missing_skills
from services.similarity import calculate_match_score

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


skills_list = [
    "python",
    "java",
    "sql",
    "aws",
    "docker",
    "machine learning",
    "excel"
]


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    resume_text = extract_resume_text(resume.file)

    resume_skills = extract_skills(resume_text, skills_list)
    job_skills = extract_skills(job_description, skills_list)

    missing = find_missing_skills(resume_skills, job_skills)

    score = round(float(calculate_match_score(resume_text, job_description)), 2)

    return {
        "match_score": score,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "missing_skills": missing
    }