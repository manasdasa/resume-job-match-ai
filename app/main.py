from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import FastAPI, UploadFile, File, Form
from services.resume_parser import extract_resume_text
from services.skill_extractor import extract_skills
from services.skill_gap import find_missing_skills
from services.similarity import calculate_match_score
from services.resume_advice import generate_resume_advice

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
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    resume_text = extract_resume_text(resume.file)

    resume_skills = extract_skills(resume_text, skills_list)
    job_skills = extract_skills(job_description, skills_list)

    missing = find_missing_skills(resume_skills, job_skills)
    suggestions = generate_resume_advice(missing)

    score = round(float(calculate_match_score(resume_text, job_description)), 2)

    return templates.TemplateResponse(
    "results.html",
    {
        "request": request,
        "match_score": score,
        "resume_skills": resume_skills,
        "missing_skills": missing,
        "suggestions": suggestions
    }
)