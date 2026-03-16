from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, FastAPI, UploadFile, File, Form
from services.faiss_matcher import search_matching_jobs
from services.resume_parser import extract_resume_text
from services.skill_extractor import extract_skills
from services.skill_gap import find_missing_skills
from services.similarity import calculate_match_score
from services.resume_advice import generate_resume_advice
from services.semantic_matcher import initialize_jobs


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

    # extract resume text
    resume_text = extract_resume_text(resume.file)

    # extract skills
    resume_skills = extract_skills(resume_text, skills_list)
    job_skills = extract_skills(job_description, skills_list)

    # find missing skills
    missing = find_missing_skills(resume_skills, job_skills)

    # generate suggestions
    suggestions = generate_resume_advice(missing)

    # semantic job matching
    recommended_jobs = search_matching_jobs(resume_text, top_k=5)
    


    # match score
    score = round(float(calculate_match_score(resume_text, job_description)), 2)



    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "match_score": score,
            "resume_skills": resume_skills,
            "missing_skills": missing,
            "suggestions": suggestions,
            "jobs": recommended_jobs
        }
    )

