from services.skill_extractor import extract_skills
from services.skill_gap import find_missing_skills

skills_list = [
    "python",
    "java",
    "sql",
    "aws",
    "docker",
    "machine learning",
    "excel"
]

resume_text = """
Python developer with experience in SQL and Excel.
"""

job_text = """
We are looking for a developer with Python, SQL, AWS and Docker experience.
"""

resume_skills = extract_skills(resume_text, skills_list)
job_skills = extract_skills(job_text, skills_list)

missing = find_missing_skills(resume_skills, job_skills)

print("\nRESUME SKILLS:", resume_skills)
print("\nJOB SKILLS:", job_skills)
print("\nMISSING SKILLS:", missing)