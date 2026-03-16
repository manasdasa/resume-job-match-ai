import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.resume_parser import extract_resume_text
from services.skill_extractor import extract_skills

resume_text = extract_resume_text("sample_resume_v1.pdf")
skills_list = [
    "python",
    "java",
    "c",
    "excel",
    "machine learning",
    "sql",
]

skills = extract_skills(resume_text, skills_list)

print("\n====== DETECTED SKILLS ======\n")
print(skills)