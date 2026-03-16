import pandas as pd

# Load dataset
jobs_df = pd.read_csv("data/glassdoor_jobs.csv")

def find_matching_jobs(resume_skills, top_n=5):

    results = []

    for _, row in jobs_df.iterrows():

        description = str(row["Job Description"]).lower()

        score = 0

        for skill in resume_skills:
            if skill.lower() in description:
                score += 1

        if score > 0:
            results.append({
                "title": row["Job Title"],
                "company": row["Company Name"],
                "location": row["Location"],
                "score": score
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_n]