from services.similarity import calculate_match_score

resume = """
Python developer with experience in machine learning, SQL, and data analysis.
"""

job = """
We are looking for a machine learning engineer with Python and SQL experience.
"""

score = calculate_match_score(resume, job)

print("\nMATCH SCORE:", score)