def extract_skills(text, skills_list):
    """
    Extract skills from resume text based on a list of known skills.
    """

    text = text.lower()

    found_skills = []

    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills