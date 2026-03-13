def generate_resume_advice(missing_skills):

    advice = []

    for skill in missing_skills:

        if skill == "sql":
            advice.append("Add experience using SQL for querying and analyzing data.")

        elif skill == "aws":
            advice.append("Mention any AWS services you have used such as S3, EC2, or Lambda.")

        elif skill == "docker":
            advice.append("Include experience with Docker or containerized applications if applicable.")

        elif skill == "machine learning":
            advice.append("Highlight any machine learning projects, coursework, or models you have built.")

        elif skill == "excel":
            advice.append("Add examples of data analysis or dashboards built using Excel.")

        else:
            advice.append(f"Consider adding experience with {skill} if you have worked with it.")

    return advice