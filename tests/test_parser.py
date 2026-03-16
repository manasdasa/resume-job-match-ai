from services.resume_parser import extract_resume_text

resume_path = "sample_resume_v1.pdf"

text = extract_resume_text(resume_path)

print("\n====== RESUME TEXT ======\n")
print(text)

