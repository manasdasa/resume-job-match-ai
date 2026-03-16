import pandas as pd

datasets = []

# postings dataset
df1 = pd.read_csv("data/postings.csv")

df1 = df1.rename(columns={
    "title": "job_title",
    "company_name": "company",
    "description": "description"
})

df1 = df1[["job_title", "company", "location", "description"]]

datasets.append(df1)


# monster dataset
df2 = pd.read_csv("data/monster_com-job_sample.csv")

df2 = df2.rename(columns={
    "job_title": "job_title",
    "organization": "company",
    "job_description": "description"
})

df2 = df2[["job_title", "company", "location", "description"]]

datasets.append(df2)

df3 = pd.read_csv("data/glassdoor_jobs.csv")

df3 = df3.rename(columns={
    "Job Title": "job_title",
    "Company Name": "company",
    "Location": "location",
    "Job Description": "description"
})

df3 = df3[["job_title", "company", "location", "description"]]
datasets.append(df3)


# combine everything
all_jobs = pd.concat(datasets)

all_jobs = all_jobs.dropna(subset=["description"])

all_jobs.to_csv("data/all_jobs_dataset.csv", index=False)

print("Dataset size:", len(all_jobs))

