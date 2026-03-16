import pandas as pd

df = pd.read_csv("data/all_jobs_dataset.csv")

# keep only 10k random jobs
df_small = df.sample(10000, random_state=42)

df_small.to_csv("data/jobs_small.csv", index=False)

print("New dataset created")
print("Rows:", len(df_small))
