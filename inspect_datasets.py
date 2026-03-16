import pandas as pd
import os

folder = "data"

for file in os.listdir(folder):

    if file.endswith(".csv"):

        path = os.path.join(folder, file)

        print("\n==========================")
        print("FILE:", file)

        df = pd.read_csv(path, nrows=5)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nSample rows:")
        print(df.head())