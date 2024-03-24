import os
import pandas as pd
from pathlib import Path

cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
our_files = Path(data_path)

for file in our_files.iterdir():
    print(file)

df = pd.read_csv("/Users/franky/Documents/CS573/project/data/cv-valid-train.csv")
df_gender = df[df["gender"].notna()]
print(df_gender.shape)