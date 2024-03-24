import os
import pandas as pd
from pathlib import Path
import shutil
from scipy import stats
from pydub import AudioSegment


our_files = Path('data')
for file in our_files.iterdir():
    print(file.name)

cwd = '/Users/franky/Documents/CS573/project'
group = "cv-valid-test"
data_path = os.path.join(cwd, "data", group)
os.chdir(data_path)
print(os.getcwd())

df = pd.read_csv("/Users/franky/Documents/CS573/project/data/" + group + ".csv")
df = df.drop(columns=['text', 'up_votes', 'down_votes', 'duration'])
df_notna = df.dropna()
df_na = df[df.isna().any(axis=1)]

df_clean = df_notna[df_notna.gender!="other"]

df_clean.age[df_clean.age=="teens"] = '10s'
df_clean.age[df_clean.age=="twenties"] = '20s'
df_clean.age[df_clean.age=="thirties"] = '30s'
df_clean.age[df_clean.age=="fourties"] = '40s'
df_clean.age[df_clean.age=="fifties"] = '50s'
df_clean.age[df_clean.age=="sixties"] = '60s'
df_clean.age[df_clean.age=="seventies"] = '70s'
df_clean.age[df_clean.age=="eighties"] = '80s'




os.makedirs(os.path.join(data_path, "clean"), exist_ok=True)

# os.rmdir(path=os.path.join(data_path, "na"))
# os.listdir()

for file in df_clean.filename[0:20]:
    if os.path.exists(file):
        print(file)
        src = file
        wav_name = Path(file).stem + ".wav"
        dst = os.path.join("clean", wav_name)
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")


for i in df_clean.index:
    new_name = os.path.splitext(df_clean['filename'][i])[0] + ".wav"
    df_clean['filename'][i] = new_name

df_clean.to_csv(os.path.join(data_path, group+".csv"))

print(df_clean)