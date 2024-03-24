import os
import pandas as pd
from pathlib import Path
import shutil
from scipy import stats
from pydub import AudioSegment

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fft import fftshift
import numpy as np

from scipy.signal import spectrogram

cwd = '/Users/franky/Documents/CS573/project'
group = "cv-valid-train"
data_path = os.path.join(cwd, "data", group)
fft_path = "cv-valid-train-window-fft"
os.chdir(data_path)
print(os.getcwd())

df = pd.read_csv( group + ".csv")
df = df.drop(df.columns[0], axis=1)
print(df)

n_sample = 20000
df_sample = df.sample(n=n_sample, random_state=33).sort_index().reset_index(drop=True)
print(df_sample)

nperseg = 4800
# kernel_length = 7
# kernel = np.ones(kernel_length)/kernel_length
for i, file  in enumerate(df_sample["filename"]):
    rate, data = wav.read(file)
    # data_avg = np.convolve(data, kernel, mode='same')
    # data_trunc = data[np.abs(data_avg)>20]
    # time = np.arange(0, float(data_trunc.shape[0]), 1) / rate

    # if data.shape[0]>=10*nperseg:
    f, t, Sxx = spectrogram(x=data, fs=rate, nperseg=nperseg, window="hamming", noverlap= int(nperseg/2))
    f_trunc = f[f<=1000]
    Sxx_trunc = Sxx[f<=1000]

    file_name = Path(file).stem + ".npz"
    dst = os.path.join(fft_path, file_name)
    np.savez(dst, f = f_trunc, t = t, Sxx = Sxx_trunc)
    # else:
    #     print(file, "is too short!")

    if i % 100 ==0:
        print("finish {} files".format(i))

