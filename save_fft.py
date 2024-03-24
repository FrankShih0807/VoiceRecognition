import os
import pandas as pd
from pathlib import Path
from scipy import stats
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np


os.chdir("/Users/franky/Documents/CS573/project/data/cv-valid-train")
fft_path = "cv-valid-train-fft"

df = pd.read_csv("cv-valid-train.csv")
print(df)

for file in df["filename"][0:2000]:
    print(file)
    rate, data = wav.read(file)
    fourier = np.fft.fft(data)
    n = len(data)
    fourier = fourier[0:int(n/2)]
    fourier = fourier / float(n)
    freqArray = np.arange(0, n, 1.0) * (rate*1.0/n)

    file_name = Path(file).stem + ".npz"
    dst = os.path.join(fft_path, file_name)
    x = freqArray[freqArray<300] #human voice range
    y = 10*np.log10(fourier)[0:len(x)]
    np.savez(dst, fourier=y, freqArray=x)
