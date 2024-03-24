import numpy as np
import os
import pandas as pd
from scipy import stats
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav


data = np.load("data/cv-valid-train/cv-valid-train-fft/sample-000005.npz")
y = data["fourier"]
x = data["freqArray"]
plt.figure(1,figsize=(20,9))
plt.plot(x, y, color='teal', linewidth=0.5)
plt.xlabel('Frequency (Hz)', fontsize=18)
plt.ylabel('Amplitude (dB)', fontsize=18)
plt.ylim(-20,20)
plt.show()
print(y)
print(np.real(y))