import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import stats
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import spectrogram
from sklearn.neural_network import MLPClassifier
import time

np.seterr(divide = 'ignore') 

cwd = '/Users/franky/Documents/CS573/project'
group = "cv-valid-train"
data_path = os.path.join(cwd, "data", group)
fft_path = "cv-valid-train-window-fft"
os.chdir(data_path)

random_state = 33
df = pd.read_csv( group + ".csv")
# df = df.rename(columns={"Unnamed: 0":"filenum"}).reset_index(drop=True)
# df.to_csv( group + ".csv")
df = df.drop(df.columns[0], axis=1)

n_sample = 20000
df_sample = df.sample(n=n_sample, random_state=random_state).sort_index().reset_index(drop=True)

print(df_sample)
print("\n")

data = np.load('cv-valid-train-window-fft/sample-000013.npz')
f_dim = data["f"].shape[0]

feature_df = np.zeros((df_sample.shape[0], f_dim))
for i, file in enumerate(df_sample["filename"]):
    file = Path(file).stem + ".npz"
    data = np.load(os.path.join(fft_path,file))
    f = data["f"]
    t = data["t"]
    Sxx = data["Sxx"]
    # Sxx_trunc = Sxx[:, Sxx.sum(axis=0)>1]
    Sxx_sum = 10*np.log10(Sxx.sum(axis=1)+1e-10)
    # Sxx_normalized = Sxx_sum/Sxx_sum.sum() 
    Sxx_normalized = Sxx_sum/t.shape[0]
    feature_df[i,:] = Sxx_normalized



# Gender SVM
X = pd.DataFrame(feature_df)
Y = df_sample["gender"]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=random_state)

clf = svm.SVC(kernel='rbf') # Linear Kernel

start_time = time.perf_counter()
clf.fit(X_train, y_train)
end_time = time.perf_counter()

y_pred = clf.predict(X_test)

print("Gender Classification")
print("SVM rbf")
print("Spent time: {}s\n".format(np.round(end_time-start_time, 3)))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(cm/cm.sum())
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, pos_label='male'))
print(precision_score(y_test, y_pred, pos_label='female'))

print(f1_score(y_test, y_pred, pos_label='male'))
print(f1_score(y_test, y_pred, pos_label='female'))






# Gender Ann
X = pd.DataFrame(feature_df)
Y = df_sample["gender"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

clf_ann = MLPClassifier(solver="adam", hidden_layer_sizes=( 128, 64, 32,16))
# clf_ann.fit(X_train, y_train)
start_time = time.perf_counter()
clf_ann.fit(X_train, y_train)
end_time = time.perf_counter()

y_pred = clf_ann.predict(X_test)

print("Gender Classification")
print("Ann")
print("Spent time: {}s\n".format(np.round(end_time-start_time, 3)))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(cm/cm.sum())
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, pos_label='male'))
print(precision_score(y_test, y_pred, pos_label='female'))

print(f1_score(y_test, y_pred, pos_label='male'))
print(f1_score(y_test, y_pred, pos_label='female'))