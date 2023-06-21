import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
import librosa.display

#possible file names
# - twistzz1v3AK.wav
# - brokySprayDownBananaInferno.wav
# - twistzzJump.wav
# - torziNuke1v3.wav

clip = "twistzz1v3AK.wav"
sound = "Videos/" + clip
print(os.listdir())

scale, sr = librosa.load(sound)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

Y_scale = np.abs(S_scale) ** 2


def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.title(clip[:-4])
    plt.show()

Y_log_scale = librosa.power_to_db(Y_scale)

plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")


weights = np.ones(np.shape(Y_log_scale)[0])
weights[  int(np.shape(Y_log_scale)[0]*0.02) : int(np.shape(Y_log_scale)[0]*0.3) ] = 3*np.ones(int(np.shape(Y_log_scale)[0]*0.3)-int(np.shape(Y_log_scale)[0]*0.02))

#
soundsArr = []
for count, i in enumerate(Y_log_scale.T):
    tmp = weights * i
    soundsArr.append(tmp)
    
soundsSumArr = []
for arr in soundsArr:
    tmpSum = 0
    for value in arr:
        if value > 0:
            tmpSum = tmpSum + value
    soundsSumArr.append(tmpSum)
soundsSumArr = np.array(soundsSumArr)

soundsSumArr = np.array(soundsSumArr)
maxCount = np.max(soundsSumArr)

normalizedCountArr = []
for i in range(len(soundsSumArr)):
    normalizedCountArr.append(float(soundsSumArr[i])/float(maxCount))
    
plt.title("Time vs Sum") 
plt.xlabel("Time (not unit of time)")
plt.ylabel("Sum positive frequencies") 
plt.plot(normalizedCountArr) 
plt.show()