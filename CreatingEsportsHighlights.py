# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:26:33 2022

@author: 857238
"""


### ---Imports--- ###
# - Spectrogram
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
import librosa.display

# - Splitting files
from pydub import AudioSegment
from pydub.utils import which
import difflib
### ---Imports--- ###

from timeit import default_timer as timer
start = timer()
#possible file names
# - twistzz1v3AK.wav
# - brokySprayDownBananaInferno.wav
# - twistzzJump.wav
# - torziNuke1v3.wav

clip = "cologneFinalsMap2.wav"
sound = "Videos/" + clip
print(os.listdir())
AudioSegment.converter = which("ffmpeg")

audio = AudioSegment.from_file(sound, "mp4") #do not change

scale, sr = librosa.load(sound)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

Y_scale = np.abs(S_scale) ** 2

Y_log_scale = librosa.power_to_db(Y_scale)

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

normalizedSumArr = []
for i in range(len(soundsSumArr)):
    normalizedSumArr.append(float(soundsSumArr[i])/float(maxCount))
    
plt.title("Time vs Sum") 
plt.xlabel("Time (not unit of time)")
plt.ylabel("Sum positive frequencies") 
plt.plot(normalizedSumArr) 
plt.show()

beg_time = 0
end_time = 30

timeArr = []
successfulTimeArr = []  # list of successful time stamps
countArr = []           # list of the counts
successfulCountArr = [] # list of successfult time stamps w/ 

for i in range(len(normalizedSumArr)-1):
    tmpArr = normalizedSumArr[beg_time:end_time]
    tmp_txt = str(beg_time) + " - " + str(end_time)
    '''
    for j in tmpArr:
        if j > 6000:
            successfulTimeArr.append(tmp_txt)
            beg_time = beg_time + 300
            end_time = end_time + 300
            break;
    '''
    
    count = 0
    for j in tmpArr:
        if j == 1.0: # only looks at top 10% of clips
            print("equal!")
            count = count + 1
    countArr.append(count)
    tmpstr = str(beg_time) + "(ms) - " + str(end_time) + "(ms)"
    timeArr.append(tmpstr)
    if count >= 1:
        
        successfulTimeArr.append(tmpstr)
        theStr = tmpstr + " >> " + str(count)
        successfulCountArr.append(theStr)
    
    beg_time = beg_time + 1 # ---> increase by 1 interval of 23 milliseconds
    end_time = end_time + 1
    
endArrTimeStampMS = []
for value in successfulCountArr:
    tmparr = value.split()
    tmpStr2 = tmparr[2]
    tmpStr2 = tmpStr2[:-4]
    endArrTimeStampMS.append(tmpStr2)
    #converting from time to min
    tmpStr2Min = int(tmpStr2) * 23 / (1000 * 60)

finalTimeStampMS = []
for index in range(len(endArrTimeStampMS)-1):
    if np.abs(int(endArrTimeStampMS[index]) - int(endArrTimeStampMS[index+1])) < 20:
        if index + 2 == len(endArrTimeStampMS):
            finalTimeStampMS.append(endArrTimeStampMS[index])
        pass;
    else:
        finalTimeStampMS.append(endArrTimeStampMS[index])
        
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
for time in finalTimeStampMS:
    ### ---Creates Clips--- ###
    t1 = int(time) * 23 / (1000) - 40 #In seconds
    
    t2 = int(time) * 23 / (1000) + 40 #In seconds
    newAudio = audio[t1:t2]
    txt = str(t1 / 60) + " - " + str(t2/(60)) + " - " + clip[:-4]
    ffmpeg_extract_subclip("Videos/" + clip, t1, t2, targetname= "Videos/subClips/" + txt + ".mp4")
    ffmpeg_extract_subclip("Videos/" + clip, t1, t2, targetname= "Videos/subClips/wavSubClips/" + txt + ".wav")
#end of finding clips


#Start of filtering
warningFiles = []
totalCounts = []
weightedFrequencyArr = []
finalClipNamesArr = []

for file in os.listdir("Videos/subClips/wavSubClips/"): #analyzing the 2 min clips and making filters
    sound = "Videos/subClips/wavSubClips/" + file
    
    AudioSegment.converter = which("ffmpeg")

    audio = AudioSegment.from_file(sound, "mp4") #do not change
    
    time = 0
    lengthOfClip = 30 # in seconds
    timeStampArr = []
    successfulArr = []
    successfulTimeArr = []
    while time <= int(len(audio)/1000)-lengthOfClip:
        ### ---Creates Clips--- ###
        t1 = time * 1000 #In seconds
        
        t2 = (time + lengthOfClip) * 1000 #In seconds
        
        time = time + 1
        newAudio = audio[t1:t2]
        txt = str(t1/(1000)) + " - " + str(t2/(1000))
        txt = "Videos/subClips/tmpClips/" + txt + ".wav"
        newAudio.export(txt, format="wav")
        ### ---Creates Clips--- ###
        
        
        clip = txt

        scale, sr = librosa.load(clip)

        FRAME_SIZE = 2048
        HOP_SIZE = 512

        S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

        Y_scale = np.abs(S_scale) ** 2

        Y_log_scale = librosa.power_to_db(Y_scale)
        
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
        
        
        count = 0
        for i in soundsSumArr[int(len(soundsSumArr)/2):]:
            if i > 6000:
                count = count + 1
        threequarterCount = 0
        for i in soundsSumArr[int(len(soundsSumArr)*3/4):]:
            if i > 6000:
                threequarterCount = threequarterCount + 1
        #print(count)
        if count < 35 or threequarterCount < 25:
            timeStampArr.append(txt)
        if count >= 35 and threequarterCount > 25:  
            successfulArr.append(txt)
            successfulTimeArr.append(t1/1000)
            weightedFrequencyArr.append(soundsSumArr)
    
    for fileInTmp in timeStampArr:
        os.remove(fileInTmp)

    removeExcessFiles = []

    for index, fileInTmp in enumerate(successfulArr):
        if index < len(successfulArr)-1:
            if successfulTimeArr[index+1] - successfulTimeArr[index] < 3:
                removeExcessFiles.append(fileInTmp)

    for fileInTmp in removeExcessFiles:
        os.remove(fileInTmp)
    
    for time in successfulTimeArr:
        originalFilePath = "Videos/subClips/" + file[:-4] + ".mp4"
        wavFilePath = "Videos/subClips/wavSubClips/" + file[:-4] + ".wav"
        t1 = time - 5
        t2 = time + 35
        txt = str(time-5) + " - " + str(time+35) + " from " + file[-21:-4]
        finalClipNamesArr.append(txt)
        ffmpeg_extract_subclip(originalFilePath, t1, t2, targetname= "Videos/subClips/finalClips/" + txt + ".mp4")
        ffmpeg_extract_subclip(wavFilePath, t1, t2, targetname= "Videos/subClips/finalWAVClips/" + txt + ".wav")
        break;
    print("done")

elapsed_time = timer() - start # in seconds
print("time:", elapsed_time, "seconds")