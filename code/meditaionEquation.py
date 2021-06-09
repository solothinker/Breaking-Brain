import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.fft import fft, fftfreq,rfft, rfftfreq
import matplotlib.pyplot as plt

df =  pd.read_csv("../data/allWaves.csv")
df = df.drop('Unnamed: 0',axis=True)
print(df.head())
##print(df.columns)

# separating brain waves and tasks
waveCols = ['DELTA', 'LOALPHA', 'HIBETA', 'MIDGAMMA','LOGAMMA', 'LOBETA','THETA', 'HIALPHA']
tsakCols = ['ATTENTION', 'MEDITATION','APPRECIATION', 'MENTAL_EFFORT', 'FAMILIARITY']

###ploting the original data
##fig = plt.figure(num='wave data', figsize=(15,10))
##for ind,col in enumerate(waveCols):
##    plt.subplot(len(waveCols),1,ind+1)
##    plt.plot(df[col],label=col)
##    plt.legend()
##    plt.grid()
##plt.draw()
##plt.waitforbuttonpress(0)
##plt.close(fig)
###removing the outlier from data using zscore
##threshold=2    
##zScore = zscore(df.DELTA)
##absZscore = np.abs(zScore)
##removeSpike = (absZscore<threshold)#.all(axis=1)
##cleanDF = df[removeSpike]
##
### comparing the clean and original data
##fig = plt.figure(num='comparing the data', figsize=(15,10))
##for ind,col in enumerate(waveCols):
##    plt.subplot(len(waveCols),1,ind+1)
##    plt.plot(df[col],label=col)
##    plt.plot(cleanDF[col],label=col+"_clean")
##    plt.legend()
##    plt.grid()
##plt.draw()
##plt.waitforbuttonpress(0)
##plt.close(fig)
##
### ploting the clean signal
##fig = plt.figure(num='clean data', figsize=(15,10))
##for ind,col in enumerate(waveCols):
##    plt.subplot(len(waveCols),1,ind+1)
##    plt.plot(cleanDF[col],label=col+"_clean")
##    plt.legend()
##    plt.grid()
##plt.draw()
##plt.waitforbuttonpress(0)
##plt.close(fig)

# measuring the fft of the signal
def plotFFT(df):#,cleanDF):
    sampleRate = 1 # Hz
    duration = df.shape[0]
    xf = fftfreq(duration)
##    print(xf)
##    xfr = rfftfreq(cleanDF.shape[0],1/sampleRate)
    fig = plt.figure(num='FFT of signal', figsize=(20,10))
    for ind,col in enumerate(waveCols[3:]):
        yf = fft(df[col])
##        yfr = rfft(cleanDF[col])
        plt.subplot(len(waveCols[3:]),1,ind+1)
        plt.plot(np.real(yf),label=col)
##        plt.plot(xfr,np.real(yfr),label=col+"_clean")
        plt.legend()
        plt.grid()
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
plotFFT(df)
##plotFFT(cleanDF)
