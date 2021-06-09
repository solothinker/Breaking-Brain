import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.fft import fft, fftfreq,rfft, rfftfreq
import matplotlib.pyplot as plt

df =  pd.read_csv("allRawData.csv")
df = df.drop('Unnamed: 0',axis=True)
print(df.head())

# measuring the fft of the signal
def plotFFT(df):#,cleanDF):
    sampleRate = 512 # Hz
    duration = df.shape[0]
    
    xf = rfftfreq(duration,1/sampleRate)
    yf = rfft(df['value'])
    
    fig = plt.figure(num='FFT of signal', figsize=(20,10))
    plt.plot(xf,np.real(yf),label='raw')
    plt.legend()
    plt.grid()
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

plotFFT(df)
