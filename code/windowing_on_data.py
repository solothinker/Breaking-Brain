import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Filters import FilterWilter


plt.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("../data/allRawData.csv")
data = df['value'].values

samples = 512.0
sampleLen = len(data)#10000
order  = 6
cutoff = 50     # desired cutoff frequency of the filter, Hz

time = np.linspace(0,sampleLen/samples,sampleLen)
data = data[:sampleLen]

b, a = FilterWilter.butter_lowpass(cutoff, samples, order)
hann = np.hanning(samples)
hann = 1
signals = ['Delta','Theta','Alpha','Beta','Gamma','R']
bandDict = dict()
for ii in signals:        
    bandDict[ii] = []
    
for ii in range(256,sampleLen,512):
    if ii+256>= sampleLen:
        continue
    
    t = time[ii-256:ii+256]
    y = data[ii-256:ii+256]
    yf = FilterWilter.butter_lowpass_filter(y*hann,b,a)
    
    plt.subplot(2,2,1)
    plt.plot(t,y,'b-',label='raw')
    plt.plot(t,yf,'g-',linewidth=2,label='filter')
    plt.legend(loc = "upper right")
    plt.xlabel('Time [sec]')
    plt.title("Time Response")
    plt.locator_params(axis='x', nbins=10)
    plt.grid()

    plt.subplot(2,2,2)
    xf,yf = FilterWilter.getFFT(yf)
    _,dyf = FilterWilter.getFFT(y)
    
    plt.plot(xf,np.abs(dyf),'b-',label='raw')
    plt.plot(xf,np.abs(yf),'g-',linewidth=2,label='filter')
    plt.xlim(0,cutoff*1.5)
    plt.ylim(0,5e3)
    plt.title("Frequency Response")
    plt.xlabel("Frequency [Hz]")
    plt.grid()
    plt.legend()
    
    temp = FilterWilter.eegBand(xf,yf)
    plt.subplot(2,2,(3,4))
    for ii in signals:        
        bandDict[ii] += [temp[ii]]
        plt.plot(bandDict[ii],label=ii)
    plt.legend(loc = "upper right")
    plt.grid()
    plt.title("Band")
    plt.xlabel("Index")

    plt.draw()
    plt.pause(0.001)
    plt.clf()
    
plt.close()    

bandDF = pd.DataFrame.from_dict(bandDict)
print(bandDF.head())
bandDF.to_csv('../data/bandsofbrain.csv')
