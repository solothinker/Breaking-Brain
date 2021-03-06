import numpy as np
import pandas as pd
from scipy.signal import freqz
import matplotlib.pyplot as plt
from Filters import FilterWilter 

# Filter requirements.
order  = 6
fs     = 512.0  # sample rate, Hz
cutoff = 50     # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = FilterWilter.butter_lowpass(cutoff, fs, order)
FilterWilter.filtPlot(b,a,fs,cutoff=[cutoff])
# Plot the frequency response.
##FilterWilter.filtPlot(a,b,fs,cutoff)
##df = pd.read_csv('../data/allRawData.csv')
##data = df['value'].values
##sampleLen = 1000
##y = FilterWilter.butter_filtering(data[:sampleLen], cutoff, fs, order)
##
####plt.subplot(2, 1, 2)
##plt.plot(np.linspace(0,sampleLen/512,sampleLen),data[:sampleLen], 'b-', label='data')
##plt.plot(np.linspace(0,sampleLen/512,sampleLen),y, 'g-', linewidth=2, label='filtered data')
##plt.xlabel('Time [sec]')
####plt.grid()
##plt.locator_params(axis='x', nbins=10)
##plt.legend()
##
####plt.subplots_adjust(hspace=0.35)
##plt.show()

##xf,yf = FilterWilter.getFFT(y)
##_,dyf = FilterWilter.getFFT(data[:sampleLen])
##dPSD = np.abs(dyf)**2/sampleLen
##yPSD = np.abs(yf)**2/sampleLen
##plt.plot(xf,dPSD,'b-',label='raw')
##plt.plot(xf,yPSD,'g-',linewidth=2,label='filter')
##plt.xlim(1,cutoff*2)
##plt.ylim(0,2e4)
##plt.legend()
##plt.xlabel('Frequency [Hz]')
##plt.title('Frequency Plot')
##plt.locator_params(axis='x', nbins=10)
##plt.grid()
##plt.show()
##
##band = FilterWilter.eegBand(xf,yf)
##print(band)

    
##b, a = FilterWilter.butter_bandpass(0.25,50, fs, 10)

