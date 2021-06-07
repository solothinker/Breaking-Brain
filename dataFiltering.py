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

# Plot the frequency response.
##FilterWilter.filtPlot(a,b,fs,cutoff)
df = pd.read_csv('allRawData.csv')
data = df['value'].values
sampleLen = 1000
y = FilterWilter.butter_lowpass_filter(data[:sampleLen], cutoff, fs, order)
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

xf,yf = FilterWilter.getFFT(y)
_,dyf = FilterWilter.getFFT(data[:sampleLen])
plt.plot(xf,np.multiply(dyf,dyf.conjugate())/sampleLen,'b-',label='raw')
plt.plot(xf,np.multiply(yf,yf.conjugate())/sampleLen,'g-',linewidth=2,label='filter')
##plt.xlim(1,cutoff*2)
##plt.ylim(0,500)
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.title('Frequency Plot')
plt.locator_params(axis='x', nbins=10)
plt.grid()
plt.show()
