#Ref-https://gist.github.com/junzis/e06eca03747fc194e322

import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class FilterWilter:
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = FilterWilter.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def filtPlot(a,b,fs,cutoff):
        w, h = freqz(b, a, worN=8000)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()

    def getFFT(yt,Samples=512.0):
        N = len(yt)
        T = 1.0/Samples
        yf = fft(yt)[:N//2]
        xf = fftfreq(N,T)[:N//2]
        yf = yf.reshape(len(yf),1)
        xf = xf.reshape(len(xf),1)
        return xf,yf

if __name__ == "__main__":
    FilterWilter()
