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
    
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_filtering(data,b,a ):
        y = lfilter(b, a, data)
        return y
           
    def filtPlot(b,a,fs,cutoff=1,worN=2000):
        
        w, h = freqz(b, a, worN=worN)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Filter")
        xlimit = max(cutoff)*2
        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--g', label='sqrt(0.5)')

        for ii in cutoff:
            plt.plot(ii, 0.5*np.sqrt(2), 'o',label = 'Cutoff_%.2f Hz' %ii)
            plt.axvline(ii, color='k')

        plt.xlim(0,xlimit)
        plt.ylim(0,1.2)
        plt.xlabel('Frequency [Hz]')
        plt.title('Filter Response')
        plt.legend()
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

    def eegBand(fft_freq,fftVal):
        # Define EEG bands
        eeg_bands = {'Delta': (0.5, 3),
             'Theta': (4, 7),
             'Alpha': (8, 13),
             'Beta': (14, 30),
             'Gamma': (30, 45)}
        
        PSD = np.abs(fftVal)**2/len(fftVal)
        eegBandPSD = dict()

        for band in eeg_bands:  
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            eegBandPSD[band] = np.sum(PSD[freq_ix])
        eegBandPSD['R'] = eegBandPSD['Alpha']/eegBandPSD['Beta']
        
        return eegBandPSD
    
    def outlier_smoother(x, m=3, win=3, plots=False):
        #https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
        ''' finds outliers in x, points > m*mdev(x) [mdev:median deviation] 
        and replaces them with the median of win points around them '''
        x_corr = np.copy(x)
        d = np.abs(x - np.median(x))
        mdev = np.median(d)
        idxs_outliers = np.nonzero(d > m*mdev)[0]
        for i in idxs_outliers:
            if i-win < 0:
                x_corr[i] = np.median(np.append(x[0:i], x[i+1:i+win+1]))
            elif i+win+1 > len(x):
                x_corr[i] = np.median(np.append(x[i-win:i], x[i+1:len(x)]))
            else:
                x_corr[i] = np.median(np.append(x[i-win:i], x[i+1:i+win+1]))
        if plots:
            plt.figure('outlier_smoother', clear=True)
            plt.plot(x, label='orig.', lw=5)
            plt.plot(idxs_outliers, x[idxs_outliers], 'ro', label='outliers')                                                                                                                    
            plt.plot(x_corr, '-o', label='corrected')
            plt.legend()
            plt.show()
        
        return x_corr
    def localNorm(x):
        return(x-x.min())/(x.max()-x.min())

if __name__ == "__main__":
    FilterWilter()
