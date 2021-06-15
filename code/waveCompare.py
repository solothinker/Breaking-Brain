import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from Filters import FilterWilter


NSdf = pd.read_csv('../data/allWaves.csv')
caldf = pd.read_csv('../data/bandsofbrain.csv')
data = caldf['Delta']
z = np.abs(stats.zscore(data))
threshold = 3

cal = FilterWilter.outlier_smoother(data.values, m=3, win=5)
NS = FilterWilter.outlier_smoother(NSdf['DELTA'].values, m=3, win=5)

plt.plot(FilterWilter.localNorm(cal))
plt.plot(FilterWilter.localNorm(NS))
plt.show()

xf,fcal = FilterWilter.getFFT(cal,Samples=1.0)
xff,fNS = FilterWilter.getFFT(NS,Samples=1.0)

plt.plot(xf,np.abs(fcal),label='cal')
plt.plot(xff,np.abs(fNS),label='NS')
plt.xlim(0.02,0.5)
plt.legend()
plt.show()
