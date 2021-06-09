import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


NSdf = pd.read_csv('allWaves.csv')
caldf = pd.read_csv('bandsofbrain.csv')
##plt.subplot(2,1,1)
##plt.plot(NSdf['DELTA'],label='Delta of NS')
##plt.plot(caldf['Delta'],label='Delta of Cal')
##plt.legend(loc = "upper right")
##plt.grid()
##plt.ylim(0,1e6)
data = caldf['Delta']
z = np.abs(stats.zscore(data))
threshold = 3
##print(np.where(z > 3))

#IQR (Inter Quartile Range)
##Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
##  
##Q3 = np.percentile(data, 75,interpolation = 'midpoint') 
##IQR = Q3 - Q1
##print(IQR)
##upper = data >= (Q3+1.5*IQR)
##lower = data <= (Q1-1.5*IQR)
##print(np.where(upper))
##print(np.where(lower))
##plt.plot(data)
##plt.plot(data[upper])
##plt.show()
##plt.plot(z)
##plt.show()
q1 = caldf['Delta'].quantile(.5)
q3 = caldf['Delta'].quantile(.95)
mask = caldf['Delta'].between(q1, q3, inclusive=True)
iqr = caldf.loc[mask, 'Delta']

q1 = NSdf['DELTA'].quantile(.5)
q3 = NSdf['DELTA'].quantile(.95)
mask = NSdf['DELTA'].between(q1, q3, inclusive=True)
iqr2 = NSdf.loc[mask, 'DELTA']
iqr = (iqr-iqr.mean())/iqr.std()
iqr2 = (iqr2-iqr2.mean())/iqr2.std()
plt.subplot(2,1,1)
plt.plot(iqr,label='Delta')
plt.plot(iqr2,label='DELTA')
plt.legend()
plt.subplot(2,1,2)
##plt.plot(iqr2.values-iqr.values)
plt.show()
