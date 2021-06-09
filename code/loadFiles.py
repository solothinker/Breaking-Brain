import numpy as np
import pandas as pd

with open("1091.csv", "r") as file:
    data = file.read()

# spliting the data into rows 
data = data.split("\n")
data.pop(-1)
dataPack = []

# storing the eeg data in list
for da in data:
    dataPack.append(da.split(" "))

# making the dataframe
df = pd.DataFrame(dataPack,columns=["time","signals","value"])
df['time'] = pd.to_datetime(df['time'].values,unit='ms').time # changing to HH:MM:SS.F formate
print(df.head())
df.to_csv("OriginalData.csv")

# storing the unique signals name
waveName = df['signals'].unique()
# separating the Raw and the other wave signals
rawInd = df['signals'] == waveName[0]
dfRaw = df[rawInd]
dfRaw = dfRaw.drop("signals",axis=True)
dfWaves = df[~rawInd]
dfRaw.to_csv("allRawData.csv")
print(dfRaw.head())
print(dfWaves.head())

df = pd.DataFrame()

for wave in waveName[4:]:
    ind = dfWaves['signals'] == wave
    df[wave] = dfWaves[ind]['value'].values
    
df.index = dfWaves[ind]['time'].values

print(df.head())
df.to_csv("allWaves.csv")

df = pd.DataFrame()

for wave in waveName[1:4]:
    df[wave] = dfWaves[dfWaves['signals'] == wave]['value'].values

print(df.head())
df.to_csv("accelerometerData.csv")
