# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:30:31 2021

@author: Condor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 300

class Spike:
    _ids = count(0)
    
    def __init__(self, peak, time):
        self.id = next(self._ids) #count spike number
        self.peak = peak
        self.time = time
        
    
    def peak_integrate(self, data, dx):
        points_duration = self.right - self.left
        time_duration = points_duration*dx
        # print(points_duration, 'points,', time_duration, 's')
        baseline_area = (data[self.left]+data[self.right])*time_duration*0.5
        integral = np.trapz(data.loc[self.left:self.right], dx=dx)
        self.integral = integral - baseline_area


'''
https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887
'''

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] < avgFilter[i-1]:                                   # spikes be negative boi
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))        


#df = pd.read_fwf('C:/Users/BRoehrich/Desktop/lsv4.txt', names=('time', 'current'), widths=[23,23])
df = pd.read_csv('D:/_DataProcessing/In use/1..txt', delimiter = '\t', skiprows = 1, names = ('time','potential','current','range'))
#df = df[1:]

df['current'] = df['current'].astype(float)
df['time'] = df['time'].astype(float)
df = df[df['time']>1]
df = df[df['time'] > 50]
df['current'] = df['current']*1e-3
# df['current'] = df['current'] + 1
current = df['current'].to_numpy()
time = df['time'].to_numpy()
dx = time[1] - time[0]

#%%

out = thresholding_algo(current, 500, 3, 0.01)
points = out['signals']
avg = out['avgFilter']
df['spike'] = points
df['avg'] = avg

spikes = {}



# My take on this
###
left_indexes = []
right_indexes = []

for i in range(df.index[1], df.index[0] + len(df)-1):
    if df.loc[(i,'spike')] - df.loc[(i+1),'spike'] in [1,2]:
        right_indexes.append(i)
        if df.loc[(i,'spike')] - df.loc[(i-1),'spike'] in [1,2]:
            left_indexes.append(i)
            continue
    elif df.loc[(i,'spike')] - df.loc[(i-1),'spike'] in [1,2]:
        left_indexes.append(i) 
    
        
# Make tuple pairs of the left and right indexes
if len(left_indexes) == len(right_indexes):
    indexes = tuple(zip(left_indexes,right_indexes))
else:
    print('\nerror, indexes have different lengths: left = {left}, right = {right}'
          .format(left = len(left_indexes),right = len(right_indexes)))


# Find the indexes of the spikes minima
spikes_min_indexes = []
for index in indexes:
    min_index = df.loc[index[0]:index[1],'current'].idxmin()
    spikes_min_indexes.append(min_index)
    spikes[min_index] = Spike(df.loc[min_index,'current'],df.loc[min_index,'time'])
    

hist = []   
for i in spikes:
    ## Set left bound for integration
    ## Boundary where current falls to less than the running average from thresholding_algo
    n_left = 0
    c = abs(df.loc[(i+n_left, 'current')])
    while c > abs(df.loc[i+n_left, 'avg']):         # this was previously just vs i
        n_left = n_left-1
        c = abs(df.loc[(i+n_left, 'current')])
        
    
    n_right = 0
    c = abs(df.loc[(i+n_right, 'current')])
    while c > abs(df.loc[i+n_right, 'avg']):        # this was previously just vs i
        n_right = n_right + 1
        try:
            c = abs(df.loc[(i+n_right, 'current')])
        except:
            print('Ran into end of dataset!')
            n_right = n_right-1
            break
        # print(c, abs(df.loc[i, 'avg']))
    
    spikes[i].left = int(i + n_left)
    spikes[i].right = int(i + n_right)
    

spikes_clean = {}
# Check that it is not a peak within another spike
for index, i in enumerate(spikes):
    # VALIDATION: print(spikes[i] == spikes[spikes_min_indexes[index]])
    ## Check with left neighbor if they share a common .left, keep the more negative spike
    try:
        if spikes[spikes_min_indexes[index]].left == spikes[spikes_min_indexes[index-1]].left:
            if spikes[spikes_min_indexes[index]].right == spikes[spikes_min_indexes[index+1]].right:
                #compare both left and right
                if spikes[spikes_min_indexes[index]].peak == spikes[spikes_min_indexes[index-1]].peak and spikes[spikes_min_indexes[index]].peak < spikes[spikes_min_indexes[index+1]].peak:    
                    spikes_clean[spikes_min_indexes[index]] = spikes[spikes_min_indexes[index]]
            else:
                #compare left only and take biggest
                if spikes[spikes_min_indexes[index]].peak < spikes[spikes_min_indexes[index-1]].peak:
                    spikes_clean[spikes_min_indexes[index]] = spikes[spikes_min_indexes[index]]
            
        ## Check with right neighbor if they share a common .right, keep the more negative spike
        elif spikes[spikes_min_indexes[index]].right == spikes[spikes_min_indexes[index+1]].right:
            if spikes[spikes_min_indexes[index]].peak < spikes[spikes_min_indexes[index+1]].peak:
                spikes_clean[spikes_min_indexes[index]] = spikes[spikes_min_indexes[index]]
        # if its a solo spike, take it
        else:
                spikes_clean[spikes_min_indexes[index]] = spikes[spikes_min_indexes[index]]
            
    except IndexError:
        continue
    
# Integrate the remaining spikes
for i in spikes_clean:
    spikes_clean[i].peak_integrate(df['current'], dx=dx)
    hist.append(1e15*spikes_clean[i].integral)
print('Integrated %s spikes.' %len(spikes_clean))




#%%
df['current/pA'] = df['current'] * 10**12
plt.figure()
plt.plot(time, df['current/pA'], '.-', label = 'current')
plt.xlabel('Time/ s')
plt.ylabel('Current/ pA')
out['avgFilter/pA'] = out['avgFilter'] * 10**12
out['stdFilter/pA'] = out['stdFilter'] * 10**12
for i in spikes_clean:
    plt.plot(spikes_clean[i].time, df.loc[(i, 'current/pA')], 'xr')
    #plt.annotate('A spike!', xy = (spikes[i].time, df.loc[(i, 'current')]), arrowprops = {'color':'orange'})
    #print(i)
plt.plot(time, out['avgFilter/pA'], '-r', label = 'mean')
plt.plot(time, out['avgFilter/pA'] + 3*out['stdFilter/pA'], '-g', label = '+ 3std')
plt.plot(time, out['avgFilter/pA'] - 3*out['stdFilter/pA'], '-g', label = '- 3std')
plt.xlim(96,97.5)
plt.ylim(-445, -390)
plt.legend()
plt.show()




plt.hist(hist, bins=np.arange(-10000,1000,500), rwidth=0.8)
plt.xlabel('Charge/ fC')
plt.ylabel('Count')