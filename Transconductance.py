# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:34:56 2021

@author: Condor
"""
#%%
### Import packages and load/clean data

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import os

plt.rcParams['figure.dpi'] = 300

# Get the data
files = os.listdir('D:/_DataProcessing/In use')
col_names = ['time', 'Ewe', 'current', 'range']


gate = pd.read_csv('D:/_DataProcessing/In use/' + files[1], sep = '\t', usecols = range(0,4), names = col_names, skiprows=1)
gate = gate[gate['time'] > 10]
gate['current'] = gate['current']*10**3 

gate2 = pd.read_csv('D:/_DataProcessing/In use/' + files[2], sep = '\t', usecols = range(0,4), names = col_names, skiprows=1)
gate2 = gate2[gate2['time'] > 10 + gate2['time'].iloc[0]]
gate2 = gate2.iloc[::10,:]
gate2['current'] = gate2['current']*10**3 

channel = pd.read_csv('D:/_DataProcessing/In use/' + files[0], sep = '\t', usecols = range(0,4), names = col_names, skiprows = 1)
channel1 = channel[channel['time'] > 10]
channel1 = channel1[channel1['time'] <= gate['time'].iloc[-1]]
channel1['current'] = channel1['current']*10**3

channel2 = channel[channel['time'] > gate2['time'].iloc[0]]
channel2 = channel2[channel2['time'] <= gate2['time'].iloc[-1]]
channel2['current'] = channel2['current']*10**3
#channel2 = channel2.iloc[::10,:] 
#%%
gm1 = pd.DataFrame(zip(gate['Ewe'],channel1['current']), columns = ['E_gate', 'i_channel'])
gm1['transcon'] = gm1['i_channel'].diff() / gm1['E_gate'].diff()
gm1.loc[abs(gm1['transcon'].diff()) >10,'transcon' ] = np.nan
#gm1.loc[gm1['transcon'] > 10,'transcon'] = np.nan
#gm1.loc[gm1['transcon'] < -40,'transcon'] = np.nan

gm2 = pd.DataFrame(zip(gate2['Ewe'],channel2['current']), columns = ['E_gate', 'i_channel'])
gm2['transcon'] = gm2['i_channel'].diff() / gm2['E_gate'].diff()
gm2.loc[abs(gm2['transcon'].diff()) >50,'transcon' ] = np.nan

#%%
### Display the data
fig, ax = plt.subplots()
ax.plot('E_gate', 'i_channel', data = gm1, label = '10   mV/s')
ax.plot('E_gate', 'i_channel', data = gm2, label = '100 mV/s')
ax2 = ax.twinx()
ax2.plot('E_gate', 'transcon', data = gm1, label = 'g_m', color = 'green')
ax2.plot('E_gate', 'transcon', data = gm2, label = 'g_m2', color = 'red')
ax.set_xlabel('E_gate / V')
ax.set_ylabel('i_channel / µA')
ax2.set_ylabel('g_m / µS')
ax.set_title('Transconductance of PEDOT:PSS in 1x PBS')

#lines, labels = ax.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
#ax.legend(lines + lines2, labels + labels2)
fig.legend(bbox_to_anchor=(1.1, 1.1), loc = 'upper right')

ax.grid(axis = 'x')
plt.show()


fig,ax = plt.subplots()
ax.plot('Ewe','current', data = gate, label = '10mV/s')
ax.plot('Ewe','current', data = gate2, label = '100mV/s')
ax.set_title('CVs')
ax.set_xlabel('Ewe / V')
ax.set_ylabel('i / µA')
fig.legend(bbox_to_anchor=(1.05, 1), loc = 'upper right')


fig, ax = plt.subplots()
ax.plot('time','current', data = gate, label = 'gate')
ax.plot('time','current', data = channel1, label = 'channel')
ax.set_title('Current vs time')
ax.set_xlabel('time /s')
ax.set_ylabel('current /µA')
fig.legend(bbox_to_anchor=(1.05, 1), loc = 'upper right')

