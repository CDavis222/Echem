# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:22:37 2022

@author: cdavis
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import os
import seaborn as sns
plt.rcParams['figure.dpi'] = 500


# Set the working folder
os.chdir('Z:/Projects/Connor/GOX kinetics/GOx_kinetics_in_use')

#%%

folders = os.listdir()
col_names = ['time/s','Ewe/V','I/mA','I Range']

concs_total = []
coefs_total = []


for folder in folders:
    file_names = []
    lin_fits = []
    df_list=[]
    fig, ax = plt.subplots()
    ax.set_xlabel('time (s)')
    ax.set_ylabel('current (pA)')
    ax.set_title(folder)
    for file in os.listdir(os.getcwd()+ '\\' + folder + '\\'):
        df_list.append(pd.read_csv('Z:/Projects/Connor/GOX kinetics/GOx_kinetics_in_use/' + folder + '/' + file, sep = '\t',skiprows =1, names =col_names))
        file_names.append(file)
    for index,df in enumerate(df_list):
        df['I/pA'] = df['I/mA'] * 10**9
        Current = df.loc[df['time/s']>100, 'I/pA'].to_numpy().reshape(-1,1)
        Time = df.loc[df['time/s']>100,'time/s'].to_numpy().reshape(-1,1)
        
        #Do linear regression and fit line
        lin_regressor = LinearRegression()
        lin_regressor.fit(Time,Current)
        lin_fits.append([lin_regressor.coef_[0][0], lin_regressor.intercept_[0]])
        predicted = lin_regressor.predict(Time)
        
        #Plot
        ax.scatter(Time.flatten(),Current.flatten())
        ax.plot(Time.flatten(),predicted, label = '{} slope = {:.2f} pA/s \n intercept = {:.2f} pA'.format(file_names[index].strip('.txt'),lin_regressor.coef_[0][0],lin_regressor.intercept_[0]))
        
    fig.legend(bbox_to_anchor = (1.4,0.85),loc = "upper right", title = 'Glucose concentration and fitting')
    plt.show()
    
    
    # Fit michaelis-menten then
    # plot the slopes as a function of concentration w/ fit
    #concs = [10,15,20,35,50,7.5]
    ###automate getting the concentrations
    concs = [float(file_name.split('. ')[1].strip('mM.txt')) for file_name in file_names]
    coefs = [item[0] for item in lin_fits]
    if 0 not in concs:
        concs.insert(0,0.0)
        coefs.insert(0,0.0)
    
    ## Add the data to aggregate dataframe
    concs_total.append(concs)
    coefs_total.append(coefs)
    
    ## OLD SORTING METHOD - FILES NOW SPECIFIED IN ORDER
    #data = list(zip(concs, coefs))
    #if 0 not in data:
    #    data.insert(0,(0,0))
    
    #data.sort()
    #concs2 = [item[0] for item in data]
    #coefs2 = [item[1] for item in data]
    
    ####MM fitting
    def model(substrate_concs, Vm, Km):
        sub = np.array(substrate_concs)
        return (Vm * sub)/(sub + Km)
    
    popt,pcov = curve_fit(model, concs, coefs, bounds = (0,[5,100]))
    pred_MM = model(concs, *popt)
    
    #popt2,pcov2 = curve_fit(model, concs, coefs, bounds = (0,[1.5,100]))
    #pred_MM2 = model(concs, *popt2)
    
    #Plotting
    fig2,ax2 = plt.subplots()
    ax2.set_xlabel('Glucose concentration (mM)')
    ax2.set_ylabel('Slope (pA/s)')
    ax2.set_title(folder)
    
    ax2.scatter(concs, coefs)
    ax2.plot(concs, pred_MM)
    #ax2.plot(concs, pred_MM2)
    ax2.annotate("V_max is {:.2f}\nKm is {:.2f}".format(popt[0], popt[1]),xy = (0.05,0.75),xycoords = 'axes fraction', fontsize = 10, color = 'blue')
    #ax2.annotate("Bound Vmax = 1.5\nV_max is {:.2f}\nKm is {:.2f}".format(popt2[0], popt2[1]),xy = (0.7,0.25),xycoords = 'axes fraction', fontsize = 10)                                                  
    for x,y in zip(concs, coefs):
        ax2.annotate('{}'.format(x),xy = (x,y))
plt.show()    
## Make a dataFrame and plot the aggregate totals
concs_total_flat = [item for sublist in concs_total for item in sublist]
coefs_total_flat = [item for sublist in coefs_total for item in sublist]
agg_df = pd.DataFrame(data = {'Concentration (mM)':concs_total_flat, 'Slope (pA/s)' : coefs_total_flat})

ax3 = sns.lineplot(x = 'Concentration (mM)', y = 'Slope (pA/s)', data =agg_df,err_style = 'bars', ci = 'sd', marker = 'o')
ax3.set_title('Aggregate data w/ Std. dev error')
    
