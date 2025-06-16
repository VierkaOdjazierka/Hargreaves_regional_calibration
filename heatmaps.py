# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:52:46 2024

@author: Viera
"""

import xarray as xr
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import math 
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 

#pip install tensorflow

from sklearn.preprocessing import StandardScaler



import hydroeval as he
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

from mpl_toolkits.axes_grid1 import make_axes_locatable

#########################################################################################
#stat. info

stat_info=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//stat_info.csv", sep=",",header=None,error_bad_lines=False)
stat_info.columns=stat_info.iloc[0,:]
stat_info.drop([0],inplace=True)
stat_info=stat_info.iloc[:,1:]
stat_info.columns=['Indikatív', 'altitude' ,'latit_deg' ,'latit_min']
print(stat_info)
stat_info['Indikatív']=stat_info['Indikatív'].astype(int)
#########################################################################################
#data loading

data_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//ress_matrix_b.csv", sep=",",header=None,error_bad_lines=False)#error_bad_lines=False,warn_bad_lines=False
data_.index=data_.iloc[:,0]
data_.columns=data_.iloc[0,:].astype(int)
data_=data_.iloc[1:,1:]
data_=data_.T
data_.index=round(pd.DataFrame(np.array([x * 0.1 for x in range(220)][170:]).astype(float)),2).iloc[:,0]
data_=data_.iloc[:,1:]


data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//ress_matrix_C.csv", sep=",",header=None,error_bad_lines=False)#error_bad_lines=False,warn_bad_lines=False
data_1.index=data_1.iloc[:,0]
data_1.columns=data_1.iloc[0,:].astype(int)
data_1=data_1.iloc[1:,1:]
data_1=data_1.T
data_1.index=round(pd.DataFrame(np.array([x * 0.001 for x in range(70)][20:]).astype(float)),3).iloc[:,0]
data_1=data_1.iloc[:,1:]
#del data_1['altitude']
#del data_['altitude']
#########################

#########################################################################################

# FIGURE

plt.rcParams.update({'font.size':20})
fig,axes=plt.subplots(1,2,figsize=(20,10))
sns.heatmap(ax=axes[0],data=data_,cmap='coolwarm',linecolor='black')#
axes[0].set(xlabel='Years',ylabel='Value of B coef. [-]')
axes[0].set_title('WAPE [%]')

sns.heatmap(ax=axes[1],data=data_1,cmap='coolwarm',linecolor='black')#
axes[1].set(xlabel='Years',ylabel='Value of C coef. [-]')
axes[1].set_title('WAPE [%]')

