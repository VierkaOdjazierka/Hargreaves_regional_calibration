# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:26:00 2024

@author: Viera
"""

import xarray as xr
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import math 
import pandas as pd
import os
import glob
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt  
import numpy as np 
import pylab as pl 
import pylab



# =============================================================================
#              MEAN PARAMETER VALUES          
# =============================================================================

data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//1981_2000//a_parameter_values_WAPE.csv")
data_2=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//1981_2000//b_parameter_values_WAPE.csv")
data_3=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//1981_2000//c_parameter_values_WAPE.csv")

data_4=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//2001_2020//a_parameter_values_WAPE.csv")
data_5=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//2001_2020//b_parameter_values_WAPE.csv")
data_6=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//2001_2020//c_parameter_values_WAPE.csv")

#R of coefficients with altitude


plt.rcParams.update({'font.size':20})


fig,ax = plt.subplots(2, 3,figsize=(15,10))
s=10
fig.set_size_inches(15, 10, forward=True)
#plt.rcParams["figure.figsize"] = [15,5]
plt.rcParams["figure.autolayout"] = True
# Add scatterplot

x1=data_1['a param.']
y1=data_1['WAPE in %']
ax[0,0].plot(x1,y1,label='A coeff.',color='black', marker='o')
refx=np.zeros(len(y1))
refy=np.zeros(len(y1))
refx[:]=data_1[data_1['WAPE in %']==data_1['WAPE in %'].min()]['a param.']
refy[:]=data_1[data_1['WAPE in %']==data_1['WAPE in %'].min()]['WAPE in %']
ax[0,0].plot(refx,y1,label='A = '+str(refx[1]),color='red')
ax[0,0].legend()
ax[0,0].set_ylabel("WAPE [%]")

x2=data_2['b param.']
y2=data_2['WAPE in %']#[-50:]
ax[0,1].plot(x2,y2,label='B coeff.',color='black', marker='o')
refx2=np.zeros(len(y2))
refy2=np.zeros(len(y2))
refx2[:]=data_2[data_2['WAPE in %']==data_2['WAPE in %'].min()]['b param.']
refy2[:]=data_2[data_2['WAPE in %']==data_2['WAPE in %'].min()]['WAPE in %']
ax[0,1].plot(refx2,y2,label= 'B = '+str(round(refx2[1],2)),color='red')
ax[0,1].legend()
ax[0,1].set_xlabel("Coeff. value [-]")


x3=data_3['c param.']
y3=data_3['WAPE in %']
ax[0,2].plot(x3,y3,label='C coeff.',color='black', marker='o')
refx3=np.zeros(len(y3))
refy3=np.zeros(len(y3))
refx3[:]=data_3[data_3['WAPE in %']==data_3['WAPE in %'].min()]['c param.']
refy3[:]=data_3[data_3['WAPE in %']==data_3['WAPE in %'].min()]['WAPE in %']
ax[0,2].plot(refx3,y3,label='C = '+str(round(refx3[1],2)),color='red')
#ax[0,2].plot(refx3,y3,label='C = '+str(0.39),color='red')
ax[0,2].legend()


x4=data_4['a param.']
y4=data_4['WAPE in %']
ax[1,0].plot(x4,y4,label='A coeff.',color='black', marker='o')
refx4=np.zeros(len(y4))
refy4=np.zeros(len(y4))
refx4[:]=data_4[data_4['WAPE in %']==data_4['WAPE in %'].min()]['a param.']
refy4[:]=data_4[data_4['WAPE in %']==data_4['WAPE in %'].min()]['WAPE in %']
ax[1,0].plot(refx4,y4,label='A = '+str(refx4[1]),color='red')
ax[1,0].legend()
ax[1,0].set_ylabel("WAPE [%]")

x5=data_5['b param.']
y5=data_5['WAPE in %']#[-50:]
ax[1,1].plot(x5,y5,label='B coeff.',color='black', marker='o')
refx5=np.zeros(len(y5))
refy5=np.zeros(len(y5))
refx5[:]=data_5[data_5['WAPE in %']==data_5['WAPE in %'].min()]['b param.']
refy5[:]=data_5[data_5['WAPE in %']==data_5['WAPE in %'].min()]['WAPE in %']
ax[1,1].plot(refx5,y5,label= 'B = '+str(round(refx5[1],2)),color='red')
ax[1,1].legend()
ax[1,1].set_xlabel("Coeff. value [-]")


x6=data_6['c param.']
y6=data_6['WAPE in %']
ax[1,2].plot(x6,y6,label='C coeff.',color='black', marker='o')
refx6=np.zeros(len(y6))
refy6=np.zeros(len(y6))
refx6[:]=data_6[data_6['WAPE in %']==data_6['WAPE in %'].min()]['c param.']
refy6[:]=data_6[data_6['WAPE in %']==data_6['WAPE in %'].min()]['WAPE in %']
#ax[1,2].plot(refx6,y6,label='C = '+str(0.4),color='red')
ax[1,2].plot(refx6,y6,label='C = '+str(round(refx6[1],2)),color='red')
ax[1,2].legend()

fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//values of B and C coef in dif periods.jpeg',dpi=300, bbox_inches='tight')



##############################################################################################################################################



data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//res_years.csv")
data_1_B=data_1
fig,ax = plt.subplots(figsize=(15,7))
ax.scatter(data_1.iloc[:,0],data_1.iloc[:,1],label='B coef.',s=s,color='red')

r_squared1 = np.corrcoef(data_1.iloc[1:21,0],data_1.iloc[1:21,1])[0,1]
m, b = np.polyfit(data_1.iloc[1:21,0],data_1.iloc[1:21,1], 1)
ax.plot(data_1.iloc[1:21,0], m*data_1.iloc[1:21,0]+b,color='dimgrey',label='trendline 1981-2000 ... R='+str(round(r_squared1,2)),linestyle='dashed')

r_squared2 = np.corrcoef(data_1.iloc[21:,0],data_1.iloc[21:,1])[0,1]
m, b = np.polyfit(data_1.iloc[21:,0],data_1.iloc[21:,1], 1)
ax.plot(data_1.iloc[21:,0], m*data_1.iloc[21:,0]+b,color='black',label='trendline 2001-2020 ... R='+str(round(r_squared2,2)),linestyle='dotted')

r_squared3 = np.corrcoef(data_1.iloc[1:,0],data_1.iloc[1:,1])[0,1]
m, b = np.polyfit(data_1.iloc[1:,0],data_1.iloc[1:,1], 1)
ax.plot(data_1.iloc[1:,0], m*data_1.iloc[1:,0]+b,color='grey',label='trendline 1981-2020 ... R='+str(round(r_squared3,2)))
ax.set_xlabel("Year")
ax.set_ylabel("Coef. B value [-]")
ax.legend()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//B_coef_trends in two periods.jpeg',dpi=300, bbox_inches='tight')




data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//res_years_C.csv")
data_1_C=data_1
fig,ax = plt.subplots(figsize=(15,7))
ax.scatter(data_1.iloc[:,0],data_1.iloc[:,1],label='C coef.',s=s,color='red')

r_squared1 = np.corrcoef(data_1.iloc[1:21,0],data_1.iloc[1:21,1])[0,1]
m, b = np.polyfit(data_1.iloc[1:21,0],data_1.iloc[1:21,1], 1)
ax.plot(data_1.iloc[1:21,0], m*data_1.iloc[1:21,0]+b,color='dimgrey',label='trendline 1981-2000 ... R='+str(round(r_squared1,2)),linestyle='dashed')

r_squared2 = np.corrcoef(data_1.iloc[21:,0],data_1.iloc[21:,1])[0,1]
m, b = np.polyfit(data_1.iloc[21:,0],data_1.iloc[21:,1], 1)
ax.plot(data_1.iloc[21:,0], m*data_1.iloc[21:,0]+b,color='black',label='trendline 2001-2020 ... R='+str(round(r_squared2,2)),linestyle='dotted')

r_squared3 = np.corrcoef(data_1.iloc[1:,0],data_1.iloc[1:,1])[0,1]
m, b = np.polyfit(data_1.iloc[1:,0],data_1.iloc[1:,1], 1)
ax.plot(data_1.iloc[1:,0], m*data_1.iloc[1:,0]+b,color='grey',label='trendline 1981-2020 ... R='+str(round(r_squared3,2)))
ax.set_xlabel("Year")
ax.set_ylabel("Coef. C value [-]")
ax.legend()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//C_coef_trends in two periods.jpeg',dpi=300, bbox_inches='tight')




fig,ax = plt.subplots()
r_squared4 = np.corrcoef(data_1_B.iloc[:,1],data_1_C.iloc[:,1])[0,1]
ax.scatter(data_1_B.iloc[:,1],data_1_C.iloc[:,1],label='R='+str(round(r_squared4,2)))
ax.set_xlabel("Value of B coef. [-]")
ax.set_ylabel("Value of C coef. [-]")
ax.legend()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//relationship between B and C.jpeg',dpi=300, bbox_inches='tight')