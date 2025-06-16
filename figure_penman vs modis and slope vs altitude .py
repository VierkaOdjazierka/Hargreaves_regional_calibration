# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:29:37 2024

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

dataset_daily=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//comparison_of_HG_modiffications.csv")
dataset_daily.index=dataset_daily.iloc[:,0]
dataset_daily=dataset_daily.iloc[:,1:]

dataset_monthly=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//comparison_of_HG_modiffications_monthly.csv")
dataset_monthly.index=dataset_monthly.iloc[:,0]
dataset_monthly=dataset_monthly.iloc[:,1:]

dataset_yearly=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//comparison_of_HG_modiffications_yearly.csv")
dataset_yearly.index=dataset_yearly.iloc[:,0]
dataset_yearly=dataset_yearly.iloc[:,1:]


######################################################################################################################################################################################################
###################################################################################################

dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//statistics_of_HG_modiffications_datasets.csv")
dataset_.index=dataset_.iloc[:,0]
dataset_=dataset_.iloc[:,1:]
dataset_=dataset_[[ 'mean_PM','mean_harg_','mean_modif_droog', 'mean_livia_', 'mean_Berti', 'mean_droogersAllen2','mean_traj_balk']]



fig, (ax1, ax2) = plt.subplots(1, 2)
s=10
# Add scatterplot
y =dataset_['mean_PM']  
x = dataset_['mean_harg_']
r_squared = np.corrcoef(x,y)[0,1]
ax1.scatter(x,y,label='Hargreaves ET0',s=s,color='black')
m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b,label="R="+str(round(r_squared,2)),color='dimgrey')
ax1.set_xlabel("ET0 [mm*deň-1] / Hargreaves ET0")
ax1.set_ylabel("ET0 [mm*deň-1] / P-M FAO56 ET0")
ax1.legend(loc='lower right')


y1 =dataset_['mean_PM']  
x1 = dataset_['mean_modif_droog']
r_squared = np.corrcoef(x1,y1)[0,1]
ax2.scatter(x1,y1,label='Modif. ET0',s=s,color='black')
m, b = np.polyfit(x1, y1, 1)
ax2.plot(x1, m*x1+b,label="R="+str(round(r_squared,2)),color='dimgrey')
ax2.set_xlabel("ET0 [mm*day-1] / Modif. method ET0")
ax2.set_ylabel("ET0 [mm*day-1] / P-M FAO56 ET0")
ax2.legend(loc='lower right')

fig.tight_layout(pad=2.0)
#fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//R_hargreaves_modif.jpeg',dpi=300, bbox_inches='tight')



###################################################################################################
###################################################################################################

fig, ax = plt.subplots()
s=20
# Add scatterplot
y =dataset_['mean_PM']  
x = dataset_['mean_harg_']
r_squared = np.corrcoef(x,y)[0,1]
ax.scatter(x,y,label='Harg.orig. R='+str(round(r_squared,2)),marker=".",color='black',s=s)
m, b = np.polyfit(x, y, 1)
ax.plot(x, m*x+b,color='black',linestyle='dotted',label="Regr.line Hargr.")


# y1 =dataset_['mean_PM']  
# x1 = dataset_['mean_modif_droog']
# r_squared = np.corrcoef(x1,y1)[0,1]
# ax.scatter(x1,y1,label='Modif',marker="v",color='firebrick',s=s)
# m, b = np.polyfit(x1, y1, 1)
# ax.plot(x1, m*x1+b,color='firebrick',linestyle=(0, (3, 10, 1, 10)),label="Regr.line modif.")

# y =dataset_['mean_PM']  
# x = dataset_['mean_livia_']
# r_squared = np.corrcoef(x,y)[0,1]
# ax.scatter(x,y,label='HG+R',marker="s",color='silver',s=s)
# m, b = np.polyfit(x, y, 1)
# ax.plot(x, m*x+b,color='dimgrey',linestyle=(0, (3, 5, 1, 5, 1, 5)),label="Regr.line HG+R")


y1 =dataset_['mean_PM']  
x1 = dataset_['mean_modif_droog']
r_squared = np.corrcoef(x1,y1)[0,1]
ax.scatter(x1,y1,label='Modif. R='+str(round(r_squared,2)),marker="x",color='dimgrey',s=s)
m, b = np.polyfit(x1, y1, 1)
ax.plot(x1, m*x1+b,color='dimgrey',linestyle=(0, (1, 10)),label="Regr.line modif.")


ax.legend(bbox_to_anchor=(1.05, 1))
ax.set_xlabel("ET0 [mm*day] - Harg. + modif. ")
ax.set_ylabel("ET0 [mm*day] - P-M ")

#fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//scaterplot_modif_harg.jpeg',dpi=300, bbox_inches='tight')
#######################################################################################################



#comparison of the methods

#print(dataset_daily.columns) #'WAPE_harg_','WAPE_harg_Rs', 'WAPE_livia_', 'WAPE_modif_', 'WAPE_modif_droog','WAPE_droogersAllen1', 'WAPE_droogersAllen2', 'WAPE_traj_balk','WAPE_traj_pol'],

fig, axs = plt.subplots()

count, bins_count = np.histogram(dataset_daily['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Daily HG-ET0",linestyle='dashdot',color='black')


count, bins_count = np.histogram(dataset_monthly['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Monthly HG-ET0",linestyle='dotted',color='black')


count, bins_count = np.histogram(dataset_yearly['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Yearly HG-ET0",linestyle='dashed',color='black')

########## #modiff

count, bins_count = np.histogram(dataset_daily['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Daily Modif-ET0",linestyle='dashdot',color='red')


count, bins_count = np.histogram(dataset_monthly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Monthly Modif-ET0",linestyle='dotted',color='red')


count, bins_count = np.histogram(dataset_yearly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Yearly Modif-ET0",linestyle='dashed',color='red')


plt.legend(bbox_to_anchor=(0.5, -0.5),loc='lower center',ncol=2) #, bbox_to_anchor=(-0.2, -0.2),
#axs.set_title('Comparison of proposed modif. performance in differend resolution')
plt.xlabel("WAPE in %")
plt.ylabel("CDF")
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//comparison of hargreaves and modif.jpeg',dpi=300, bbox_inches='tight')


#######################################################################################################
fig, axs = plt.subplots()

count, bins_count = np.histogram(dataset_daily['WAPE_droogersAllen2'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="denné Allen-ET0",color='dimgrey',linestyle='dashdot')


count, bins_count = np.histogram(dataset_monthly['WAPE_droogersAllen2'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="mesačné Allen-ET0", color='dimgrey',linestyle='dotted',linewidth=2)


count, bins_count = np.histogram(dataset_yearly['WAPE_droogersAllen2'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="ročné Allen-ET0",color='dimgrey',linestyle='dashed')

########## #modiff

count, bins_count = np.histogram(dataset_daily['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="denné Modif-ET0",linestyle='dashdot', color='black')


count, bins_count = np.histogram(dataset_monthly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="mesačné Modif-ET0",linestyle='dotted', color='black',linewidth=2)


count, bins_count = np.histogram(dataset_yearly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="ročné Modif-ET0",linestyle='dashed', color='black')



plt.legend(bbox_to_anchor=(0.5, -0.8),loc='lower center',ncol=2) #, bbox_to_anchor=(-0.2, -0.2),
#axs.set_title('Comparison of proposed modif. performance in differend resolution')
plt.xlabel("WAPE in %")
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//comparison of dodgersallen and modif.jpeg',dpi=300, bbox_inches='tight')




dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//comparison_of_HG_modiffications.csv")
dataset_=dataset_['WAPE_harg_']
dataset_nmv=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//statistics_of_HG_modiffications_datasets.csv")
dataset_nmv=dataset_nmv['nadm_vyska']

fig, ax = plt.subplots()
r_squared4 = np.corrcoef(dataset_,dataset_nmv)[0,1]
ax.scatter(dataset_,dataset_nmv,label='R='+str(round(r_squared4,2)))
ax.set_xlabel("Harg. WAPE [%]")
ax.set_ylabel("Altitude [m a.s.l.]")
ax.legend()
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//relationship between hargWAPE and altitude.jpeg',dpi=300, bbox_inches='tight')