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

dataset_daily=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//comparison_of_HG_modiffications.csv")
dataset_daily.index=dataset_daily.iloc[:,0]
dataset_daily=dataset_daily.iloc[:,1:]

dataset_monthly=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//comparison_of_HG_modiffications_monthly.csv")
dataset_monthly.index=dataset_monthly.iloc[:,0]
dataset_monthly=dataset_monthly.iloc[:,1:]

dataset_yearly=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//comparison_of_HG_modiffications_yearly.csv")
dataset_yearly.index=dataset_yearly.iloc[:,0]
dataset_yearly=dataset_yearly.iloc[:,1:]

############################################################
#comparison of the methods

print(dataset_daily.columns) #'WAPE_harg_','WAPE_harg_Rs', 'WAPE_livia_', 'WAPE_modif_', 'WAPE_modif_droog','WAPE_droogersAllen1', 'WAPE_droogersAllen2', 'WAPE_traj_balk','WAPE_traj_pol'],

fig, axs = plt.subplots()

count, bins_count = np.histogram(dataset_daily['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Hargreaves")


count, bins_count = np.histogram(dataset_daily['WAPE_livia_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M1")


count, bins_count = np.histogram(dataset_daily['WAPE_Berti'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M2")


count, bins_count = np.histogram(dataset_daily['WAPE_droogersAllen2'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M3")


count, bins_count = np.histogram(dataset_daily['WAPE_traj_balk'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M4")

# count, bins_count = np.histogram(dataset_daily['WAPE_Ravazzani'], bins=10)
# pdf = count / sum(count)
# cdf = np.cumsum(pdf)
# axs.plot(bins_count[1:], cdf, label="M5")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
axs.set_title('Comparison of Hargreaves modiffications performance')
plt.xlabel("WAPE v %")
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()


############################################################
############################################################
############################################################



fig, axs = plt.subplots()

count, bins_count = np.histogram(dataset_daily['R_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Hargreaves")




count, bins_count = np.histogram(dataset_daily['R_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Modif.")


count, bins_count = np.histogram(dataset_daily['R_livia_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M1")

count, bins_count = np.histogram(dataset_daily['R_Berti'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M2")


# count, bins_count = np.histogram(dataset_daily['R_droogersAllen2'], bins=10)
# pdf = count / sum(count)
# cdf = np.cumsum(pdf)
# axs.plot(bins_count[1:], cdf, label="M3")


count, bins_count = np.histogram(dataset_daily['R_traj_balk'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="M3")

# count, bins_count = np.histogram(dataset_daily['R_traj_pol'], bins=10)
# pdf = count / sum(count)
# cdf = np.cumsum(pdf)
# axs.plot(bins_count[1:], cdf, label="M5")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=2) #, bbox_to_anchor=(-0.2, -0.2),
#axs.set_title('Comparison of Hargreaves modiffications performance')
plt.xlabel("Pearson R [-]")
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()


##############################################################################################################################################


dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets.csv")
dataset_.index=dataset_.iloc[:,0]
dataset_=dataset_.iloc[:,1:]
dataset_=dataset_[[ 'mean_PM','mean_harg_','mean_modif_droog', 'mean_livia_', 'mean_Berti','mean_traj_balk']]

# Making a plot 
plt.boxplot(dataset_,labels=['FAO P-M','Hargr.','ET0,Modif2','M1','M2','M3']) 


#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
#plt.xlabel("methods")
plt.ylabel("ET0 [mm*day-1]")
plt.xticks(rotation=90)
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()


######################################################################################################################################################################################################
######################
######################################################################################################################################################################################################

dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets.csv")
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
ax1.plot(x, m*x+b,color='dimgrey')
ax1.set_xlabel("ET0 [mm*deň-1] / Hargreaves ET0")
ax1.set_ylabel("ET0 [mm*deň-1] / P-M FAO56 ET0")
ax1.legend(loc='lower right')


y1 =dataset_['mean_PM']  
x1 = dataset_['mean_modif_droog']
r_squared = np.corrcoef(x1,y1)[0,1]
ax2.scatter(x1,y1,label='Modif. ET0',s=s,color='black')
m, b = np.polyfit(x1, y1, 1)
ax2.plot(x1, m*x1+b,color='dimgrey')
ax2.set_xlabel("ET0 [mm*day-1] / Modif. method ET0")
ax2.set_ylabel("ET0 [mm*day-1] / P-M FAO56 ET0")
ax2.legend(loc='lower right')

fig.tight_layout(pad=2.0)
#fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//R_hargreaves_modif.jpeg',dpi=300, bbox_inches='tight')



#############################################################################


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
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//hargreaves with altitude//comparison of hargreaves and modif.jpeg',dpi=300, bbox_inches='tight')


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
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//hargreaves with altitude//comparison of dodgersallen and modif.jpeg',dpi=300, bbox_inches='tight')


##############################################################################################################################################


dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//comparison_of_HG_modiffications.csv")
dataset_.index=dataset_.iloc[:,0]
dataset_=dataset_.iloc[:,1:]
dataset_=dataset_[[ 'WAPE_harg_','WAPE_modif_droog', 'WAPE_livia_', 'WAPE_Berti','WAPE_traj_balk']]

# Making a plot 
plt.boxplot(dataset_,labels=['Hargr.','Modif.','M1','M2','M3']) 


#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
#plt.xlabel("methods")
plt.ylabel("WAPE [%]")
plt.xticks(rotation=90)
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()


######################################################################################################################################################################################################
###################################################################################################

##############################################################################################################################################


dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets.csv")
dataset_.index=dataset_.iloc[:,0]
dataset_=dataset_.iloc[:,1:]
dataset_=dataset_[[ 'var_harg_','var_modif_droog', 'var_livia_', 'var_Berti','var_traj_balk']]

# Making a plot 
plt.boxplot(dataset_,labels=['Hargr.','Modif.','M1','M2','M3']) 


#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
#plt.xlabel("methods")
plt.ylabel("Variance ")
plt.xticks(rotation=90)
#fig.savefig('C://Users//Viera//Desktop//dizertacka_res//res_analyses//ML_alt2//FIGURES//figure_'+str(mod)+'_'+str(alt)+'.jpeg',dpi=fig.dpi, bbox_inches='tight')
plt.plot()


######################################################################################################################################################################################################
###################################################################################################


dataset_=pd.read_csv('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_parameters//relationship altitude ax.csv')
dataset_1=pd.read_csv('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_parameters//relationship altitude wapemodif.csv')
dataset_1=dataset_1.dropna()
dataset_=dataset_.dropna()
dataset_.modif=dataset_.modif.astype(float)

fig, (ax1, ax2) = plt.subplots(1, 2)
s=10
# Add scatterplot
y =np.array(dataset_['modif'])
x = dataset_['FAO']
r_squared = np.corrcoef(x,y)[0,1]
ax1.scatter(x,y,label='Hargreaves ET0 \n mean a = 0.224',s=s,color='black')
m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b,label="R="+str(round(r_squared,2)),color='dimgrey')
ax1.set_xlabel("ET0 [mm*day-1] / FAO56 P-M ET0")
ax1.set_ylabel("ET0 [mm*day-1] / Modif ET0")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=1)


y1 =dataset_['alt']
x1 = dataset_['ax']
r_squared = np.corrcoef(x1,y1)[0,1]
ax2.scatter(x1,y1,label='Modif. ET0 \n -0.0002*ET0_HG + 0.9766',s=s,color='black')
m, b = np.polyfit(x1, y1, 1)
ax2.plot(x1, m*x1+b,label="R="+str(round(r_squared,2)),color='dimgrey')
ax2.set_xlabel("Regression slope ") # "Regression slope "
ax2.set_ylabel("Altitude [m a.s.l.]")
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=1)

fig.tight_layout(pad=2.0)
#fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//R_hargreaves_modif.jpeg',dpi=300, bbox_inches='tight')




####################################################################################################################

####################################################################################################################

####################################################################################################################




dataset_=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets.csv")
dataset_.index=dataset_.iloc[:,0]
dataset_=dataset_.iloc[:,1:]
#dataset_=dataset_[[ 'mean_PM','mean_harg_','mean_modif_droog', 'mean_livia_', 'mean_Berti', 'mean_droogersAllen2','mean_traj_balk']]


dataset_M=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets_monthly.csv")
dataset_M.index=dataset_M.iloc[:,0]
dataset_M=dataset_M.iloc[:,1:]
#dataset_M=dataset_M[[ 'mean_PM','mean_harg_','mean_modif_droog', 'mean_livia_', 'mean_Berti', 'mean_droogersAllen2','mean_traj_balk']]


dataset_Y=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//statistics_of_HG_modiffications_datasets_yearly.csv")
dataset_Y.index=dataset_Y.iloc[:,0]
dataset_Y=dataset_Y.iloc[:,1:]
#dataset_Y=dataset_Y[[ 'mean_PM','mean_harg_','mean_modif_droog', 'mean_livia_', 'mean_Berti', 'mean_droogersAllen2','mean_traj_balk']]
plt.rcParams.update({'font.size':15})
fig, ax = plt.subplots(1,3,figsize=(15,5))
s=50
# Add scatterplot
y =dataset_['mean_PM']  
x = dataset_['mean_harg_']
r_squared = np.corrcoef(x,y)[0,1]
ax[0].scatter(x,y,marker=".",color='black',s=s)
m, b = np.polyfit(x, y, 1)
ax[0].plot(x, m*x+b,color='black',label="Regr.line Hargr.")

y1 =dataset_['mean_PM']  
x1 = dataset_['mean_modif_droog']
r_squared = np.corrcoef(x1,y1)[0,1]
ax[0].scatter(x1,y1,marker="x",color='dimgrey',s=s)
m, b = np.polyfit(x1, y1, 1)
ax[0].plot(x1, m*x1+b,color='dimgrey',label="Regr.line ET0,Modif.2")

ax[0].grid()
ax[0].legend(loc='upper left')
ax[0].set_xlabel("ET0 Hargr.;ET0,Modif.2  [mm*day⁻¹]")
ax[0].set_ylabel("ET0 PM [mm*day⁻¹]")



y2 =dataset_M['mean_PM']  
x2 = dataset_M['mean_harg_']
r_squared = np.corrcoef(x2,y2)[0,1]
ax[1].scatter(x2,y2,marker=".",color='black',s=s)
m, b = np.polyfit(x2, y2, 1)
ax[1].plot(x2, m*x2+b,color='black',label="Regr.line Hargr.")

y2a =dataset_M['mean_PM']  
x2a = dataset_M['mean_modif_droog']
r_squared = np.corrcoef(x2a,y2a)[0,1]
ax[1].scatter(x2a,y2a,marker="x",color='dimgrey',s=s)
m, b = np.polyfit(x2a, y2a, 1)
ax[1].plot(x2a, m*x2a+b,color='dimgrey',label="Regr.line ET0,Modif.2")

ax[1].grid()
#ax[1].legend(bbox_to_anchor=(1.05, 1))
ax[1].set_xlabel("ET0 Hargr.;ET0,Modif.2  [mm*month⁻¹]")
ax[1].set_ylabel("ET0 PM [mm*month⁻¹]")



y3 =dataset_Y['mean_PM']  
x3 = dataset_Y['mean_harg_']
r_squared = np.corrcoef(x3,y3)[0,1]
ax[2].scatter(x3,y3,marker=".",color='black',s=s)
m, b = np.polyfit(x3, y3, 1)
ax[2].plot(x3, m*x3+b,color='black',label="Regr.line Hargr.")

y3a =dataset_Y['mean_PM']  
x3a = dataset_Y['mean_modif_droog']
r_squared = np.corrcoef(x3a,y3a)[0,1]
ax[2].scatter(x3a,y3a,marker="x",color='dimgrey',s=s)
m, b = np.polyfit(x3a, y3a, 1)
ax[2].plot(x3a, m*x3a+b,color='dimgrey',label="Regr.line ET0,Modif.2")

ax[2].grid()
#ax[2].legend(bbox_to_anchor=(1.05, 1))
ax[2].set_xlabel("ET0 Hargr.;ET0,Modif.2  [mm*year⁻¹]")
ax[2].set_ylabel("ET0 PM [mm*year⁻¹]")
#fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//nove vystupy//scaterplot_modif_harg.jpeg',dpi=300, bbox_inches='tight')
#######################################################################################################

