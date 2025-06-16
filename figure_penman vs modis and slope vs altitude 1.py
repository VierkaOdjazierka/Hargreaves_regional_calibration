# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:29:37 2024

@author: Viera
"""
import seaborn as sns
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
#pip install tensorflow

from sklearn.preprocessing import StandardScaler



import hydroeval as he
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
plt.boxplot(dataset_,labels=['ET0 PM','Hargr.','ET0,Modif2.','M1','M2','M3']) 


#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
#plt.xlabel("methods")
plt.ylabel("ET0 [mm*day⁻¹]")
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
fig.tight_layout(pad=2.5)
count, bins_count = np.histogram(dataset_daily['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Daily ET0 Hargr.",linestyle='dashdot',color='black')


count, bins_count = np.histogram(dataset_monthly['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Monthly ET0 Hargr.",linestyle='dotted',color='black')


count, bins_count = np.histogram(dataset_yearly['WAPE_harg_'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Yearly ET0 Hargr.",linestyle='dashed',color='black')

########## #modiff

count, bins_count = np.histogram(dataset_daily['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Daily ET0 Modif.2",linestyle='dashdot',color='red')


count, bins_count = np.histogram(dataset_monthly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Monthly ET0 Modif.2",linestyle='dotted',color='red')


count, bins_count = np.histogram(dataset_yearly['WAPE_modif_droog'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
axs.plot(bins_count[1:], cdf, label="Yearly ET0 Modif.2",linestyle='dashed',color='red')


plt.legend(bbox_to_anchor=(0.5, -0.4),loc='lower center',ncol=2) #, bbox_to_anchor=(-0.2, -0.2),
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
ax1.scatter(x,y,label='ET0 PM= β * ET,Modif1 + α' ,color='black')
m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b,label="R="+str(round(r_squared,2)),color='dimgrey')
ax1.set_xlabel("ET0 PM [mm*day⁻¹]")
ax1.set_ylabel("ET0 Modif.1 [mm*day⁻¹]")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=1)


y1 =dataset_['ax']
x1 = dataset_['alt']
r_squared = np.corrcoef(x1,y1)[0,1]
ax2.scatter(x1,y1,label='β=f(ET0 Modif.1, ET0 PM)',s=s,color='black')
m, b = np.polyfit(x1, y1, 1)
ax2.plot(x1, m*x1+b,label="R="+str(round(r_squared,2))+'\n β=-0.0002* Altitude + 0.9766',color='dimgrey')
ax2.set_xlabel("Altitude [m a.s.l.]") # "Regression slope "
ax2.set_ylabel("Regression slope (β)")
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

fig.tight_layout(pad=1.5)

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

plt.rcParams.update({'font.size':25})

data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//res_years_b.csv")

fig,ax = plt.subplots(figsize=(15,7))
ax.scatter(data_1.iloc[:,0],data_1.iloc[:,1],label='B coef.',s=s,color='red')

r_squared1 = np.round(np.corrcoef(data_1.iloc[1:21,0],data_1.iloc[1:21,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[1:21,0],data_1.iloc[1:21,1], 1)
ax.plot(data_1.iloc[1:21,0], m*data_1.iloc[1:21,0]+b,color='dimgrey',label='trendline 1981-2000 ... R='+str(r_squared1),linestyle='dashed',linewidth=4)

r_squared2 = np.round(np.corrcoef(data_1.iloc[21:,0],data_1.iloc[21:,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[21:,0],data_1.iloc[21:,1], 1)
ax.plot(data_1.iloc[21:,0], m*data_1.iloc[21:,0]+b,color='black',label='trendline 2001-2020 ... R='+str(r_squared2),linestyle='dotted',linewidth=4)

r_squared3 = np.round(np.corrcoef(data_1.iloc[1:,0],data_1.iloc[1:,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[1:,0],data_1.iloc[1:,1], 1)
ax.plot(data_1.iloc[1:,0], m*data_1.iloc[1:,0]+b,color='grey',label='trendline 1981-2020 ... R='+str(r_squared3),linewidth=4)
ax.set_xlabel("Year")
ax.set_ylabel("Coef. B value [-]")
ax.legend()



data_1=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//results_yearlyB//res_years_C.csv")

fig,ax = plt.subplots(figsize=(15,7))
ax.scatter(data_1.iloc[:,0],data_1.iloc[:,1],label='C coef.',s=s,color='red')

r_squared1 = np.round(np.corrcoef(data_1.iloc[1:21,0],data_1.iloc[1:21,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[1:21,0],data_1.iloc[1:21,1], 1)
ax.plot(data_1.iloc[1:21,0], m*data_1.iloc[1:21,0]+b,color='dimgrey',label='trendline 1981-2000 ... R='+str(r_squared1),linestyle='dashed',linewidth=4)

r_squared2 = np.round(np.corrcoef(data_1.iloc[21:,0],data_1.iloc[21:,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[21:,0],data_1.iloc[21:,1], 1)
ax.plot(data_1.iloc[21:,0], m*data_1.iloc[21:,0]+b,color='black',label='trendline 2001-2020 ... R='+str(r_squared2),linestyle='dotted',linewidth=4)

r_squared3 = np.round(np.corrcoef(data_1.iloc[1:,0],data_1.iloc[1:,1])[0,1],2)
m, b = np.polyfit(data_1.iloc[1:,0],data_1.iloc[1:,1], 1)
ax.plot(data_1.iloc[1:,0], m*data_1.iloc[1:,0]+b,color='grey',label='trendline 1981-2020 ... R='+str(r_squared3),linewidth=4)
ax.set_xlabel("Year")
ax.set_ylabel("Coef. C value [-]")
ax.legend()


################################################################################################################
#heatmap



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
data_1.index=round(pd.DataFrame(np.array([x * 0.005 for x in range(120)][52:102]).astype(float)),3).iloc[:,0]
data_1=data_1.iloc[:,1:]
#del data_1['altitude']
#del data_['altitude']
#########################

#########################################################################################
fig.tight_layout(pad=2)

# FIGURE
data_x=data_.iloc[::-1]
data_y=data_1.iloc[::-1]
plt.rcParams.update({'font.size':30})
fig,axes=plt.subplots(1,2,figsize=(22,10))

sns.heatmap(ax=axes[0],data=data_x,cmap='coolwarm',linecolor='black')#
axes[0].locator_params(tight=True, nbins=7)
axes[0].tick_params(axis='x', rotation=30)
axes[0].set(xlabel='Years',ylabel='Value of B coef. [-]')
axes[0].set_title('WAPE [%]')

sns.heatmap(ax=axes[1],data=data_y,cmap='coolwarm',linecolor='black')#
axes[1].locator_params(tight=True, nbins=7)
axes[1].tick_params(axis='x', rotation=30)
axes[1].set(xlabel='Years',ylabel='Value of C coef. [-]')
axes[1].set_title('WAPE [%]')

