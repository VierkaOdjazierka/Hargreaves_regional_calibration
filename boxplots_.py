# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:10:07 2024

@author: Viera
"""
import xarray as xr
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import math 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#pip install tensorflow

from sklearn.preprocessing import StandardScaler



import hydroeval as he
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

data_input=pd.read_csv("C://Users//Viera//Desktop//data_klima_hydro//kli_export_moda.dat", sep=";",header=None,names=['IndKl','Datum','T_MAX_1','T_MIN_1','T_DEN_PR','VIE_RYCH_PR','OBL_PR','TLAK_VOD_PAR_PR','SLN_SVIT_1','ZRA_UHRN_1'],error_bad_lines=False)#error_bad_lines=False,warn_bad_lines=False
data_input.columns=data_input.iloc[0,:]
data_input=data_input.iloc[3:,:]
data_input.index=data_input.iloc[:,1]
data_input.index=pd.to_datetime(data_input.index, format="%d.%m.%Y")
data_input['IndKl']=data_input['IndKl'].astype(str)
del data_input['Datum']
data_=np.where(data_input=='-',np.nan,data_input)
data_input.iloc[:,:]=data_

stat_info=pd.read_csv("C://Users//Viera//Desktop//projekty-clanky//stat_info.csv", sep=",",header=None,error_bad_lines=False)
stat_info.columns=stat_info.iloc[0,:]
stat_info.drop([0],inplace=True)
stat_info=stat_info.iloc[:,1:]
stat_info.columns=['Indikatív', 'altitude' ,'latit_deg' ,'latit_min']
print(stat_info)



#load shapefile
shapefile = gpd.read_file('C://Users//Viera//Desktop//projekty-clanky//EWRA//klima_stations.shp')  #NASTAV 
# stations=shapefile["ID"]
# stations.to_csv('C://Users//ratta//OneDrive//Desktop//projekty-clanky//Net_rad//stations.csv')


stations=stat_info['Indikatív']


et0_pm_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))
et0_harg_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))
et0_droogers_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))
et0_berti_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))
et0_modif_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))
et0_livia_res=pd.DataFrame(np.zeros((len(pd.date_range('1980-01-01','2020-12-31')))))

et0_pm_res.index=pd.date_range('1980-01-01','2020-12-31')
et0_harg_res.index=pd.date_range('1980-01-01','2020-12-31')
et0_droogers_res.index=pd.date_range('1980-01-01','2020-12-31')
et0_berti_res.index=pd.date_range('1980-01-01','2020-12-31')
et0_modif_res.index=pd.date_range('1980-01-01','2020-12-31')
et0_livia_res.index=pd.date_range('1980-01-01','2020-12-31')


stat=11858
#will be loop trought stations
for stat in stations:
    try:
        et0=pd.read_csv('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//vysledky_stanice//et0_rozne_metody'+str(stat)+'.csv',sep=",")
        et0.index=et0.iloc[:,0]
        et0.index=pd.to_datetime(et0.index, format="%Y-%m-%d")
        et0=et0.iloc[:,1:]
        et0=et0.dropna()
        print(et0.columns)
        
        et0_pm=pd.DataFrame(et0['et0'])
        et0_harg=pd.DataFrame(et0['et0_harg_orig'])
        et0_droogers=pd.DataFrame(et0['et0_droogersAllen2'])
        et0_berti=pd.DataFrame(et0['et0_Berti'])
        et0_modif=pd.DataFrame(et0['et0_modif_droog'])
        et0_livia=pd.DataFrame(et0['et0_liv'])
        
        et0_pm_res=pd.concat([et0_pm_res,et0_pm], axis = 1)
        et0_harg_res=pd.concat([et0_harg_res,et0_harg], axis = 1)
        et0_droogers_res=pd.concat([et0_droogers_res,et0_droogers], axis = 1)
        et0_berti_res=pd.concat([et0_berti_res,et0_berti], axis = 1)
        et0_modif_res=pd.concat([et0_modif_res,et0_modif], axis = 1)
        et0_livia_res=pd.concat([et0_livia_res,et0_livia], axis = 1)
        
    except:
        pass
        



et0_pm_res['month']=et0_pm_res.index
et0_harg_res['month']=et0_harg_res.index
et0_droogers_res['month']=et0_droogers_res.index
et0_berti_res['month']=et0_berti_res.index
et0_modif_res['month']=et0_modif_res.index
et0_livia_res['month']=et0_livia_res .index


et0_pm_res['month']=et0_pm_res['month'].dt.month
et0_harg_res['month']=et0_harg_res['month'].dt.month
et0_droogers_res['month']=et0_droogers_res['month'].dt.month
et0_berti_res['month']=et0_berti_res['month'].dt.month
et0_modif_res['month']=et0_modif_res['month'].dt.month
et0_livia_res['month']=et0_livia_res['month'].dt.month   


et0_pm_res1_mean_m=pd.DataFrame(np.zeros((100)))
et0_harg_res1_mean_m=pd.DataFrame(np.zeros((100)))
et0_droogers_res1_mean_m=pd.DataFrame(np.zeros((100)))
et0_berti_res1_mean_m=pd.DataFrame(np.zeros((100)))
et0_modif_res1_mean_m=pd.DataFrame(np.zeros((100)))
et0_livia_res1_mean_m=pd.DataFrame(np.zeros((100)))



et0_pm_res=et0_pm_res.iloc[:,1:]
et0_harg_res=et0_harg_res.iloc[:,1:]
et0_droogers_res=et0_droogers_res.iloc[:,1:]
et0_berti_res=et0_berti_res.iloc[:,1:]
et0_modif_res=et0_modif_res.iloc[:,1:]
et0_livia_res=et0_livia_res.iloc[:,1:]  

mon=1
results_mean=np.zeros((6,12))
for mon in range(1,13):
    et0_pm_res1=et0_pm_res[et0_pm_res['month']==mon]
    et0_harg_res1=et0_harg_res[et0_harg_res['month']==mon]
    et0_droogers_res1=et0_droogers_res[et0_droogers_res['month']==mon]
    et0_berti_res1=et0_berti_res[et0_berti_res['month']==mon]
    et0_modif_res1=et0_modif_res[et0_modif_res['month']==mon]
    et0_livia_res1=et0_livia_res[et0_livia_res['month']==mon]
    
    results_mean[0,mon-1]=np.nanmean(et0_pm_res1)
    results_mean[1,mon-1]=np.nanmean(et0_harg_res1)
    results_mean[2,mon-1]=np.nanmean(et0_droogers_res1)
    results_mean[3,mon-1]=np.nanmean(et0_berti_res1)
    results_mean[4,mon-1]=np.nanmean(et0_modif_res1)
    results_mean[5,mon-1]=np.nanmean(et0_livia_res1)
    
    
    
    # et0_pm_res1=et0_pm_res['mean']
    # et0_harg_res1=et0_harg_res['mean']
    # et0_droogers_res1=et0_droogers_res['mean']
    # et0_berti_res1=et0_berti_res['mean']
    # et0_modif_res1=et0_modif_res['mean']
    # et0_livia_res1=et0_livia_res['mean']
    
    del et0_pm_res1['month']
    del et0_harg_res1['month']
    del et0_droogers_res1['month']
    del et0_berti_res1['month']
    del et0_modif_res1['month']
    del et0_livia_res1['month']
    
    et0_pm_res1_mean=pd.DataFrame(et0_pm_res1.mean(axis=0))
    et0_harg_res1_mean=pd.DataFrame(et0_harg_res1.mean(axis=0))
    et0_droogers_res1_mean=pd.DataFrame(et0_droogers_res1.mean(axis=0))
    et0_berti_res1_mean=pd.DataFrame(et0_berti_res1.mean(axis=0))
    et0_modif_res1_mean=pd.DataFrame(et0_modif_res1.mean(axis=0))
    et0_livia_res1_mean=pd.DataFrame(et0_livia_res1.mean(axis=0)) 
    
    et0_pm_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    et0_harg_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    et0_droogers_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    et0_berti_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    et0_modif_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    et0_livia_res1_mean.index=np.array(range(len(et0_pm_res1_mean)))
    
    
    et0_pm_res1_mean_m=pd.concat([et0_pm_res1_mean_m,et0_pm_res1_mean], axis = 1)
    et0_harg_res1_mean_m=pd.concat([et0_harg_res1_mean_m,et0_harg_res1_mean], axis = 1)
    et0_droogers_res1_mean_m=pd.concat([et0_droogers_res1_mean_m,et0_droogers_res1_mean], axis = 1)
    et0_berti_res1_mean_m=pd.concat([et0_berti_res1_mean_m,et0_berti_res1_mean], axis = 1)
    et0_modif_res1_mean_m=pd.concat([et0_modif_res1_mean_m,et0_modif_res1_mean], axis = 1)
    et0_livia_res1_mean_m=pd.concat([et0_livia_res1_mean_m,et0_livia_res1_mean], axis = 1)
    
import matplotlib.pyplot as plt
    
et0_pm_res1_mean_m=et0_pm_res1_mean_m.iloc[:,1:]
et0_harg_res1_mean_m=et0_harg_res1_mean_m.iloc[:,1:]
et0_droogers_res1_mean_m=et0_droogers_res1_mean_m.iloc[:,1:]
et0_berti_res1_mean_m=et0_berti_res1_mean_m.iloc[:,1:]
et0_modif_res1_mean_m=et0_modif_res1_mean_m.iloc[:,1:]
et0_livia_res1_mean_m=et0_livia_res1_mean_m.iloc[:,1:]
    

et0_pm_res1_mean_m=et0_pm_res1_mean_m.dropna(how='all')
et0_harg_res1_mean_m=et0_harg_res1_mean_m.dropna(how='all')
et0_droogers_res1_mean_m=et0_droogers_res1_mean_m.dropna(how='all')
et0_berti_res1_mean_m=et0_berti_res1_mean_m.dropna(how='all')
et0_modif_res1_mean_m=et0_modif_res1_mean_m.dropna(how='all')
et0_livia_res1_mean_m=et0_livia_res1_mean_m.dropna(how='all')  
    
et0_pm_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']
et0_harg_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']
et0_droogers_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']
et0_berti_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']
et0_modif_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']
et0_livia_res1_mean_m.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']  

results_mean=pd.DataFrame(results_mean)
results_mean.columns=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec']



# Making a plot 
plt.rc('grid', linestyle='dotted', color='grey')
fig, ax = plt.subplots()
#plt.lineplot(x) 
medianprops=dict(linestyle='-',linewidth=0)
ax.boxplot(et0_modif_res1_mean_m,labels=['jan.','feb.','march','apr.','may','jun','july','aug.','sep.','oct.','nov','dec'],medianprops=medianprops) 
locs=ax.get_xticks()
ax.plot(locs,results_mean.iloc[0,:],label='P-M', color='black')
locs=ax.get_xticks()
ax.plot(locs,results_mean.iloc[1,:],label='HG',linestyle='dashed', color='black')
#ax.plot(locs,results_mean.iloc[2,:],label='DG', color='grey')
#ax.plot(locs,results_mean.iloc[5,:],label='DG-P',linestyle='dashdot', color='grey')
ax.plot(locs,results_mean.iloc[4,:],label='Modif.',linestyle='dashed', color='red')

ax.grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=3) #, bbox_to_anchor=(-0.2, -0.2),
#plt.xlabel("methods")
plt.ylabel("ET0 [mm*day-1]")
plt.xticks(rotation=90)
fig.savefig('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//hargreaves with altitude//boxplots with mean lineplots.jpeg',dpi=300, bbox_inches='tight')
plt.plot()
    
