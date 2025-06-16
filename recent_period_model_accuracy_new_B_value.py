# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 22:38:43 2023

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
from sklearn import linear_model
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

#pip install tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import hydroeval as he
from scipy.stats import sem
import statistics
import math

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense




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
shapefile = gpd.read_file("C://Users//Viera//Desktop//dizertacka_res//vstupne_vrstvy//shapefile_all_stations_finish.shp")  #NASTAV 
shapefile.index=shapefile["ID"]

# stations=shapefile["ID"]
# stations.to_csv('C://Users//ratta//OneDrive//Desktop//projekty-clanky//Net_rad//stations.csv')


stations=stat_info['Indikatív']
#preparing dataframes for results


uplnost=pd.DataFrame(stations)
uplnost.index=uplnost.iloc[:,0]
uplnost.index=uplnost.index.astype(str)


leng=len(pd.date_range('1980-01-01','2020-12-31'))
stat=11800
for stat in stations:
    stanica=str(stat)
    data= data_input.loc[(data_input['IndKl'] ==stanica)]
    data=data['1980-01-01':'2020-12-31']
    data=data['SLN_SVIT_1']
    dataa=np.array(data)
    dataa=np.where(dataa=='-',np.nan, dataa)
    dataa=pd.DataFrame(dataa)
    dataa.index=data.index
    data=dataa.dropna()
    data_per=(len(data)/(leng/100))
    uplnost[uplnost.iloc[:,0]==stat]=data_per

uplnost=uplnost[uplnost.iloc[:,0]>50]

stations=np.array(uplnost.index).astype(int)


datumy=pd.date_range('2001-01-01', '2020-12-31', freq='D')
datumy_M=pd.date_range('2001-01-01', '2020-12-31', freq='M')

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    
res_bias=pd.DataFrame(np.zeros((4,1))).T
res_cc=pd.DataFrame(np.zeros((4,1))).T
    

# merra=xr.open_mfdataset('D://NET_RADIATION//merra//*.nc')
# merra=merra['__xarray_dataarray_variable__']
# print(merra)

datum_merra=pd.date_range('2001-01-01', '2020-12-31', freq='D')
bicor_res=pd.DataFrame(np.zeros((1,7)))
    
    
    ###############################################    ###############################################    ###############################################
    ###############################################        
    ###############################################    ###############################################    ###############################################
results_s=pd.DataFrame(np.zeros(27)).T
results_c=pd.DataFrame(np.zeros(56)).T

stat=stations[2]
stat=11858
#will be loop trought stations
for stat in stations:
    
        
    result=pd.DataFrame(datumy)
    result.index=result.iloc[:,0]


    # refevapo_clara=pd.DataFrame(datumy)
    # refevapo_clara=refevapo_clara.iloc[:,0]
    
    
    stanica=str(stat)
    data= data_input.loc[(data_input['IndKl'] ==stanica)]
    data=data.loc['2001-01-01':'2020-12-31']

    #dates
    try:
        fd=data.index[0]
        ld=data.index[-1]
    except:
        data= data_input.loc[(data_input['IndKl'] ==stat)]
        fd=data.index[0]
        ld=data.index[-1]
        

    #cleaning inputs
    datum_i=pd.date_range(fd, ld)
    data= data.reindex(datum_i, fill_value=np.nan)
    data.index=pd.to_datetime(data.index, format="%d.%m.%Y")
    data["SLN_SVIT_1"]=np.where(data["SLN_SVIT_1"]=='-', np.nan,data["SLN_SVIT_1"] )
    


    #select input variables from dataframe

    tmax1=data.iloc[:,1]
    tmax1=np.asarray(tmax1)
    tmax1=np.where(tmax1=='-',np.nan,tmax1)
    tmax1=tmax1.astype(float)
    tmax=tmax1+273.16 #Tmax, K maximum absolute temperature during the 24-hour period [K = °C + 273.16],

    tmin1=data.iloc[:,2]
    tmin1=np.asarray(tmin1)
    tmin1=np.where(tmin1=='-',np.nan,tmin1)
    tmin1=tmin1.astype(float)
    tmin=tmin1+273.16 #Tmin, K minimum absolute temperature during the 24-hour period [K = °C + 273.16],

    ##actual vapour pressure [kPa],
    ea=data.iloc[:,6]
    ea=np.asarray(ea)
    ea=np.where(ea==0,np.nan,ea)
    ea=np.where(ea=='-',np.nan,ea)
    ea=ea.astype(float)
    ea=ea*0.1


    tmean=data.iloc[:,3]
    tmean=np.asarray(tmean)
    tmean=np.where(tmean=='-',np.nan,tmean)
    tmean=tmean.astype(float)


    wind=data.iloc[:,4]
    wind=np.asarray(wind)
    wind=np.where(wind=='-',np.nan,wind)
    wind=wind.astype(float)

    wind=wind*(4.87/np.log(67.8*6-5.42))

    svit=data.iloc[:,7]
    svit=np.asarray(svit)
    svit=np.where(svit==0,np.nan,svit)
    svit=np.where(svit=='-',np.nan,svit)
    svit=svit.astype(float)
    
    prec=data.iloc[:,8]
    prec=np.asarray(prec)
    prec=np.where(prec=='-',np.nan,prec)
    prec=prec.astype(float)
    prec[np.isnan(prec)] = 0


    #set elevations and longitude of station (from file )

    info= stat_info.loc[(stat_info['Indikatív'] ==stanica)]
    nad_vyska_st=np.array(info['altitude']) # z station elevation above sea level [m]
    nad_vyska_st=nad_vyska_st.astype(float)

    stupne=np.array(info['latit_deg'])   # according the station longitude
    stupne=stupne.astype(float)


    minuty=np.array(info['latit_min'])   # according the station longitude
    minuty=minuty.astype(float)
    minuty=minuty/60


    j=datum_i
    j = j.strftime('%j')
    j=j.astype(int)
    j=np.array(j) #where J is the number of the day in the year between 1 (1 January) and 365 or 366 (31 December).

    #######################################################################


    #########################################################################################################
    #########################################################################################################

                                                  #CALCULATION
                                                  
    #########################################################################################################


    # Calculation of solar radiation
    # Calculation of solar radiation
    pi=3.14
    solconst=0.0820 #solar constant = 0.0820 MJ m-2 min-1,
    ass=0.25
    bs=0.5
    cp=1.013*10**-3 #specific heat at constant pressure, 1.013 10-3 [MJ kg-1 °C-1]
    lh=2.45#latent heat of vaporization, 2.45 [MJ kg-1],
    e=0.622#ratio molecular weight of water vapour/dry air
    #j=365 #where J is the number of the day in the year between 1 (1 January) and 365 or 366 (31 December).

    stupne=stupne+minuty


    delta=np.zeros(len(j))
    for i in range(len(j)):
        delta1=0.409*math.sin((2*pi*j[i]/365)-1.39)
        delta[i]=delta1
        
    dr=np.zeros(len(j))
    for i in range(len(j)):
        dr1=1+0.033*math.cos(((2*pi*j[i])/365)) 
        dr[i]=dr1


    fi=(pi/180)*(stupne)   #20°S or j = (p /180) (-20) = (the value is negative for the southern hemisphere)

    omegas=np.zeros(len(delta))
    for i in range(len(delta)):
        omegas1=np.arccos(-math.tan(fi)*math.tan(delta[i]))  #sunset hour angle
        omegas[i]=omegas1
        
    dthours_N=np.zeros(len(omegas))
    for i in range(len(omegas)):
        dthours_N1=(24/pi)*omegas[i]   #The daylight hours, N
        dthours_N[i]=dthours_N1

    extraterrad=np.zeros(len(omegas))
    for i in range(len(omegas)):
        extraterrad1=((24*60)/pi)*solconst*dr[i]*(omegas[i]*math.sin(fi)*math.sin(delta[i])+math.cos(fi)*math.cos(delta[i])*math.sin(omegas[i]))#Extraterrestrial radiation
        extraterrad[i]=extraterrad1
        
    
    rs=(ass+bs*(svit/dthours_N))*extraterrad #solar radiation, Rs, is not measured, it can be calculated with the Angstrom formula which relates solar radiation to extraterrestrial radiation and relative sunshine duration


    #Calculation of Net solar or net shortwave radiation (Rns)

    a=0.23 #a albedo or canopy reflection coefficient, which is 0.23 for the hypothetical grass reference crop [dimensionless],
    rns = (1-a)*rs 

    #longwave radiation
    # rs0=(0.75+(2*10**(-5)*nad_vyska_st)*extraterrad) #!!!toto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #rs0=(ass+bs)*extraterrad #Where no actual solar radiation data are available and no calibration has been carried out for improved as and bs parameters, the values as = 0.25 and bs = 0.50 are recommended.
    rs0=(0.75+0.00002*nad_vyska_st)*extraterrad
    ro=4.903*10**(-9) # Stefan-Boltzmann constant [4.903 10-9 MJ K-4 m-2 day-1],

    leng=len(ea)
    ae1=np.zeros(leng)
    dl=range(leng)
    for i in dl:
        res=math.sqrt(ea[i])
        ae1[i]=res

    rnl=(ro*((tmax**4+tmin**4)/2)*(0.34-0.14*ae1)*(1.35*rs/rs0-0.35))


    #Net radiation (Rn)
    #The net radiation (Rn) is the difference between the incoming net shortwave radiation (Rns) and the outgoing net longwave radiation (Rnl):

    net_radiation=rns-rnl   
    radiation=net_radiation
    
    
    sos=(40989*(0.6108*np.exp((17.27*tmean)/(tmean+237.3))))/((tmean+237.3)**2)
    e0max=0.6108*np.exp((17.27*tmax1)/(tmax1+237.3))    #saturation vapour pressure at the air temperature T [kPa],
    e0min=0.6108*np.exp((17.27*tmin1)/(tmin1+237.3))
    es=(e0max+e0min)/2# mean saturation vapour pressure
    p=101.3*((293-0.0065*nad_vyska_st)/293)**5.26#atmospheric pressure [kPa],
    psi=(cp*p)/(e*lh)#Psychrometric constant
    g=0 #As the magnitude of the day or ten-day soil heat flux beneath the grass reference surface is relatively small, it may be ignored and thus:

    et0=((0.408*sos*(radiation-g))+psi*((900/(tmean+273))*wind*(es-ea)))/(sos+psi*(1+0.34*wind))
       
    nad_vyska_st=np.array(nad_vyska_st[0])
    # skus    ET₀ = 0.0023 * Ra * (Tmax - Tmin)^0.5 * (Tmean + 17.8 + 0.004 * Elev)
    #moj najpresnejsi vyorec
    #et0_modif=0.408*0.0030*(((tmax1-tmin1)/2)+19.97)*((tmax1-tmin1)**0.39)*(extraterrad)
    
    et0_short=0.0023*(((tmax1+tmin1)/2)+17.8)*((tmax1-tmin1)**0.5)*(extraterrad)*0.408 #original hargreaves
    et0_har_Rs=0.013*(rs*0.408)*(((tmax1+tmin1)/2)+17.8)
    et0_liv=0.0013*0.408*extraterrad*(((tmax1+tmin1)/2)+17.0)*((tmax1-tmin1)-0.0123*prec)**0.76 #(Estimating Reference Evapotranspiration Under Inaccurate Data Conditions)
    #et0_modif=0.408*0.0030*(((tmax1-tmin1)/2)+19.97)*((tmax1-tmin1)**0.39)*(extraterrad)
    #et0_modif=0.0023*0.408*((tmax1-tmin1)**0.5)*(((tmax1-tmin1)/2)+17.8+0.004*nad_vyska_st)*(extraterrad) #veruzia z chatgpt
    et0_modif=0.0018*0.408*((tmax1-tmin1)**0.5)*(((tmax1+tmin1)/2)+16.2+nad_vyska_st/100)*(extraterrad)
    et0_droogersAllen1=0.00193*0.408*(((tmax1+tmin1)/2)+17.8)*((tmax1-tmin1)**0.517)*(extraterrad) # Berti et al. (2014): Modified Hargreaves   Berti, A., Tardivo, G., Chiaudani, A., Rech, F., Borin, M., 2014. Assessing reference evapotranspiration by the Hargreaves methodin north-eastern Italy. Agric. Water Manage. 140, 20–25, http://dx.doi.org/10.1016/j.agwat.2014.03.015
    et0_droogersAllen2=0.408*0.003*(((tmax1+tmin1)/2)+20)*((tmax1-tmin1)**0.4)*(extraterrad) #(Estimating reference evapotranspiration under inaccurate data conditions)
    et0_traj_balk=0.408*0.0023*(((tmax1+tmin1)/2)+17.8)*((tmax1-tmin1)**0.424)* (extraterrad)#method abjusted by Trajkovic(2007) for Balkan Trajkovic, S., 2007. Hargreaves versus Penman–Monteith under humid conditions. J. Irrig. Drain. Eng. 133 (1), 38–42
    et0_traj_pol=(0.817*0.00022*nad_vyska_st)*0.0023*(((tmax1+tmin1)/2)+17.8)*((tmax1-tmin1)**0.5)* (extraterrad)  #Ravazzani et al. (2012) method Ravazzani, G., Corbari, C., Morella, S., Gianoli, P., Mancini, M., 2012. Modified Hargreaves-Samani equation for theassessment of reference evapotranspiration in Alpine River Basins. J. Irrig. Drain. Eng. ASCE 138 (7), 592–599,http://dx.doi.org/10.1061/(ASCE)IR.1943-4774.0000453.
    #et0_modif_droog=0.408*0.0031*(((tmax1-tmin1)/2)+19.97)*((tmax1-tmin1)**0.395)*(extraterrad)
    et0_modif_droog=0.408*0.0029*(((tmax1+tmin1)/2)+19.7)*((tmax1-tmin1)**0.39)*(extraterrad)*(0.00014 *nad_vyska_st+0.97)

    #    et0_modif_droog=0.408*0.0029*(((tmax1+tmin1)/2)+19.7)*((tmax1-tmin1)**0.39)*(extraterrad)*(0.00014 *nad_vyska_st+0.97)-0.02
    
    
    bias_1=np.nanmean(et0_short-et0)
    bias_2=np.nanmean(et0_har_Rs-et0)
    bias_3=np.nanmean(et0_liv-et0)
    bias_4=np.nanmean(et0_modif-et0)
    bias_44=np.nanmean(et0_modif_droog-et0)
    bias_5=np.nanmean(et0_droogersAllen1-et0)
    bias_6=np.nanmean(et0_droogersAllen2-et0) 
    bias_7=np.nanmean(et0_traj_balk-et0)
    bias_8=np.nanmean(et0_traj_pol-et0)
    
    bias_perc_1=np.nanmean((et0_short-et0)/et0)*100
    bias_perc_2=np.nanmean((et0_har_Rs-et0)/et0)*100
    bias_perc_3=np.nanmean((et0_liv-et0)/et0)*100
    bias_perc_4=np.nanmean((et0_modif-et0)/et0)*100
    bias_perc_44=np.nanmean((et0_modif_droog-et0)/et0)*100
    bias_perc_5=np.nanmean((et0_droogersAllen1-et0)/et0)*100
    bias_perc_6=np.nanmean((et0_droogersAllen2-et0)/et0)*100  
    bias_perc_7=np.nanmean((et0_traj_balk-et0)/et0)*100
    bias_perc_8=np.nanmean((et0_traj_pol-et0)/et0)*100 
    
    mape_1=np.nanmean(np.abs(et0-et0_short)/np.nanmean(et0))*100
    mape_2=np.nanmean(np.abs(et0-et0_har_Rs)/np.nanmean(et0))*100
    mape_3=np.nanmean(np.abs(et0-et0_liv)/np.nanmean(et0))*100
    mape_4=np.nanmean(np.abs(et0-et0_modif)/np.nanmean(et0))*100
    mape_44=np.nanmean(np.abs(et0-et0_modif_droog)/np.nanmean(et0))*100
    mape_5=np.nanmean(np.abs(et0-et0_droogersAllen1)/np.nanmean(et0))*100
    mape_6=np.nanmean(np.abs(et0-et0_droogersAllen2)/np.nanmean(et0))*100
    mape_7=np.nanmean(np.abs(et0-et0_traj_balk)/np.nanmean(et0))*100
    mape_8=np.nanmean(np.abs(et0-et0_traj_pol)/np.nanmean(et0))*100
    
    
    mae_1=np.nanmean(abs(np.subtract(et0,et0_short)))
    mae_2=np.nanmean(abs(np.subtract(et0,et0_har_Rs)))
    mae_3=np.nanmean(abs(np.subtract(et0,et0_liv)))
    mae_4=np.nanmean(abs(np.subtract(et0,et0_modif)))
    mae_44=np.nanmean(abs(np.subtract(et0,et0_modif_droog)))
    mae_5=np.nanmean(abs(np.subtract(et0,et0_droogersAllen1)))
    mae_6=np.nanmean(abs(np.subtract(et0,et0_droogersAllen2)))
    mae_7=np.nanmean(abs(np.subtract(et0,et0_traj_balk)))
    mae_8=np.nanmean(abs(np.subtract(et0,et0_traj_pol)))
    
    kge_1= he.evaluator(he.kge, et0_short, et0)[0,0]
    kge_2= he.evaluator(he.kge, et0_har_Rs, et0)[0,0]
    kge_3= he.evaluator(he.kge, et0_liv, et0)[0,0]
    kge_4= he.evaluator(he.kge, et0_modif, et0)[0,0]
    kge_44= he.evaluator(he.kge, et0_modif_droog, et0)[0,0]
    kge_5= he.evaluator(he.kge, et0_droogersAllen1, et0)[0,0]
    kge_6= he.evaluator(he.kge, et0_droogersAllen2, et0)[0,0]
    kge_7= he.evaluator(he.kge, et0_traj_balk, et0)[0,0]
    kge_8= he.evaluator(he.kge, et0_traj_pol, et0)[0,0]
    
    wape_1=np.nanmean(abs(np.subtract(et0_short,et0)))/(np.nanmean(abs(et0))/100)
    wape_2=np.nanmean(abs(np.subtract(et0_har_Rs,et0)))/(np.nanmean(abs(et0))/100)
    wape_3=np.nanmean(abs(np.subtract(et0_liv,et0)))/(np.nanmean(abs(et0))/100)
    wape_4=np.nanmean(abs(np.subtract(et0_modif,et0)))/(np.nanmean(abs(et0))/100)
    wape_44=np.nanmean(abs(np.subtract(et0_modif_droog,et0)))/(np.nanmean(abs(et0))/100)
    wape_5=np.nanmean(abs(np.subtract(et0_droogersAllen1,et0)))/(np.nanmean(abs(et0))/100)
    wape_6=np.nanmean(abs(np.subtract(et0_droogersAllen2,et0)))/(np.nanmean(abs(et0))/100)
    wape_7=np.nanmean(abs(np.subtract(et0_traj_balk,et0)))/(np.nanmean(abs(et0))/100)
    wape_8=np.nanmean(abs(np.subtract(et0_traj_pol,et0)))/(np.nanmean(abs(et0))/100)
    
    #CC a rmse

    datasets=pd.DataFrame(et0)
    datasets['a']=et0_short
    datasets=datasets.dropna()
    cc_1=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_1 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['b']=et0_har_Rs
    datasets=datasets.dropna()
    cc_2=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_2 = math.sqrt(mse)
    
    
    datasets=pd.DataFrame(et0)
    datasets['c']=et0_liv
    datasets=datasets.dropna()
    cc_3=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_3 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['d']=et0_modif
    datasets=datasets.dropna()
    cc_4=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_4 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['dd']=et0_modif_droog
    datasets=datasets.dropna()
    cc_44=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_44 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['e']=et0_droogersAllen1
    datasets=datasets.dropna()
    cc_5=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_5 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['f']=et0_droogersAllen2
    datasets=datasets.dropna()
    cc_6=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_6 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['f']=et0_traj_balk
    datasets=datasets.dropna()
    cc_7=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_7 = math.sqrt(mse)
    
    datasets=pd.DataFrame(et0)
    datasets['f']=et0_traj_pol
    datasets=datasets.dropna()
    cc_8=np.corrcoef(datasets.iloc[:,0],datasets.iloc[:,1])[0,1]
    mse = np.square(np.subtract(datasets.iloc[:,0],datasets.iloc[:,1])).mean() 
    rmse_8 = math.sqrt(mse)
    
    
    

    
    datasets=pd.DataFrame(et0)
    datasets['a']=et0_short
    datasets['d']=et0_har_Rs
    datasets['b']=et0_liv
    datasets['c']=et0_modif
    datasets['cc']=et0_modif_droog
    datasets['e']=et0_droogersAllen1
    datasets['f']=et0_droogersAllen2
    datasets['g']=et0_traj_balk
    datasets['h']=et0_traj_pol
    datasets.columns=['et0','et0_harg_orig','et0_harg_Rs','et0_liv', 'et0_modif','et0_modif_droog','et0_Berti','et0_droogersAllen2','et0_et0_traj_balk','et0_et0_Ravazzani']
    datasets.index=data.index
    #datasets.to_csv('C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//vysledky_stanice//et0_rozne_metody'+str(stat)+'.csv')
    
    ####################################################################
    #statistics of datasets
    altitude=nad_vyska_st
    #Variance of sample set is %
    
    var_1=np.nanvar(et0)
    var_2=np.nanvar(et0_short)
    var_3=np.nanvar(et0_har_Rs)
    var_4=np.nanvar(et0_liv)
    var_5=np.nanvar(et0_modif)
    var_55=np.nanvar(et0_modif_droog)
    var_6=np.nanvar(et0_droogersAllen1)
    var_7=np.nanvar(et0_droogersAllen2)
    var_8=np.nanvar(et0_traj_balk)
    var_9=np.nanvar(et0_traj_pol)
    
    
    #Standard error
    
    se_1=sem(et0)
    se_2=sem(et0_short)
    se_3=sem(et0_har_Rs)
    se_4=sem(et0_liv)
    se_5=sem(et0_modif)
    se_55=sem(et0_modif_droog)
    se_6=sem(et0_droogersAllen1)
    se_7=sem(et0_droogersAllen2)
    se_8=sem(et0_traj_balk)
    se_9=sem(et0_traj_pol)
    
    
    #Mean
    
    mean_1=np.nanmean(et0)
    mean_2=np.nanmean(et0_short)
    mean_3=np.nanmean(et0_har_Rs)
    mean_4=np.nanmean(et0_liv)
    mean_5=np.nanmean(et0_modif)
    mean_55=np.nanmean(et0_modif_droog)
    mean_6=np.nanmean(et0_droogersAllen1)
    mean_7=np.nanmean(et0_droogersAllen2)
    mean_8=np.nanmean(et0_traj_balk)
    mean_9=np.nanmean(et0_traj_pol)
    
    #median
    median1=statistics.median(et0)
    median2=statistics.median(et0_short)
    median3=statistics.median(et0_har_Rs)
    median4=statistics.median(et0_liv)
    median5=statistics.median(et0_modif)
    median55=statistics.median(et0_modif_droog)
    median6=statistics.median(et0_droogersAllen1)
    median7=statistics.median(et0_droogersAllen2)
    median8=statistics.median(et0_traj_balk)
    median9=statistics.median(et0_traj_pol)
    

   
    
    results_compar=np.array([bias_1,bias_2,bias_3,bias_4,bias_44,bias_5,bias_6,bias_7,bias_8,bias_perc_1,bias_perc_2,bias_perc_3,bias_perc_4,bias_perc_44,bias_perc_5,bias_perc_6,bias_perc_7,bias_perc_8,mape_1,mape_2,mape_3,mape_4,mape_44,mape_5,mape_6,mape_7,mape_8,mae_1,mae_2,mae_3,mae_4,mae_44,mae_5,mae_6,mae_7,mae_8,kge_1,kge_2,kge_3,kge_4,kge_44,kge_5,kge_6,kge_7,kge_8,cc_1,cc_2,cc_3,cc_4,cc_44,cc_5,cc_6,cc_7,cc_8,rmse_1,rmse_2,rmse_3,rmse_4,rmse_44,rmse_5,rmse_6,rmse_7,rmse_8,wape_1,wape_2,wape_3,wape_4,wape_44,wape_5,wape_6,wape_7,wape_8])
    results_compar=pd.DataFrame(results_compar).T
    results_c=pd.concat([results_c,results_compar], axis = 0)  
    
    results_stat=np.array([var_1,var_2,var_3,var_4,var_5,var_55,var_6,var_7,var_8,var_9,se_1,se_2,se_3,se_4,se_5,se_55,se_6,se_7,se_8,se_9,mean_1,mean_2,mean_3,mean_4,mean_5,mean_55,mean_6,mean_7,mean_8,mean_9,median1,median2,median3,median4,median5,median55,median6,median7,median8,median9,altitude])
    results_stat=pd.DataFrame(results_stat).T
    results_s=pd.concat([results_s,results_stat], axis = 0)  
    
    
results_c=results_c.iloc[1:,:]
results_c.columns=['bias_harg_','bias_harg_Rs','bias_livia_','bias_modif_','bias_modif_droog','bias_Berti','bias_droogersAllen2','bias_traj_balk','bias_Ravazzani','bias_perc_harg_','bias_perc_harg_Rs','bias_perc_livia_','bias_perc_modif_','bias_perc_modif_droog','bias_perc_Berti','bias_perc_droogersAllen2','bias_perc_traj_balk','bias_perc_Ravazzani','MAPE_harg_','MAPE_harg_Rs','MAPE_livia_','MAPE_modif_','MAPE_modif_droog','MAPE_Berti','MAPE_droogersAllen2','MAPE_traj_balk','MAPE_Ravazzani','MAE_harg_','MAE_harg_Rs','MAE_livia_','MAE_modif_','MAE_modif_droog','MAE_Berti','MAE_droogersAllen2','MAE_traj_balk','MAE_Ravazzani','KGE_harg_','KGE_harg_Rs','KGE_livia_','KGE_modif_','KGE_modif_droog','KGE_Berti','KGE_droogersAllen2','KGE_traj_balk','KGE_Ravazzani','R_harg_','R_harg_Rs','R_livia_','R_modif_','R_modif_droog','R_Berti','R_droogersAllen2','R_traj_balk','R_Ravazzani','RMSE_harg_','RMSE_harg_Rs','RMSE_livia_','RMSE_modif_','RMSE_modif_droog','RMSE_Berti','RMSE_droogersAllen2','RMSE_traj_balk','RMSE_Ravazzani','WAPE_harg_','WAPE_harg_Rs','WAPE_livia_','WAPE_modif_','WAPE_modif_droog','WAPE_Berti','WAPE_droogersAllen2','WAPE_traj_balk','WAPE_Ravazzani']
results_c.index=stations

results_c.to_csv("C://Users//Viera//Desktop//projekty-clanky//clanok hargreaves//changes in period//change of accuracy in last period using general and new coef B//new_coef_results.csv")


   
results_s=results_s.iloc[1:,:]
results_s.columns=['var_PM','var_harg_','var_harg_Rs','var_livia_','var_modif_','var_modif_droog','var_Berti','var_droogersAllen2','var_traj_balk','var_Ravazzani','se_PM','se_harg_','se_harg_Rs','se_livia_','se_modif_','se_modif_droog','se_Berti','se_droogersAllen2','se_traj_balk','se_Ravazzani','mean_PM','mean_harg_','mean_harg_Rs','mean_livia_','mean_modif_','mean_modif_droog','mean_Berti','mean_droogersAllen2','mean_traj_balk','mean_Ravazzani','median_PM','median_harg_','median_harg_Rs','median_livia_','median_modif_','median_modif_droog','median_Berti','median_droogersAllen2','median_traj_balk','median_Ravazzani','nadm_vyska']
results_s.index=stations
