# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:30:20 2025

@author: Viera Rattayová
"""
import pandas as pd
import numpy as np
import math 

def parse_date(date_str, formats=None):
    formats = formats or ["","%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y" , "%d/%m/%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(date_str, errors='coerce')  # fallback



def ext_rad(date_ts, latit_deg, latit_min):
    date = parse_date(date_ts)
   
    fd=date[0]
    ld=date[-1]
    datum_i=pd.date_range(fd, ld)
    
    stupne=np.array(latit_deg).astype(float)   # according the station longitude
    minuty=(np.array(latit_min).astype(float))/60  # according the station longitude
    stupne=stupne+minuty


    j=datum_i
    j = j.strftime('%j')
    j=j.astype(int)
    j=np.array(j) #where J is the number of the day in the year between 1 (1 January) and 365 or 366 (31 December).


    pi=3.14
    solconst=0.0820 #solar constant = 0.0820 MJ m-2 min-1,
    #j=365 #where J is the number of the day in the year between 1 (1 January) and 365 or 366 (31 December).
    

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
        
    # Return as time series
    return pd.Series(data=extraterrad, index=datum_i, name="MJ/m2*day")




def et0_RCI_select(date_ts, ext_rad, tmax, tmin, precip=None, altitude=None, modif=None):
    """
    Calculate reference evapotranspiration (ET0) using a modified Hargreaves approach.

    Parameters:
    - date_ts: array-like or datetime index
    - ext_rad: extraterrestrial radiation [MJ/m2/day] -np.array
    - tmax, tmin: np.array with Tmax and Tmin [°C]
    - altitude: elevation of station [m]
    - modif: modification type string
    
    optional modification type strings:
    'orig_Harg'- Reference Crop Evapotranspiration from Temperature, George H. Hargreaves, Zohrab A. Samani, Applied Engineering in Agriculture. 1(2): 96-99. (doi: 10.13031/2013.26773) @1985
    'Droogers_2002'- Droogers, P., Allen, R.G. Estimating Reference Evapotranspiration Under Inaccurate Data Conditions. Irrigation and Drainage Systems 16, 33–45 (2002). https://doi.org/10.1023/A:1015508322413
    'Rattayova'- Rattayová, Viera, et al. "Regional calibration of the Hargreaves model for estimation of reference evapotranspiration" Journal of Hydrology and Hydromechanics, vol. 72, no. 4, Slovak Academy of Sciences, 2024, pp. 513-521. https://doi.org/10.2478/johh-2024-0023 DOI: https://doi.org/10.2478/johh-2024-0023
    'Berti'- Berti et al. (2014): Modified Hargreaves   Berti, A., Tardivo, G., Chiaudani, A., Rech, F., Borin, M., 2014. Assessing reference evapotranspiration by the Hargreaves methodin north-eastern Italy. Agric. Water Manage. 140, 20–25, http://dx.doi.org/10.1016/j.agwat.2014.03.015
    'Trajkovic'- Trajkovic(2007) for Western Balkans region-  Trajkovic, S., 2007. Hargreaves versus Penman–Monteith under humid conditions. J. Irrig. Drain. Eng. 133 (1), 38–42
    'Ravazzani'- Ravazzani, G., Corbari, C., Morella, S., Gianoli, P., Mancini, M., 2012. Modified Hargreaves-Samani equation for theassessment of reference evapotranspiration in Alpine River Basins. J. Irrig. Drain. Eng. ASCE 138 (7), 592–599,http://dx.doi.org/10.1061/(ASCE)IR.1943-4774.0000453.
    'Sepaskhah'- Sepaskhah, Ali & Razzaghi, Fatemeh. (2009). Evaluation of the adjusted Thornthwaite and Hargreaves-Samani methods for estimation of daily evapotranspiration in a semi-arid region of Iran. Archives of Agronomy and Soil Science. 55. 51-66. 10.1080/03650340802383148. 
    'Droogers_precip' -Droogers, P., Allen, R.G. Estimating Reference Evapotranspiration Under Inaccurate Data Conditions. Irrigation and Drainage Systems 16, 33–45 (2002). https://doi.org/10.1023/A:1015508322413
    
    """

    date = parse_date(date_ts)
    fd = date[0]
    ld = date[-1]
    datum_i = pd.date_range(fd, ld)


    tmean = (tmax + tmin) / 2
    delta_t = tmax - tmin
    

    # Calculate ET0 based on selected method
    if modif == 'orig_Harg':
        et0 = 0.0023 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad * 0.408

    elif modif == 'Droogers_2002':
        et0 = 0.408 * 0.0025 * (tmean + 16.8) * ((delta_t)**0.5) * ext_rad
        
    elif modif == 'Sepaskhah':
        et0 = 0.408 * 0.0026 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad
        
        
    elif modif == 'Droogers_precip':
        if precip is not None:
            et0 = 0.408 * 0.0013 * (tmean + 17.0) * ((delta_t - 0.0123 * precip) ** 0.76) * ext_rad
        else:
            print("ERROR: 'Droogers_precip' was skipped because precipitation data is missing.")
            et0 = np.full(len(tmin), np.nan)
        
    elif modif == 'Rattayova':
        et0 = 0.0029 * (tmean + 21.27) * ((delta_t)**0.39) * ext_rad * 0.408 * (0.00014 * altitude + 0.97)

    elif modif == 'Berti':
        et0 = 0.00193 * 0.408 * (tmean + 17.8) * ((delta_t)**0.517) * ext_rad

    elif modif == 'Trajkovic':
        et0 = 0.408 * 0.0023 * (tmean + 17.8) * ((delta_t)**0.424) * ext_rad

    elif modif == 'Ravazzani':
        et0 = (0.817 * 0.00022 * altitude) * 0.0023 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad* 0.408

    else:
        et0=pd.DataFrame()
        et0['orig_Harg']=0.0023 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad * 0.408
        et0['Droogers_2002']= 0.408 * 0.003 * (tmean + 20) * ((delta_t)**0.4) * ext_rad
        et0['Rattayova']= 0.0029 * (tmean + 21.27) * ((delta_t)**0.39) * ext_rad * 0.408 * (0.00014 * altitude + 0.97)
        et0['Berti']= 0.00193 * 0.408 * (tmean + 17.8) * ((delta_t)**0.517) * ext_rad
        et0['Trajkovic']= 0.408 * 0.0023 * (tmean + 17.8) * ((delta_t)**0.424) * ext_rad
        et0['Ravazzani']= (0.817 + 0.00022 * altitude) * 0.0023 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad
        et0['Sepaskhah']= 0.408 * 0.0026 * (tmean + 17.8) * ((delta_t)**0.5) * ext_rad
        if precip is not None:
            et0['Droogers_precip'] = 0.408 * 0.0013 * (tmean + 17.0) * ((delta_t - 0.0123 * precip) ** 0.76) * ext_rad
        else:
            print("Note: 'Droogers_precip' was skipped because precipitation data is missing.")
    
   
    
    return pd.DataFrame(data=et0, index=datum_i) if isinstance(et0, dict) or isinstance(et0, pd.DataFrame) else pd.DataFrame(data=et0, index=datum_i, columns=[modif])


