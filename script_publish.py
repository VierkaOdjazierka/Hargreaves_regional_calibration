# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:07:01 2025

@author: Viera Rattayová, rattayovaviera@gmail.com 
This example demonstrates how to use the functions provided in the HG_func_RCI.py script to calculate reference evapotranspiration (ET₀) using various methods. The script includes steps for data loading, date series setup, and function calls:
    
-Set Working Directory
Define the directory where the HG_func_RCI.py file is located.

-Import Required Libraries
Use pandas and numpy for data handling and numerical operations.

-Read Meteorological Data
Load daily minimum and maximum temperature data from CSV files. Optionally, include daily precipitation data.
-Create or Import Time Series- choose how to define the date series:
    -From the DataFrame index
    -From a specific column
    -By column name
    -Or generate your own date range

-Set Station Parameters
Define the station’s altitude, latitude (degrees), and latitude (minutes).

-Calculate Extraterrestrial Radiation
Call ext_rad() to compute daily extraterrestrial radiation based on latitude and date.

-Calculate ET₀ using RCI Method
Use et0_RCI_select() to compute ET₀ from temperature and radiation data.

-Print the Result - Output the calculated ET₀ time series.
"""
##########################################################################################################################
import pandas as pd
import numpy as np
import os
import math 

# Define the directory where the HG_func_RCI.py file is located.
os.chdir('C:/Users/set/working/directory')

import HG_func_RCI

##########################################################################################################################
# Load daily minimum and maximum temperature data from CSV files
tmin=pd.read_csv("C://set//path//to//data//data_tmin.csv", sep=",",header=None)
tmin=np.array(tmin.iloc[:,num_of_columns]).astype(float)
tmax=pd.read_csv("C://set//path//to//data//data_tmax.csv", sep=",",header=None)
tmax=np.array(tmax.iloc[:,num_of_columns]).astype(float)


# Optionally, include daily precipitation data
precip=pd.read_csv("C://set//path//to//data//data_precip.csv", sep=",",header=None)
precip=np.array(precip.iloc[:,num_of_columns]).astype(float)


#select one option how to define the date series
date_ts=tmax.index #if dates are included in .csv file as index
date_ts=tmax.iloc[:,num_of_columns] #select each rows in defined column
date_ts=tmax['name_of_column'] #select column according name
date_ts=pd.date_range('1980-01-01', '2020-12-31', freq='D') #create own time series

##########################################################################################################################

altitude=   #set altitude of station
lat_deg=    #set degrees of latitude of station
lat_min=    #set minutes of latitude of station

##########################################################################################################################

ext_rad= FUNC_HARG.ext_rad(date_ts=date_ts, latit_deg=lat_deg, latit_min=lat_min)
et=FUNC_HARG.et0_RCI_select(date_ts=date_ts, ext_rad=ext_rad, tmax=tmax, tmin=tmin, altitude=altitude, precip=None)
print(et)