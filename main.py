# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 22:35:38 2025

@author: pdki009
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:59:13 2024

@author: pdki009
"""

import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import netCDF4
from windpowerlib import data as wt
from windpowerlib import ModelChain, WindTurbine
from pvlib import pvsystem, location, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS as PARAMS
import xgboost as xgb
import shap
import xarray as xr
from windpowerlib import power_output
from windpowerlib import power_curves
from windpowerlib import wake_losses
from statsmodels.tsa.stattools import acf
import os
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize

from google.colab import drive
drive.mount('/content/drive')

def coordFinder(lonv,latv,lons,lats):
    ll = len(lons)
    ii = np.zeros(ll)
    jj = np.zeros(ll)
    for i in range(ll):
        ii[i] = np.where(np.abs(lonv-lons[i]-.125)==np.min(np.abs(lonv-lons[i]-.125)))[0][0]
        jj[i] = np.where(np.abs(latv-lats[i]-.125)==np.min(np.abs(latv-lats[i]-.125)))[0][0]
    return ii,jj

def xgbCorrelatePredict(features,wind):
    #Initialize parameters for XGBoost runs
    param  = {'max_depth':3, 'eta':.3, 'objective':'reg:squarederror',
              'nthread':20, 'gamma':6, 'subsample':0.6, 'eval_metric':'mae'}
    num_round = 36


    k0 = np.where((wind>0)&(wind.index<wind.index[-8760]))[0]
    k1 = np.where((wind>0)&(wind.index>=wind.index[-8760]))[0]

    labels=wind.copy()

    dtrain = xgb.DMatrix(features.iloc[k0],
        label=labels.iloc[k0],missing=np.NaN)
    dtest = xgb.DMatrix(features.iloc[k1],
        label=labels.iloc[k1],missing=np.NaN)
    evallist = [(dtest,'eval'), (dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=True)

    return k0,k1,bst



#Initialize windpowerlib settings for calculation of wind generation from ERA5
#Winds

datadir = r'/content/drive/MyDrive/Colab Data/NextGenDemo'

curvefile = r'/PowerCurves.xlsx'
curves = pd.read_excel(datadir+curvefile)

#Define NEOMwt5 as EN182 5.0 MW Wind Turbine
spds = np.arange(0,25,0.1)
rho = 1.25*np.ones_like(spds)

pcurve = power_curves.create_power_curve(curves['Wind Speed (m/s)'],
        curves['EN182 5.0 MW'])
pcurve2 = power_output.power_curve(spds,pcurve['wind_speed'],pcurve['value'],
            density=1.25/1.05*1.25*np.ones_like(spds),density_correction=True)
pcurve = power_curves.create_power_curve(spds, pcurve2)
NEOMwt5 = WindTurbine(hub_height=120,nominal_power=5000.0,power_curve=pcurve,
                      rotor_diameter=182)

#Define NEOMwt8 as EN182 8.0 MW Wind Turbine
pcurve = power_curves.create_power_curve(curves['Wind Speed (m/s)'],
        curves['EN182 8.0 MW'])
pcurve2 = power_output.power_curve(spds,pcurve['wind_speed'],pcurve['value'],
            density=1.25/1.05*1.25*np.ones_like(spds),density_correction=True)
pcurve = power_curves.create_power_curve(spds, pcurve2)

NEOMwt8 = WindTurbine(hub_height=120,nominal_power=8000.0,power_curve=pcurve,
                      rotor_diameter=182)


#Define NEOMwt75 as GWH182-7.5 7.5 MW Wind Turbine
pcurve = power_curves.create_power_curve(curves['Wind Speed (m/s)'],
        curves['GWH182-7.5 MW'])
pcurve2 = power_output.power_curve(spds,pcurve['wind_speed'],pcurve['value'],
            density=1.25/1.134*1.25*np.ones_like(spds),density_correction=True)
pcurve = power_curves.create_power_curve(spds, pcurve2)
NEOMwt7p5 = WindTurbine(hub_height=120,nominal_power=7500.0,power_curve=pcurve,
                      rotor_diameter=182)


modelchain_data = {
    'wind_speed_model': 'logarithmic',  # 'logarithmic' (default),
    # 'hellman' or
    # 'interpolation_extrapolation'
    'density_model': 'ideal_gas',  # 'barometric' (default), 'ideal_gas'
    #  or 'interpolation_extrapolation'
    'temperature_model': 'linear_gradient',  # 'linear_gradient' (def.) or
    # 'interpolation_extrapolation'
    'power_output_model':
        'power_curve',  # 'power_curve' (default) or
    # 'power_coefficient_curve'
    'density_correction': True,  # False (default) or True
    'obstacle_height': 0,  # default: 0
    'hellman_exp': None}  # None (default) or None

# initialize ModelChain with own specifications and use run_model method to
# calculate power output

#Initialize pvlib settings for calculation of solar generation from location,
#GHI and temperature

# create site system characteristics

sandia_modules = pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = PARAMS['sapm']['open_rack_glass_glass']

array_kwargs = dict(
    module_parameter=module,
    temperature_model_parameters=temperature_model_parameters
)

inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

inverter.Paco = 176
inverter.Pdco = 190
inverter_parameters={'pdc0': 5000, 'eta_inv_nom': 0.96}

def windToPow(S100,T,Farms,WakeLoss):
    tl0 = S100.shape
    if len(tl0)==1:
        tl = tl0[0]
        ll = 1
        Pow = pd.DataFrame(np.zeros((tl,ll)),index=S100.index,columns=['Dummy'])
    else:
        tl = tl0[0]
        ll = tl0[1]
        Pow = pd.DataFrame(np.zeros((tl,ll)),index=S100.index,columns=S100.columns)
    for i,col in enumerate(Pow.columns):
        if Farms['Capacity'].iloc[i]>0:
            tl,ll = Pow.shape
            if ll>1:
                dum0 = S100[col].values[:,0]
                dum1 = T[col].values[:,0]
            elif ll==1:
                dum0 = S100.values[:,0]
                dum1 = T.values[:,0]
            else:
                dum0 = S100.values
                dum1 = T.values
            if WakeLoss:
                dum0 = wake_losses.reduce_wind_speed(pd.Series(dum0,index=S100.index),
                                    wind_efficiency_curve_name='knorr_extreme3')
            weather = pd.DataFrame(np.vstack([0.1*np.ones(tl),
                dum0,dum1, 101300.0*np.ones(tl)]).T,
                index=S100.index, columns = pd.MultiIndex.from_arrays([['roughness_length',
                  'wind_speed', 'temperature','pressure'], [0.0, 100.0, 2.0, 0.0]],
                names=('variable_name', 'height')),dtype=float)
            if (Farms['Project Name']=='Gayal').values:
                mc = ModelChain(NEOMwt7p5, **modelchain_data).run_model(weather)
                mxcp = NEOMwt7p5.nominal_power
            elif (Farms['Project Name']=='Wind Garden Phase 3').values:
                mc = ModelChain(NEOMwt8, **modelchain_data).run_model(weather)
                mxcp = NEOMwt8.nominal_power
            else:
                mc = ModelChain(NEOMwt5, **modelchain_data).run_model(weather)
                mxcp = NEOMwt5.nominal_power
            pout = mc.power_output.to_numpy()
            Pow[col] = pout/mxcp*Farms['Capacity'].iloc[i]

    return Pow
#%%
def ghiToPow(G,T,S10,Tracking,Cap,Lat,Lon,tilt,wgt):

    if len(G.shape)>1:
        Pow = pd.DataFrame(index=G.index,columns=G.columns)
    else:
        Pow = pd.DataFrame(index=G.index,columns=['Dummy'])
        G = pd.DataFrame(G.values,index=G.index,columns=['Dummy'])
        T = pd.DataFrame(T.values,index=T.index,columns=['Dummy'])
        S10 = pd.DataFrame(S10.values,index=S10.index,columns=['Dummy'])
    for i,col in enumerate(Pow.columns):
        if wgt[i]>0:
            if Tracking=='fixed':
                mount = pvsystem.FixedMount(surface_tilt =
                    Lat[i]+tilt, surface_azimuth=180.0)
            else:
                mount = pvsystem.SingleAxisTrackerMount(
                    axis_tilt=0.0, axis_azimuth=0.0,
                    max_angle=90.0, backtrack=True,
                    gcr=0.285714, cross_axis_tilt= 0.0,
                    racking_model=None,
                    module_height=None)
            pvarray = pvsystem.Array(
                mount=mount,
                module_parameters=module,
    #            module_parameters=dict(pdc0=1, gamma_pdc=-0.004),
                temperature_model_parameters=temperature_model_parameters)

            loc = location.Location(latitude=Lat[i],longitude=Lon[i],tz='Etc/GMT')
            ghi = G[col]
            clsky = loc.get_clearsky(G.index)
#            ghi = clsky['ghi']
            dni = (ghi*clsky['dni'].\
                to_numpy()/(clsky['ghi'].to_numpy()+0.01))
            dhi = (ghi*clsky['dhi'].\
                to_numpy()/(clsky['ghi'].to_numpy()+0.01))
            plt.plot(ghi.iloc[10000:11000],dni[10000:11000],'.')
            plt.show()
            plt.plot(clsky['ghi'].iloc[10000:10120])
            plt.plot(ghi.iloc[10000:10120])
            plt.show()
            k = np.where(ghi<1.0)[0]
            dni[k] = 0.0
            dhi[k] = 0.0
            k = np.where(dni>1100)[0]
            dni[k] = 1000.0
            t2m = T[col]-273.15
#            t2m[t2m>-100] = 25.0
            t2m[t2m<-100] = 0.0
#            t2m[t2m>-100] = 20.0
            weather = pd.DataFrame(np.vstack([ghi,dni,dhi,
                    t2m,S10[col]]).T,index=G.index,
                    columns=['ghi','dni','dhi','temp_air','wind_speed'],dtype=float)
            pvsys = pvsystem.PVSystem(arrays=[pvarray],
                                      inverter_parameters=inverter)
    #                                inverter_parameters=dict(pdc0=3))
            mc_solar = modelchain.ModelChain(pvsys,loc,aoi_model="physical",
                                             spectral_model="first_solar")
            try:
                mc_solar.run_model(weather)
                pout = mc_solar.results.ac.to_numpy()
                print(np.nanmax(pout),inverter.Paco,Cap[i])
                pout = pout*Cap[i]/inverter.Paco
            except:
                print('Error in zenith angle calculation')
                print(Cap[i],np.max(ghi))
                print(ghi)
                pout = ghi*Cap[i]/np.max(ghi)
            Pow[col] = pout

#    coeffs = 500 #Wh/person
#    Pow = np.zeros_like(G)
#    Pow = G*coeffs


    return Pow


#%% Simple Load Model
def tempsToLoad(T,coeffs):
    #Calculates load assuming a fixed diurnal and weekday cycle and a piecewise
    #linear temperature dependence (dependent only on the daily average temperature)

    Load = pd.DataFrame(np.zeros_like(T.values),index=T.index,
                        columns=list(coeffs.keys()))

    for col in coeffs.keys():

        Tc = 15.0+273.15 #Ambient temperature above which cooling is required
        Th = 5.0+273.15  #Ambient temperature below which heating is required
        #Calculate rolling daily mean temperature
        Trm = T[col].shift(freq='-12H').rolling('D').mean().reindex(T.index)

        #Calculate Day-of-Week load variability
        DoWL = pd.Series(np.ones(len(T[col]))*coeffs[col]['basel'],index=T.index)
        DoWL[(T.index.dayofweek<4)|(T.index.dayofweek==6)] += coeffs[col]['dow']
        DoWL =  DoWL.shift(freq='-12H').rolling('D').mean()

        #Calculate Hour-of-Day load variability
        HoDL = pd.Series(coeffs[col]['diurnal']*(np.sin(np.pi*(T.index.hour -\
                7)/12) + 0.25*np.sin(np.pi*(T.index.hour-3)/6)),index=T.index)

        #Calculate Weather Driven Load

        WDL = pd.Series(np.zeros_like(T[col]),index=T.index)

        #Temperatures above Tc result in cooling load
        WDL[Trm>Tc] = WDL[Trm>Tc]+ (Trm[Trm>Tc]-Tc)*coeffs[col]['coeffc']
        #Temperatures below Th result in heating load
        WDL[Trm<Th] = WDL[Trm<Th] + (Th-Trm[Trm<Th])*coeffs[col]['coeffw']

        Load[col] = WDL + HoDL + DoWL

    Load = Load[list(coeffs.keys())].sum(axis=1)

    return Load

#%% Adjust Temperatures for Climate Change

def climateAdjust(T,year=2023,order=6,delT=False):
    #year: year to which climate should be adjusted
    #order: order of polynomial fit (default = 6)
    #delT:  if not False, a number of degrees change applied to the mean
    #climate of the years 1990-2019
    #If delT is false, and year is after the last year in the record,
    #the temperature rate of change in the smoothed data is measured over the
    #last 5 years and extrapolated forward to the year requested.

    #Generate 8-year rolling mean of temperatures for fitting to smooth curve
    Tr = T.rolling(8*8766).mean().shift(-4*8766)
    Tei = np.arange(len(T))


    Tadjust = Tr.copy()
    for col in Tr.columns:
        k = np.where(T[col]>0)[0]
        pp = np.polyfit(Tei[k[0]+4*8766:k[-1]-4*8766],Tr[col].iloc[k[0]+4*8766:k[-1]-4*8766],order)
        Tsm = np.polyval(pp,Tei)
        ymax = np.max(Tr[Tr[col]>0].index.year)
        ymin = np.min(Tr[Tr[col]>0].index.year)
        if (year>ymin)&(year<ymax): #Year is within the observed data
            k1 = np.where(T.index.year==year)[0]
            Tp = Tsm[k1[0]]
            #Tp is the smoothed temperature coresponding to the desired year
        elif (year>ymax):  #Year is in the future
            k1 = np.where(T.index.year==ymax)[0]
            Tslope = Tsm[-1]-Tsm[-8766*5]
            Tp = Tsm[k1[0]] + (year-ymax+1)/5*Tslope
            #Tp is the temperature extrapolated to the future desired year, using
            #the slope of the last 5 years of the smoothed data set
        else: #Year is before the observed data begins
            k1 = np.where(T.index.year==ymin)[0]
            Tslope = Tsm[8766*5]-Tsm[0]   #Slope of first 5 years of the smoothed
                                          #time series
            Tp = Tsm[k1[0]] + (year-ymin-1)/5*Tslope
        if delT:  #If delT is provided, just adjust the temperature by that amount,
                  # from the climate of the period 1990-2020
            Tp = np.mean(Tsm[-8766*33:-8766*4])+delT
        Tdel = (Tp-Tsm)
#        plt.plot(Tr[col].iloc[k[0]+4*8766:k[-1]-4*8766])
#        plt.plot(T.index,Tsm);plt.plot(pd.to_datetime(str(year)+'-06-30 00:00+00:00'),Tp,'*');plt.show()
#        print(year,Tp,T[col][T.index.year==year].mean(),Tdel.mean(),pp)
        Tadjust[col] = T[col].values+Tdel

    return Tadjust


#%% Quantile Climatology Plot

def quantileClimatologyPlot(df,avgp,qts,title,ylim,units):
    avgplabel = {'D':'Daily','3D':'Three Daily','7D':'Weekly','W':'Weekly',
                 '2W':'Biweekly','14D':'Biweekly','M':'Monthly','5D':'Pentad',
                 '10D':'Decad'}
    try:
        avgpl = avgplabel[avgp]
    except KeyError:
        avgpl = avgp
    qtlabels = [str(int(qt*100))+'%-ile' for qt in qts]
    dfavg = df.resample(avgp).mean()
    jitter = np.random.normal(0,0.1,len(dfavg.index))
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot()
    ax.plot(dfavg.index.month+jitter,dfavg,'.',color='#C0C0C0',markersize=1)
    for qt in qts:
        ax.plot(np.arange(1,13),dfavg.groupby(dfavg.index.month).\
        quantile(qt))
    ax.set_xticks(np.arange(1,12,2),labels=['Jan','Mar','May','Jul','Sep','Nov'])
    ax.legend([avgpl]+qtlabels)
    ax.set(ylim=ylim,ylabel=units)
    ax.grid()
    ax.set_title(title)
    plt.show()


def quantileClimatologyPlotWithTemp(df,temp,avgp,qts,title,ylim,units):
    avgplabel = {'D':'Daily','3D':'Three Daily','7D':'Weekly','W':'Weekly',
                 '2W':'Biweekly','14D':'Biweekly','M':'Monthly','5D':'Pentad',
                 '10D':'Decad'}
    tempd = temp.rolling(24).mean()
    try:
        avgpl = avgplabel[avgp]
    except KeyError:
        avgpl = avgp
    qtlabels = [str(int(qt*100))+'%-ile' for qt in qts]
    dfavg = df.resample(avgp).mean()
    jitter = np.random.normal(0,0.1,len(dfavg.index))
    plt.figure(figsize=(6,3))
    for i in range(1,13):
        tempd = temp.rolling(24).mean().resample(avgp).mean()[temp.resample(avgp).mean().index.month==i]
        dfavg = df.resample(avgp).mean()[df.resample(avgp).mean().index.month==i]
        jitter = np.random.normal(0,0.1,len(dfavg.index))
        k25 = np.where(tempd<tempd.quantile(.25))[0]
        k50 = np.where((tempd>=tempd.quantile(.25))&(tempd<=tempd.quantile(.75)))[0]
        k75 = np.where(tempd>tempd.quantile(0.75))[0]
        print(len(k25),len(k50),len(k75))
        p1, = plt.plot(dfavg.index.month[k25]+jitter[k25],dfavg.iloc[k25],'.',color='#1010EE',markersize=1,label=avgpl+'Cold')
        p2, = plt.plot(dfavg.index.month[k50]+jitter[k50],dfavg.iloc[k50],'.',color='#C0C0C0',markersize=1,label=avgpl)
        p3, = plt.plot(dfavg.index.month[k75]+jitter[k75],dfavg.iloc[k75],'.',color='#EE1010',markersize=1,label=avgpl+'Hot')

    dfavg = df.resample(avgp).mean()
    qleg = []
    for qt in qts:
        pp, = plt.plot(dfavg.groupby(dfavg.index.month).quantile(qt))
        qleg.append(pp)
    plt.xticks(np.arange(1,12,2),labels=['Jan','Mar','May','Jul','Sep','Nov'])
    plt.legend([p1,p2,p3]+qleg,[avgpl+'Cold',avgpl,avgpl+'Warm']+qtlabels,loc=1)
    plt.ylim(ylim)
    plt.xlim(0,18)
    plt.grid()
    plt.ylabel(units)
    plt.title(title)
    plt.show()


#%%  Diurnal Cycle Climatology Plot

def quantileClimoDiurnalPlot(df,month,qts,title,xlim, ylim,units):
    monlab = pd.to_datetime('2010-'+'{:02}'.format(month)).strftime('%B')
    xlab = str(df.index[0].tz)
    qtlabels = [str(int(qt*100))+'%-ile' for qt in qts]
    k = np.where(df.index.month==month)[0]
    dfm = df.iloc[k]
    dfm.index = dfm.index.tz_convert('Asia/Riyadh')
    jitter = np.random.normal(0,0.1,len(dfm.index))
    plt.plot(dfm.index.hour+jitter,dfm,'.',color='#C0C0C0',markersize=1)
    for qt in qts:
        dfm.groupby(dfm.index.hour).\
            quantile(qt).plot(legend=False)
    plt.legend(['Hourly']+qtlabels)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()
    plt.xlabel(xlab)
    plt.ylabel(units)
    plt.title(title+' for '+monlab)
    plt.show()



#%%

worldbd = gpd.read_file(datadir+r'/world-administrative-boundaries.shp')
era5 = pd.read_pickle(datadir+r'/WindSolarDemoFenner.pkl')

farm = 'Fenner'
ghi = era5['ghi'].to_frame(farm)
t2m = era5['t2m'].to_frame(farm)
s10 = era5['s10'].to_frame(farm)
s100 = era5['s100'].to_frame(farm)


Lat = [42.98]
Lon = [284.24-360.0]
tilt = [0.0]
tracking = 'single'
wgt = [1.0]
cap = [100.0]

solar_firstsolar = ghiToPow(ghi,t2m,s10,tracking,cap,Lat,Lon,tilt,
                 wgt)

Farms = pd.DataFrame(np.array([100.0]),columns=['Capacity'])
Farms['Project Name'] = 'Fenner'
WakeLoss = True

wind = windToPow(s100,t2m,Farms,WakeLoss=WakeLoss)
WakeLoss = False
windNoWake = windToPow(s100,t2m,Farms,WakeLoss=WakeLoss)

qts = np.array([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])
for avgp in ['D','W',"ME"]:
    quantileClimatologyPlot(wind,avgp,qts=qts,title='Wind Generation (1970-2025)',
                            ylim=(0,100), units='MW')

    quantileClimatologyPlot(solar,avgp,qts=qts,title='Solar Generation (1970-2022)',
                            ylim=(0,100), units='MW')


