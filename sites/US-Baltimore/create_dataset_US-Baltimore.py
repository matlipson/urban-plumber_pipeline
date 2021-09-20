'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "site-specific processing wrapper"
__version__ = "2021-09-20"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__description__ = 'Wrapper for processing individual sites. Includes setting site-specific information, importing raw site data, calling pipeline functions, creating site plots and webpages etc.'

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import argparse
import importlib

# paths
oshome=os.getenv('HOME')
projpath = f'{oshome}/git/urban-plumber_pipeline'              # root of repository
datapath = f'{oshome}/git/urban-plumber_pipeline/input_data'   # raw data path (site data, global data)

sys.path.append(projpath)
import pipeline_functions
importlib.reload(pipeline_functions)

##########################################################################
# MANUAL USER INPUTS
##########################################################################
# these are overridden with --existing flag (i.e. python create_dataset_XX.py --existing)

create_raw_obs_nc      = True  # create obs nc from original format
create_rain_file       = True  # find and load nearest GHCND
qcplotdetail           = True  # plot quality control diurnal and timeseries
forcingplots           = True  # plot forcing and analysis period obs and gap-filling
create_outofsample_obs = True  # for testing bias-correction on half of available obs
fullpipeline           = True  # undertake full pipeline e.g. cleaning, bias correction, data creation

##########################################################################
# COMMAND LINE ARGUMENT PARSING
##########################################################################

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--log',       help='log print statements to file', action='store_true')
parser.add_argument('--projpath',  help='replaces projpath with new path')
parser.add_argument('--datapath',  help='replaces datapath with new path')
parser.add_argument('--existing',  help='use existing outputs (processing already run)', action='store_true')
parser.add_argument('--globaldata',help='output site characteristics from global dataset (if available)', action='store_true')

args = parser.parse_args()

log_stout = False
if args.log:
    log_stout = True
if args.projpath:
    print(f'updating projpath to {projpath}')
    projpath = args.projpath
if args.datapath:
    print(f'updating datapath to {datapath}')
    datapath = args.datapath
if args.existing:
    print('using existing files')
    create_raw_obs_nc      = False
    create_rain_file       = False
    qcplotdetail           = False
    forcingplots           = False
    create_outofsample_obs = False
    fullpipeline           = False

##########################################################################
# SITE SPECIFIC INFORMATION
##########################################################################

sitename = 'US-Baltimore'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = -5
long_sitename = 'Cub Hill, Baltimore, United States'
obs_contact = 'Ben Crawford (benjamin.crawford@ucdenver.edu), Sue Grimmond (c.s.grimmond@reading.ac.uk)'
obs_reference = 'Crawford, Grimmond and Christen (2011): https://doi.org/10.1016/j.atmosenv.2010.11.017'
obs_comment = 'Hourly rainfall from Baltimore Washington International Airport where available. SoilTemp at 10cm depth.'
photo_source='[Crawford et al. (2011)](https://doi.org/10.1016/j.atmosenv.2010.11.017)'
history = 'v0.9 (2021-09-08): beta issue'

##########################################################################
# MAIN
##########################################################################

def main():

    sitepath = f'{projpath}/sites/{sitename}'

    print('preparing site data and attributes')
    sitedata, siteattrs = pipeline_functions.prep_site(
        sitename, sitepath, out_suffix, sitedata_suffix, long_sitename, 
        local_utc_offset_hours, obs_contact, obs_reference, obs_comment,
        history, photo_source, args.globaldata, datapath)

    print('getting observation netcdf\n')
    if create_raw_obs_nc:

        print(f'creating observational NetCDF in ALMA format\n')
        raw_ds = import_obs(sitedata,siteattrs)
        raw_ds = pipeline_functions.set_raw_attributes(raw_ds, siteattrs)
        
    else:
        fpath = f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.nc'
        raw_ds = xr.open_dataset(fpath)

    if create_rain_file:
        syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=19)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['USW00093721'] # BALTIMORE WASHINGTON AIRPORT, MD US  39.1667   -76.6833

        rain_obs = pipeline_functions.get_ghcnd_precip(sitepath,datapath,syear,eyear,rain_sites)
        pipeline_functions.write_ghcnd_precip(sitepath,sitename,rain_obs)

    ############################################
    ############ pipeline MAIN call ############
    raw_ds, clean_ds, watch_ds, era_ds, corr_ds, lin_ds, forcing_ds = pipeline_functions.main(
        datapath     = datapath,
        sitedata     = sitedata,
        siteattrs    = siteattrs,
        raw_ds       = raw_ds,
        fullpipeline = fullpipeline,
        qcplotdetail = qcplotdetail)
    ############################################

    print('post processing, plotting and checking')
    pipeline_functions.post_process_site(sitedata,siteattrs,datapath,
        raw_ds,forcing_ds,clean_ds,era_ds,watch_ds,corr_ds,lin_ds,
        forcingplots,create_outofsample_obs)

    print(f'{sitename} done!')

    return raw_ds, clean_ds, watch_ds, era_ds, corr_ds, forcing_ds

##########################################################################
# specific functinos
##########################################################################

def import_obs(sitedata,siteattrs):

    # read data csv
    print('reading raw data file')
    # names = ['dectime','HR','MN','IL','ustar','Qh','QhTc','Qe','FCo2','U','V','W','T', 'TC', 'H20','C02','WS','DIR','UU','VV','WW','TT','TCTC','HH','CC','wT','wTC','wH','wC','rho_a','rho_v','cp','ea','es','T2','RH2','P','Vane','Kdn','Kup','Ldn','Lup','Tkz','Qstar','QG','Tsoil','Moist','SW','Rain','PAR','sin','lin','u_g','v_g','w_g','t_g','tcg','h_g','c_g','u_s','v_s','w_s','t_s','tcs ','h_s ','c_s','WPL_Q','WPL_C','QH_bcor','Tsens']
    # # units:           h   m    n     m/s   W/m2  W/m2  W/m2 uml/m2/s   m/s     degC degC  g/m3  mmol/m3 m/s deg    ^2/100     ^2/100   ^2/100   ^2/100   ^2/100      cm*C/s kg/m3  g/m3  J/K/kg  hPa hPa degC    %   hPa   deg  W/m2   W/m2  W/m2  W/m2  degC   W/m2   W/m2  degC    0/0    N/A   mm   W/m2   N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A                                    W/m2  uml/m2/s   W/m2

    raw_list = []
    for i in range(2,7):
        tmp = pd.read_csv(f'{datapath}/{sitename}/from_crawford/Ba200{i}.csv', 
            na_values=[-999.000,'-999.000'], skiprows=[0,2,3])
        # get index from dectime
        tmp.index = pd.to_datetime(tmp.DECIMALTIME-1, unit='D', origin=f'200{i}').round('60Min')
        raw_list.append(tmp)
    raw = pd.concat(raw_list)
    # # remove observed rain period where no zero rain recorded, but rain fell
    # raw.loc['2003-11-20':'2004-07-03','Rain'] = np.nan
    # # rescale site rain based on GHCND daily rainfall in same periods
    # raw.loc[:'2003-11-20','Rain'] = raw.loc[:'2003-11-20','Rain']*5.652
    # raw.loc['2004-07-03':,'Rain'] = raw.loc['2004-07-03':,'Rain']*6.974

    # get times from data and reindex
    times = pd.date_range(start='2002-01-01 00:00:00', end='2006-12-31 23:00:00', freq='60Min')

    # converting absolute humidity to specific through air density gives unrealistically high humidity values
    qair = (raw['H2O']/1000.)/raw['rho_a']

    # relative humidity givers values >> 100 if 'T' tower temperature is used, ea must be at 2m, not tower
    esat = pipeline_functions.calc_esat(raw['T2']+273.15,raw['P']*100.,mode=0)
    rh = 100 * (raw['ea']*100)/esat
    rh2 = 100*raw['ea']/raw['es']

    '''
    Header information from IUFLUX_ColumnDefs.csv
    Kdn:    Incoming shortwave          [W/m2]
    Ldn:    Downwelling longwave        [W/m2]
    Kup:    Reflected shortwave         [W/m2]
    Lup:    Upwelling longwave          [W/m2]
    RH2:    Slow RH sensor              [%]
    T:      Acoustic temperature
    T2:     Slow temp sensor            [degC]
    TC:     Thermocouple air temperatuer [degC]
    P:      Air pressure                [hPa]
    V:      V-component wind velocity   [m/s]
    U:      U-component wind velocity   [m/s]
    Qe:     Latent heat flux            [W/m2]
    Qh:     Sensible heat flux (acoustic temperature)   [W/m2]
    Tsoil:  Soil temperature            [degC]
    USTAR:  Friction velocity           [m/s]
    '''

    bwi = get_hourly_precip(siteattrs)

    TairC = raw['TC']+273.15
    Tair2 = raw['T2']+273.15
    # drop where TC diverges from ERA for short period
    TairC.loc['2003-09-28':'2003-11-18'] = np.nan
    TairC.loc['2004-07-21 12:00':'2004-07-21 17:00'] = np.nan
    Tair2.loc['2005-02-15':'2005-03-03'] = np.nan
    # combine using TC first
    Tair = TairC.combine_first(Tair2)

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['Kdn'],
            LWdown = raw['Ldn'],
            Tair   = Tair,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=raw['RH2'],
                        temp=Tair,
                        pressure=raw['P']*100.),
            PSurf  = raw['P']*100.,
            Rainf  = bwi/3600,
            Snowf  = np.nan,
            Wind_N = -raw['V'],
            Wind_E = -raw['U'],
            SWup   = raw['Kup"'],
            LWup   = raw['Lup'],
            Qle    = raw['Qe'],
            Qh     = raw['Qh'],
            ######################
            # Qg     = raw['QG'],
            SoilTemp = raw['Tsoil']+273.15, # @ 10cm
            Qtau     = pipeline_functions.convert_ustar_to_qtau(
                        ustar=raw['USTAR'],
                        temp=Tair,
                        pressure=raw['P']*100.,
                        air_density=raw['rho_a'])
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)
    # replace all rain flags with filled from obs (from nearby met station)
    df['Rainf_qc'] = np.where(df['Rainf'].isna(), 3, 1)

    # convert times
    df = pipeline_functions.convert_local_to_utc(df,siteattrs['local_utc_offset_hours'])

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds

def get_hourly_precip(siteattrs):

    hourly = pd.read_csv(f'{datapath}/{sitename}/GHCND_USW00093721_hourly.csv',na_values=25399.75)
    hourly.index = pd.to_datetime(hourly.DATE)

    times = pd.date_range(start='2002-01-01 00:00', end='2006-12-31 23:00', freq='60Min')
    bwi=pd.Series(data=hourly.loc['2002':'2006','HPCP'],index=times)

    # fill na with 0 where observations available
    bwi.loc[:'2006-08-01 01:00'] = bwi.loc[:'2006-08-01 01:00'].fillna(0)
    bwi.loc['2006-09-01 00:00':] = bwi.loc['2006-09-01 00:00':].fillna(0)

    return bwi

################################################################

if __name__ == "__main__":

    if log_stout:
        fname = f'log_processing_{sitename}_{out_suffix}.txt'
        print(f'logging print statements to {fname}')

        orig_stdout = sys.stdout
        f = open(f'{projpath}/sites/{sitename}/{fname}', 'w')
        sys.stdout = f

    raw_ds, clean_ds, watch_ds, era_ds, corr_ds, forcing_ds = main()

    if log_stout:
        sys.stdout = orig_stdout
        f.close()

    print('done!')

