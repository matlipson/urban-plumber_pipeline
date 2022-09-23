'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2022 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "site-specific processing wrapper"
__version__ = "2022-09-15"
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

sitename = 'US-WestPhoenix'
out_suffix = 'v1'
sitedata_suffix = 'v1'

local_utc_offset_hours = -7.0
long_sitename = 'West Phoenix, Arizona, United States'
obs_contact = 'Stevan Earl: caplter.data@asu.edu, Winston Chow: winstonchow@smu.edu.sg'
obs_reference = 'Chow (2017): https://doi.org/10.6073/pasta/fed17d67583eda16c439216ca40b0669; Chow et al (2014): https://doi.org/10.1002/joc.3947'
obs_comment = 'SoilTemp at 2cm depth.'
photo_source='S. Earl'
history = 'v0.9 (2021-09-08): beta issue; v1 (2022-09-15): with publication in ESSD'

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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=3)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['US1AZMR0015', # PHOENIX 3.6 NW
                      'USC00028598', # TOLLESON
                      'USW00023183'] # PHOENIX AIRPORT

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

    z0 = pipeline_functions.calc_roughness_from_clean(clean_ds,sitedata)
    z0 = pd.Series(z0).mean()
    print(f'z0 from obs: {z0:.3f}')

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
    raw = pd.read_csv(f'{datapath}/{sitename}/591_fluxtower_data_a16dafefa6836b419580f353b76bc2a0.csv', na_values='NA')

    # get times from data and reindex
    times = pd.date_range(start='2011-12-16 11:30', end='2012-12-31 23:30', freq='30Min')
    raw.index = times

    ts = (times[1]-times[0]).seconds

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['K_inc'],
            LWdown = raw['L_inc'],
            Tair   = raw['t_hmp_mean'] + 273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh = raw['RH'],
                        temp = raw['t_hmp_mean'] + 273.15, 
                        pressure = raw['press_mean']*1000.,
                        ),
            PSurf  = raw['press_mean']*1000.,
            Rainf  = raw['Rain']/ts,
            Snowf  = np.nan,
            Wind_N = raw['Uy_Avg'],
            Wind_E = raw['Ux_Avg'],
            SWup   = raw['K_out'],
            LWup   = raw['L_out'],
            Qle    = raw['LE_wpl_c'],
            Qh     = raw['Hc_c'],
            ######################
            Qtau   = raw['tau'],
            SoilTemp = raw['Temp_C_Avg'] + 273.15 # 2cm
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)


    # convert times
    df = pipeline_functions.convert_local_to_utc(df,siteattrs['local_utc_offset_hours'])

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds

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

