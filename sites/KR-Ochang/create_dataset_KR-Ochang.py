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
import glob

# paths
oshome=os.getenv('HOME')
projpath = f'{oshome}/git/urban-plumber_pipeline'            # root of repository
datapath = f'{oshome}/git/urban-plumber_pipeline/input_data'       # raw data path (site data, global data)

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

sitename = 'KR-Ochang'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 9
long_sitename = 'Ochang, South Korea'
obs_contact = 'Jinkyu Hong (jhong@yonsei.ac.kr), Je-Woo Hong (jwhong@kei.re.kr)'
obs_reference = 'Hong, Hong, Chun, Lee, Chang, Lee, Yi, Park, Byun, Joo (2019): https://doi.org/10.1186/s13021-019-0128-6; Hong, Lee, Hong (2021): https://doi.org/10.22647/EAPL-OC_JN2021'
obs_comment = 'No bias correction applied to ERA5 derived precipitation (no nearby long-term and complete GHCND site data)'
photo_source='Keunmin Lee'
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
        pass
        # syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        # nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,projpath,nshow=6)
        # print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        # rain_sites = ['KSM00047133',    # TAEJON, KS       36.300    127.400
        #               'KSM00047108',    # SEOUL CITY, KS   37.567    126.967
        #               'KS000047112',    # INCHEON, KS      37.467    126.633
        #               'KSM00047101']    # CHUNCHEON, KS    37.900    127.733

        # rain_obs = pipeline_functions.get_ghcnd_precip(sitepath,syear,eyear,rain_sites)
        # pipeline_functions.write_ghcnd_precip(sitepath,sitename,rain_obs)

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

    fpath_obs = f'{datapath}/{sitename}/Ochang_v2.xls'

    # read data csv
    print('reading raw data file')
    raw1 = pd.read_excel(fpath_obs, sheet_name='forcing data')
    raw2 = pd.read_excel(fpath_obs, sheet_name='evaluation data')

    # get times from data and reindex
    times = pd.date_range(start='2015-06-08 00:00', end='2017-07-26 9:00', freq='30Min')
    raw1.index = times
    raw2.index = times

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw1['SWdown'],
            LWdown = raw1['LWdown'],
            Tair   = raw1['Tair'],
            #### provided Qair gives RH >> 100, using HMP155 instead
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=raw1['HMP155_RH'],
                        temp=raw1['Tair'],
                        pressure=raw1['PSurf']),
            PSurf  = raw1['PSurf'],
            Rainf  = raw1['Rainf']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw1['wind_speed'],
                        wind_dir_from=raw1['wind_dir'], # NOTE wdir taken from nearby site
                        )[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw1['wind_speed'],
                        wind_dir_from=raw1['wind_dir'], # NOTE wdir taken from nearby site
                        )[0],
            SWup   = raw2['SWup'],
            LWup   = raw2['LWup'],
            Qle    = raw2['Qle'],
            Qh     = raw2['Qh'],
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds

def plot_rh(df,raw1):

    RH_from_Qair = pipeline_functions.convert_qair_to_rh(raw1['Qair'],df['Tair'],df['PSurf'])
    Qair_from_HMP155 = pipeline_functions.convert_rh_to_qair(raw1['HMP155_RH'],raw1['Tair'],raw1['PSurf'])
    Qair_from_RH = pipeline_functions.convert_rh_to_qair(raw1['RH'],raw1['Tair'],raw1['PSurf'])

    ####################################

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    raw1['Qair'].plot(ax=ax, color='b',label='Qair as provided', lw=1)
    ax.set_ylabel('Specific humidity (kg/kg)')
    ax.set_title('As provided: specific humidity (Qair)')
    ax.legend(loc='upper right')
    plt.savefig('01-observed_Qair.png')

    ####################################

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    RH_from_Qair.plot(ax=ax,color='b',label='RH converted from Qair', lw=0.25,mew=1,marker='.',ms=3)
    ax.set_ylabel('Relative humidity (%)')
    ax.set_title('Relative humidity converted from Qair')
    ax.legend(loc='upper right')
    plt.savefig('02-converted_Qair.png')

    ####################################

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    RH_from_Qair.plot(ax=ax,color='b',label='RH converted from Qair', lw=0.25,mew=1,marker='.',ms=3)
    raw1['RH'].plot(ax=ax,color='g',label='RH as provided', lw=0.25,mew=1,marker='.',ms=3)
    ax.set_ylabel('Relative humidity (%)')
    ax.set_title('Relative humidity converted from Qair')
    ax.legend(loc='upper right')
    plt.savefig('03-converted_Qair_with_RH.png')

    ####################################

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    RH_from_Qair.plot(ax=ax,color='b',label='RH converted from Qair', lw=0.25,mew=1,marker='.',ms=3)
    raw1['RH'].plot(ax=ax,color='g',label='RH as provided', lw=0.25,mew=1,marker='.',ms=3)
    raw1['HMP155_RH'].plot(ax=ax,color='r',label='RH from HMP155', lw=0.25,mew=1,marker='.',ms=3)
    ax.set_ylabel('Relative humidity (%)')
    ax.set_title('Relative humidity converted from Qair')
    ax.legend(loc='upper right')
    plt.savefig('04-converted_Qair_with_RH_and_HMP155.png')

    ####################################

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    raw1['Qair'].plot(ax=ax, color='b',label='Qair as provided', lw=0.25, mew=1, marker='.', ms=3)
    Qair_from_HMP155.plot(ax=ax,color='r',label='Qair converted from HMP155', lw=0.25, mew=1, marker='.', ms=3)
    Qair_from_RH.plot(ax=ax,color='g',label='Qair converted from RH', lw=0.25,mew=1,marker='.',ms=3)

    ax.set_ylabel('Specific humidity (kg/kg)')
    ax.set_title('Specific humidity (3 methods)')
    ax.legend(loc='upper right')
    plt.savefig('05-Qair_comparisons.png')

    return

################################################################################

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

