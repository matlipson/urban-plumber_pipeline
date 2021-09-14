'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "site-specific processing wrapper"
__version__ = "2021-09-08"
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

sitename = 'UK-Swindon'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 0.
long_sitename = 'Swindon, United Kingdom'
obs_contact = 'Helen Claire Ward (helen.ward@uibk.ac.at), Jonathan Evans (jge@ceh.ac.uk), Sue Grimmond (c.s.grimmond@reading.ac.uk)'
obs_reference = 'Ward, Evans and Grimmond (2013): https://doi.org/10.5194/acp-13-4645-2013'
obs_comment = '-'
photo_source='H. C. Ward'
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

    if create_raw_obs_nc:

        print(f'creating observational NetCDF in ALMA format\n')
        raw_ds = import_obs(sitedata,siteattrs)
        raw_ds = pipeline_functions.set_raw_attributes(raw_ds, siteattrs)

    else:
        print('loading existing observational netcdf')
        fpath = f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.nc'
        raw_ds = xr.open_dataset(fpath)

    if create_rain_file:
        syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=3)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites =['UKE00102158',   # LYNEHAM, UK    51.5017     -1.9908
                     'UKM00003740',   # LYNEHAM, UK    51.505      -1.993
                     'UK000056225']   # OXFORD, UK     51.7667     -1.2667

        rain_obs = pipeline_functions.get_ghcnd_precip(sitepath,datapath,syear,eyear,rain_sites)
        pipeline_functions.write_ghcnd_precip(sitepath,sitename,rain_obs)

    ############################################
    ############ pipeline MAIN call ############
    ############################################
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
    '''During co-author comments on manuscript new data was provided which supersedes input data provided to modellers
    This data is in the original 30 min processing periods, other data had been reprocessed to 60 min
    see email from Helen Ward at 8:48 16 August 2021
    '''

    # read data csv
    print('reading raw data file')

    raw = pd.read_csv(f'{datapath}/{sitename}/AnalysisData_Sw.txt', na_values=-999)
    raw2 = pd.read_csv(f'{datapath}/{sitename}/InputData_SW.txt', na_values=-999)

    # get times from data and reindex
    times = pd.date_range(start='2011-01-01 00:30:00', end='2013-05-01 0:00:00', freq='30Min')
    raw.index = times
    raw2.index = times

    raw = raw.loc['2011-05-11 19:00':'2013-04-25 11:00']

    # create dataframe in ALMA format
    df = pd.DataFrame(index=raw.index)
    df = df.assign(
            SWdown = raw['kdown'],
            LWdown = raw['ldown'],
            Tair   = raw['Tair'] + 273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh = raw['RH'],
                        temp = raw['Tair'] + 273.15, 
                        pressure = raw['pres']*1000.,
                        ),
            PSurf  = raw['pres']*1000.,
            Rainf  = raw['rain']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['U'],
                        wind_dir_from=raw['wdir'],
                        )[0],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['U'],
                        wind_dir_from=raw['wdir'],
                        )[1],
            SWup   = raw['kup'],
            LWup   = raw['lup'],
            Qle    = raw['qe'],
            Qh     = raw['qh'],
            ######################
            Qtau   = pipeline_functions.convert_ustar_to_qtau(
                        ustar = raw['ustar'],
                        temp = raw['Tair'] + 273.15,
                        pressure = raw['pres']*1000.,
                        air_density = pipeline_functions.calc_density(
                            temp=raw['Tair'] + 273.15,
                            pressure=raw['pres']*1000.))
            )
    # create dataframe without gap filling
    df2 = pd.DataFrame(index=raw.index)
    df2 = df2.assign(
            SWdown = raw2['kdown'],
            LWdown = raw2['ldown'],
            Tair   = raw2['Tair'] + 273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh = raw2['RH'],
                        temp = raw2['Tair'] + 273.15, 
                        pressure = raw2['pres']*1000.,
                        ),
            PSurf  = raw2['pres']*1000.,
            Rainf  = raw2['rain']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw2['U'],
                        wind_dir_from=raw['wdir'],
                        )[0],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw2['U'],
                        wind_dir_from=raw['wdir'],
                        )[1],
            )

    for key in df.columns:
        # set qc flag to 0 for missing values
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    # gap fill with local observations
    for key in df2.columns:
        # find additional values
        tmp = df2[key].where(df[key].isna())
        # get indexes of additional values
        tmp_idx = tmp.dropna().index
        # fill with additional values
        df[key] = df[key].fillna(tmp)
        # set qc flag to 1 for additional values
        df.loc[tmp_idx,f'{key}_qc'] = 1

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds

#########################

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

#########################

