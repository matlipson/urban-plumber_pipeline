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

sitename = 'KR-Jungnang'
out_suffix = 'v1'
sitedata_suffix = 'v1'

local_utc_offset_hours = 9
long_sitename = 'Jungnang, Seoul, South Korea'
obs_contact = 'Jinkyu Hong (jhong@yonsei.ac.kr), Je-Woo Hong (jwhong@kei.re.kr)'
obs_reference = 'Hong, Lee, Hong (2021): https://doi.org/10.22647/EAPL-OC_JN2021'
obs_comment = 'No bias correction applied to ERA5 derived precipitation (no nearby long-term and complete GHCND site data)'
photo_source='Keunmin Lee'
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
        pass
        # syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        # nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=6)
        # print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        # rain_sites = ['KSM00047108',    # SEOUL CITY, KS   36.300    127.400
        #               'KSM00047108',    # SEOUL CITY, KS   37.567    126.967
        #               'KS000047112',    # INCHEON, KS      37.467    126.633
        #               'KSM00047101']    # CHUNCHEON, KS    37.900    127.733

        # rain_obs = pipeline_functions.get_ghcnd_precip(sitepath,datapath,syear,eyear,rain_sites)
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

    fpath_obs = f'{datapath}/{sitename}/Jungnang_v3.xls'

    # read data csv
    print('reading raw data file')
    forcing = pd.read_excel(fpath_obs, sheet_name='forcing data')
    evaluation = pd.read_excel(fpath_obs, sheet_name='evaluation data')

    # get times from data and reindex
    times = pd.date_range(start='2017-01-25 01:00', end='2019-04-29 15:30', freq='30Min')
    forcing.index = times
    evaluation.index = times

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = forcing['SWdown'],
            LWdown = forcing['Lwdown'],
            Tair   = forcing['Tair'],
            #### provided Qair gives RH >> 100, using HMP155 instead
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=forcing['HMP155_RH'],
                        temp=forcing['Tair'],
                        pressure=forcing['PSurf']),
            PSurf  = forcing['PSurf'],
            Rainf  = forcing['Rainf']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=forcing['wind_speed'],
                        wind_dir_from=forcing['wind_dir'], # NOTE wdir taken from nearby site
                        )[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=forcing['wind_speed'],
                        wind_dir_from=forcing['wind_dir'], # NOTE wdir taken from nearby site
                        )[0],
            SWup   = evaluation['SWup'],
            LWup   = evaluation['LWup'],
            Qle    = evaluation['Qle'],
            Qh     = evaluation['Qh'],
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

    # df = df.loc[:'2018']

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds

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

