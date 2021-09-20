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

sitename = 'AU-SurreyHills'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 10.0
long_sitename = 'Surrey Hills, Melbourne, Australia'
obs_contact = 'Andrew Coutts (andrew.coutts@monash.edu), Nigel Tapper (nigel.tapper@monash.edu)'
obs_reference = 'Coutts, Beringer and Tapper (2007a): https://doi.org/10.1175/JAM2462.1; Coutts, Beringer and Tapper (2007b) https://doi.org/10.1016/j.atmosenv.2006.08.030'
obs_comment = 'Gap-filled from nearby AU-Preston tower where available'
history = 'v0.9 (2021-09-08): beta issue'
photo_source='[Coutts et al. (2007)](http://doi.org/10.1016/j.atmosenv.2006.08.030)'

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
        print('creating GHCND continuous rain files')
        syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=2)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['ASN00086096',   # PRESTON RESERVOIR, AS                 -37.7214   145.0059
                      'ASN00086351']   # BUNDOORA LATROBE UNIVERSITY, AS       -37.7163   145.0453

        rain_obs = pipeline_functions.get_ghcnd_precip(sitepath,datapath,syear,eyear,rain_sites)
        pipeline_functions.write_ghcnd_precip(sitepath,sitename,rain_obs)

    ############################################
    ############ pipeline MAIN call ############
    raw_ds, clean_ds, watch_ds, era_ds, corr_ds, lin_ds, forcing_ds= pipeline_functions.main(
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
# specific functions
##########################################################################

def import_obs(sitedata,siteattrs):

    sitepath = siteattrs["sitepath"]

    # read data csv
    print('reading raw data file')
    raw = pd.read_excel(f'{datapath}/{sitename}/SURREY_HILLS_CITIES.xls',skiprows=1)
    times = pd.date_range(start='2004-02-23 15:00:00', end='2004-07-20 9:30:00', freq='30Min')
    raw.index = times

    rad1 = pd.read_csv(f'{datapath}/{sitename}/Original_Radiation_Surrey_Hills_1.txt',delim_whitespace=True,header=None,
                names=['datetime','SWdown','SWup','LWdown','LWup','Rnet_calc','Rnet_obs'])
    rad1.index = times
    rad1 = rad1.drop('datetime',axis=1)

    fill = xr.open_dataset(f'{projpath}/sites/AU-Preston/timeseries/AU-Preston_clean_observations_{out_suffix}.nc').squeeze().to_dataframe()

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['Kdown'].where(rad1['SWdown'].notna()),
            LWdown = raw['Ldown'].where(rad1['LWdown'].notna()),
            Tair   = raw['temp']+273.15,
            Qair   = raw['specific Q'],
            PSurf  = raw['press_gap']*1000.,
            Rainf  = raw['precip']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['wind_spd'],
                        wind_dir_from=raw['compass wind_final'])[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['wind_spd'],
                        wind_dir_from=raw['compass wind_final'])[0],
            SWup   = raw['Kup'].where(rad1['SWup'].notna()),
            LWup   = raw['Lup'].where(rad1['LWup'].notna()),
            Qle    = raw['QE'].where(raw['Flux_validity (2=good, gap filled otherwise)']==2),
            Qh     = raw['QH'].where(raw['Flux_validity (2=good, gap filled otherwise)']==2),
            ######################
            Qtau   = raw['tau'],
            )

    # remove spurious Qtau
    df.loc[:,'Qtau'][df.loc[:,'Qtau']<-2] = np.nan

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    # convert times
    df.index.name='time'
    df = pipeline_functions.convert_local_to_utc(df,siteattrs['local_utc_offset_hours'])

    # this dataset has been pre-gap filled, but without flags to indicate which periods
    # try to find gap filled periods through where Preston==SurryHills
    for flux in ['SWdown','LWdown','PSurf','Wind_N','Wind_E']:
        # first remove variables that are very close to Preston (assumed previously filled)
        df.loc[abs(df[flux]-fill[flux])<0.01,flux] = np.nan
        # then refill and update qc flags
        tmp = fill[flux].where(df[flux].isna())
        tmp_idx = tmp.dropna().index
        df.loc[tmp_idx,f'{flux}_qc'] = 1
        df[flux] = df[flux].fillna(tmp)

    # convert pandas dataframe to xarray dataset
    obs_ds = df.to_xarray()

    return obs_ds

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

