'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "site-specific processing wrapper"
__version__ = "2022-05-23"
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

sitename = 'SG-TelokKurau06'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'


local_utc_offset_hours = 8.0
long_sitename = 'Telok Kurau, Singapore (2006-2007)'
obs_contact = 'Matthias Roth (geomr@nus.edu.sg)'
obs_reference = 'Roth, Jansson and Velasco (2017) https://doi.org/10.1002/joc.4873; Flux tower website: https://www.nusurbanclimate.com/tkfluxtowersingapore' 
obs_comment = 'ERA5 data for gap-filling is from adjacent grid over land.'
photo_source='M. Roth'
history = 'v0.92 (2022-05-29): beta issue'

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

        # for singapore use pre-downloaded Changi rainfall data
        rain_outpath = f'{datapath}/{sitename}/GHCND_Changi_local_data.csv'
        get_sing_rainfall(rain_outpath,syear,eyear)

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=2)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['SNM00048698',       # SINGAPORE CHANGI INTERNATIONAL, SN     1.35       103.994
                      'Changi_local_data'] # Changi from SN met data (brought back 1 day to align with GHNCD): 

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

    # rainfall missing periods set to zero per advice from M. Roth (27 May 2022)

    fpath_obs = f'{datapath}/{sitename}/TK0607_fluxtower_paper.xls'

    # read data csv
    print('reading raw data file')
    raw = pd.read_excel(fpath_obs, sheet_name='data')

    # remove trailing whitespace
    raw.columns = raw.columns.str.strip()

    # get times from data and reindex
    times = pd.date_range(start='2006-05-01 00:30:00', end='2007-04-01 00:00:00', freq='30Min')
    raw.index = times

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['SW in  (W/m2)'],
            LWdown = raw['LW in (W/m2)'],
            Tair   = raw['Air temp sensor height (degC)']+273.15,
            Qair   = pipeline_functions.convert_vapour_pressure_to_qair(
                        e = raw['Vapour pressure sensor height (kPa)']*1000,
                        temp = raw['Air temp sensor height (degC)']+273.15, 
                        pressure = raw['Air pressure [kPa]']*1000.,
                        ),
            PSurf  = raw['Air pressure [kPa]']*1000.,
            Rainf  = raw['Rainfall (mm)'].fillna(0)/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['Mean wind speed [m/s]'],
                        wind_dir_from=raw['Wind direction (degrees)'],
                        )[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['Mean wind speed [m/s]'],
                        wind_dir_from=raw['Wind direction (degrees)'],
                        )[0],
            SWup   = raw['SW out  (W/m2)'],
            LWup   = raw['LW out (W/m2)'],
            Qle    = raw['Latent heat flux [W/m2]'],
            Qh     = raw['Sensible heat flux [W/m2]'],
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

def get_sing_rainfall(rain_outpath,syear=2005,eyear=2016):
    '''imports daily rainfall data (must be pre-downloaded as curl requests disabled by their server'''

    df = pd.DataFrame()
    stations = ['Changi']
    sids = [24]
    # stations = ['Changi','MarineParade','TanjongKatong']
    # sids = [24,113,78]

    syear=int(syear)
    eyear=int(eyear)

    years = [str(x) for x in np.arange(syear,eyear+1,1)]
    months = [str(x).zfill(2) for x in np.arange(1,13)]

    for station,sid in zip(stations,sids):
        tmp_list = []
        for year in years:
            for month in months:

                print(f'processing {sid}_{year}{month}')

                fname = f'DAILYDATA_S{sid}_{year}{month}.csv'
                fpath = f'{datapath}/{sitename}/MetStation/{fname}'

                try:
                    tmp_list.append(get_met_obs_file(fpath))
                except:
                    print(f'{fname} not found, try passing download_files=True')
                    
        tmp = pd.concat(tmp_list, axis=0)
        df['PRCP'] = tmp['Daily Rainfall Total (mm)']
        df['NAME'] = station
        # shift to match day with GHCND
        df['PRCP'] = df['PRCP'].shift(1).values

        df.index.name = 'DATE'
        df = df.reset_index()
        df.to_csv(rain_outpath)

    return

def get_met_obs_file(fpath):

    tmp = pd.read_csv(fpath, encoding = "ISO-8859-1", engine='python', 
        usecols=[0,1,2,3,4,5], parse_dates=[[1,2,3]],index_col=[0],na_values='\x97')

    return tmp

################################################################################

if __name__ == "__main__":

    if log_stout:
        fname = f'log_processing_{sitename}_{out_suffix}.txt'
        print(f'logging print statements to {fname}')

        orig_stdout = sys.stdout
        f = open(f'{projpath}/sites/{sitename}/{fname}', 'w')
        sys.stdout = f

    raw_ds, clean_ds, watch_ds, era_ds, corr_ds, forcing_ds  = main()

    if log_stout:
        sys.stdout = orig_stdout
        f.close()

    print('done!')


