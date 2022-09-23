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

sitename = 'PL-Narutowicza'
out_suffix = 'v1'
sitedata_suffix = 'v1'

local_utc_offset_hours = 1
long_sitename = 'Narutowicza Street, Łódź, Poland'
obs_contact = 'Wlodzimierz Pawlak: wlodzimierz.pawlak@geo.uni.lodz.pl, Krzysztof Fortuniak: krzysztof.fortuniak@geo.uni.lodz.pl'
obs_reference = 'Fortuniak, Pawlak and Siedlecki (2013): https://doi.org/10.1007/s10546-012-9762-1; Fortuniak, Kłysik, Siedlecki (2006): http://www.urban-climate.org/documents/ICUC6_Preprints.pdf (p64-67)'
obs_comment = 'Missing forcing filled with PL-Lipowa tower site where available. Precipitation from IMGW Łódź Lublinek.'
photo_source='Włodzimierz Pawlak'
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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=5)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['PLM00012375',   # OKECIE, PL              52.166     20.967
                      'PLM00012566',   # BALICE, PL              50.078     19.785
                      'PLM00012160',   # ELBLAG MILEJEWO, PL     54.167     19.433
                      'LOE00116364']   # ORAVSKA LESNA, LO      49.3667    19.1667

        # NO REASONABLY CLOSE GHCND SITES

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

    sitepath = siteattrs["sitepath"]

    # read data csv
    print('reading raw data file')
    raw1 = pd.read_csv(f'{datapath}/{sitename}/01_PL-LODZ_NARUTOWICZA_STREET_2005-2013_forcing_data.dat', 
                na_values='-999', delim_whitespace=True, comment='#', header=None, 
                names=['datetime','SWdown','LWdown','Tair','Qair','pressure','Wind_N','Wind_E'])
    raw2 = pd.read_csv(f'{datapath}/{sitename}/02_PL-LODZ_NARUTOWICZA_STREET_2005-2013_evaluation_data.dat', 
                na_values='-999', delim_whitespace=True, comment='#', header=None, 
                names=['datetime','SWup','LWup','Qle','Qh'])

    raw3 = pd.read_csv(f'{datapath}/{sitename}/01_PL-LODZ_NARUTOWICZA_STREET_2005-2013_forcing_data_RH.dat', 
                na_values='-999', delim_whitespace=True)

    # get times from data and reindex
    times = pd.date_range(start='2005-06-15 01:00:00', end='2013-08-09 00:00:00', freq='60Min')
    raw1.index = times
    raw2.index = times
    raw3.index = times

    # rain from IMGW Łódź Lublinek (hourly)
    rain = pd.read_csv(f'{datapath}/{sitename}/Lodz_Lublinek_2006-2015_precipitation.txt',delim_whitespace=True)
    rain_time = pd.date_range(start='2006-01-01 00:00:00', end='2015-12-31 23:00:00', freq='60Min')
    rain.index = rain_time
    rain_utc = pipeline_functions.convert_local_to_utc(rain,siteattrs['local_utc_offset_hours'])

    # limit relative humidity to 100
    rh_limited = raw3['RH'].where(raw3['RH']<=100,100)

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw1['SWdown'],
            LWdown = raw1['LWdown'],
            Tair   = raw1['Tair'] + 273.15,
            # Qair1  = raw1['Qair'], # results in values >>100
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=rh_limited,
                        temp=raw1['Tair'] + 273.15,
                        pressure=raw1['pressure']*100.
                        ), 
            PSurf  = raw1['pressure']*100.,
            Rainf  = rain_utc['precipitation']/3600.,
            Snowf  = np.nan,
            Wind_N = -raw1['Wind_E'],   # noted error in raw file, swapped columns
            Wind_E = -raw1['Wind_N'],   # noted error in raw file, swapped columns
            SWup   = raw2['SWup'],
            LWup   = raw2['LWup'],
            Qle    = raw2['Qle'],
            Qh     = raw2['Qh'],
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)
    # replace all rain flags with filled from obs (from nearby met station)
    df['Rainf_qc'] = np.where(df['Rainf'].isna(), 3, 1)

    # flag any rh_limited value that was > 100
    df['Qair_qc'] = np.where(raw3['RH']>100, 1, df['Qair_qc'])

    # use nearby tower to fill forcing where available
    fill = pd.read_csv(f'{datapath}/PL-Lipowa/01_PL-LODZ_LIPOWA_STREET_2006-2015_forcing_data.dat', 
                na_values='-999', delim_whitespace=True, comment='#', header=None, 
                names=['datetime','SWdown','LWdown','Tair','Qair','PSurf','Wind_N','Wind_E'])
    fill['Tair'] = fill['Tair']+273.15
    fill['PSurf'] = fill['PSurf']*100.
    fill_times = pd.date_range(start='2006-07-11 01:00:00', end='2015-09-25 00:00:00', freq='60Min')
    fill.index = fill_times

    for flux in ['SWdown','LWdown','Tair','Qair','PSurf']:
        tmp = fill[flux].where(df[flux].isna())
        tmp_idx = tmp.dropna().index
        df.loc[tmp_idx,f'{flux}_qc'] = 1
        df[flux] = df[flux].fillna(tmp)

    # get lipowa to check same RH values
    lip_raw3 = pd.read_csv(f'{datapath}/PL-Lipowa/01_PL-LODZ_LIPOWA_STREET_2006-2015_forcing_data_RH.dat', 
                na_values='-999', delim_whitespace=True)
    times = pd.date_range(start='2006-07-11 01:00:00', end='2015-09-25 00:00:00', freq='60Min')
    lip_raw3.index = times

    # where RH is same as at lipowa, mark as filled from obs
    sdate,edate = '2006-07-11 01:00:00','2013-08-09 00:00:00'
    df.loc[sdate:edate,'Qair_qc'] = np.where(raw3.loc[sdate:edate,'RH']==lip_raw3.loc[sdate:edate,'RH'],1,df.loc[sdate:edate,'Qair_qc'])

    # USE SUBSET ONLY (a lot of missing data in 2001)
    df = df.loc['2008':'2012']

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

