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

sitename = 'JP-Yoyogi'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 9.0
long_sitename = 'Yoyogi, Tokyo, Japan'
obs_contact = 'Hirofumi Sugawara (hiros@nda.ac.jp)'
obs_reference = 'Hirano, Sugawara, Murayama, Kondo (2015): https://doi.org/10.2151/sola.2015-024; Sugawara, Shimizu, Hirano, Murayama and Kondo (2014): https://doi.org/10.14887/kazekosymp.23.0_49; Ishidoya, Sugawara,Terao, Kaneyasu, Aoki, Tsuboi and Kondo. (2020): https://doi.org/10.5194/acp-20-5293-2020'
obs_comment = 'Observations for turbulent fluxes removed from sectors 170°-260° because of land surface inhomogeneities. Gap-filling for the forcing data using data at nearby observatories (less than 8 km distance).'
photo_source='H. Sugawara'
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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=3)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['JA000047662',        # TOKYO, JA         35.683    139.767
                      'Tokyo_monthly_fill'] # fill gaps with monthly totals difference from https://www.data.jma.go.jp/obd/stats/etrn/view/monthly_s3_en.php?block_no=47662&view=13

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
    raw1 = pd.read_csv(f'{datapath}/{sitename}/yoyogi_forcing_rev.dat', na_values=['NaN',-9999.], delim_whitespace=True)
    raw2 = pd.read_csv(f'{datapath}/{sitename}/yoyogi_evaluation.dat', na_values=['NaN',-9999.], delim_whitespace=True)

    # get times from data and reindex
    times = pd.date_range(start='2016-04-01 00:00', end='2020-03-31 23:00', freq='60Min')
    raw1.index = times
    raw2.index = times

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw1['SWdown[W/m2]'],
            LWdown = raw1['LWdown[W/m2]'],
            Tair   = raw1['Tair[dC]'] + 273.15,
            Qair   = raw1['Qa[kg/kg]'],
            PSurf  = raw1['PSurf[hPa]']*100.,
            Rainf  = raw1['Rainf[mm/hr]']/3600.,
            Snowf  = raw1['Snowf[cm/hr]']*10/3600.,
            Wind_N = raw1['Wind_N[m/s]'],
            Wind_E = raw1['Wind_E[m/s]'],
            SWup   = raw2['SWup[W/m2]'],
            LWup   = raw2['LWup[W/m2]'],
            Qle    = raw2['Qle[W/m2]'],
            Qh     = raw2['Qh[W/m2]'],
            )

    wind_dir = pd.Series(data=pipeline_functions.convert_uv_to_wdir(raw1['Wind_E[m/s]'],raw1['Wind_N[m/s]']),index=times)

    # remove wind sectors that are inhomogenous 
    df['Qle'] = df['Qle'].where(~wind_dir.between(170,260))
    df['Qh'] = df['Qh'].where(~wind_dir.between(170,260))

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    # import qc flags
    qc = pd.read_csv(f'{datapath}/{sitename}/flag_gapfill.dat', na_values=['NaN',-9999.], delim_whitespace=True)
    qc.index = times

    # replace qc flags with provided file where 1=Gap-filled using the data at nearby observatories (less than 8 km distance).
    df['SWdown_qc'] = np.where(qc['SWdown'] == 1, 1 , df['SWdown_qc'])
    df['LWdown_qc'] = np.where(qc['LWdown'] == 1, 1 , df['LWdown_qc'])
    df['Tair_qc']   = np.where(qc['Tair']   == 1, 1 , df['Tair_qc'])
    df['Qair_qc']   = np.where(qc['Qa']    == 1, 1 , df['Qair_qc'])
    df['PSurf_qc']  = np.where(qc['PSurf'] == 1, 1 , df['PSurf_qc'])
    df['Rainf_qc']  = np.where(qc['Rainf'] == 1, 1 , df['Rainf_qc'])
    df['Snowf_qc']  = np.where(qc['Snowf'] == 1, 1 , df['Snowf_qc'])
    df['Wind_N_qc'] = np.where(qc['Wind'] == 1, 1 , df['Wind_N_qc'])
    df['Wind_E_qc'] = np.where(qc['Wind'] == 1, 1 , df['Wind_E_qc'])

    # convert times
    df = pipeline_functions.convert_local_to_utc(df,siteattrs['local_utc_offset_hours'])

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
