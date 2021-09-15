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

sitename = 'MX-Escandon'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = -6.0
long_sitename = 'Escandon, Mexico City, Mexico'
obs_contact = 'Eric Velasco: evelasco@mce2.org, he_velasco2003@yahoo.com'
obs_reference = 'Velasco, Pressley, Allwine, Grivicke, Molina and Lamb (2011): https://doi.org/10.1007/s00704-010-0314-7; Velasco, Perrusquia, Jiménez, Hernández, Camacho, Rodríguez, Retama, Molina (2014): https://doi.org/10.1016/j.atmosenv.2014.08.018'
obs_comment = 'No LW radiation available during this period, ERA5 is used with bias-correction from 2006 data at same site. Wind direction taken from nearby site. Potential unidentified mismatch between local DST and standard times.'
photo_source='E. Velasco'
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

        print(f'creating 2006 period observational NetCDF in ALMA format\n')
        raw2006 = import_obs_2006(sitedata,siteattrs)
        raw2006.attrs['time_analysis_start'] = pd.to_datetime(raw2006.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
        raw2006.attrs['timestep_number_analysis'] = len(raw2006.time)
        raw2006 = pipeline_functions.set_variable_attributes(raw2006)
        raw2006 = pipeline_functions.set_global_attributes(raw2006,siteattrs,ds_type='raw_obs')

        print(f'writing raw observations to NetCDF\n')
        fpath = f'{sitepath}/timeseries/{sitename}_raw2006_observations.nc'
        pipeline_functions.write_netcdf_file(ds=raw2006,fpath_out=fpath)

    else:
        fpath = f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.nc'
        raw_ds = xr.open_dataset(fpath)
        # special case for Escandon (for 2006 LWdown obs)
        fpath = f'{sitepath}/timeseries/{sitename}_raw2006_observations.nc'
        raw2006 = xr.open_dataset(fpath)

    if create_rain_file:
        syear, eyear = raw_ds.time.dt.year.values[0] - 10, raw_ds.time.dt.year.values[-1]

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=7)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['MXM00076680',    # MEXICO CITY, MX       19.4000   -99.1830
                      'MXN00009015',    # RODANO 14 CFE, MX     19.4167   -99.1667
                      'MXN00009070',    # CAMPO EXP. COYOACAN   19.3500   -99.1667
                      'MXN00015127',    # TOTOLICA SAN BARTOLO  19.4500   -99.2333
                      'MXN00015058',    # MOLINITO, MX          19.4500   -99.2333
                      'MXN00015059',    # MOLINO BLANCO, MX     19.4667   -99.2167
                      'MXM00076681']    # GEOGRAFIA UNAM, MX    19.3333   -99.1167

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

    # special case for Escandon LW (2006 obs)
    pipeline_functions.compare_corrected_errors_escandon(raw2006,era_ds,watch_ds,corr_ds,lin_ds,sitename,sitepath,'all')
    
    if create_outofsample_obs:

        # clear out values in 2006 except LWdown and test out-of-sample
        LWdown2006 = raw2006.copy()
        LWdown2006 = LWdown2006.where(False)
        LWdown2006['LWdown'].values = raw2006['LWdown'].values
        _,_ = pipeline_functions.test_out_of_sample(LWdown2006,era_ds,watch_ds,sitedata,siteattrs)

    print(f'{sitename} done!')

    return raw_ds, clean_ds, watch_ds, era_ds, corr_ds, forcing_ds


##########################################################################
# specific functinos
##########################################################################

def import_obs(sitedata,siteattrs):

    fpath_obs = f'{datapath}/{sitename}/Urban_PLUMBER_Mexico_City_eddy_covariance_fluxes_update_DSTadjusted.xls'
    # fpath_obs = f'{siteattrs["sitepath"]}/raw_data/Urban_PLUMBER_Mexico_City_eddy_covariance_fluxes_update.xls'

    # read data csv
    print('reading raw data file')
    raw = pd.read_excel(fpath_obs, na_values=-999, sheet_name='30 min data', skiprows=[0])

    # get times from data and reindex
    times = pd.date_range(start='2011-06-01 11:00', end='2012-09-13 8:00', freq='30Min')
    raw.index = times

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['Kdwn (W/m2)'],
            LWdown = raw['Ldwn (W/m2)*'],
            Tair   = raw['Tair (degK)'],
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh = raw['RH (%)'].where(raw['RH (%)']>0.01)*100,
                        temp = raw['Tair (degK)'], 
                        pressure = raw['Ambient Pressure (kPa)']*1000.,
                        ),
            PSurf  = raw['Ambient Pressure (kPa)']*1000.,
            Rainf  = raw['rain (mm)']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['u (m/s)'],
                        wind_dir_from=raw['ud MER (deg) *'], # NOTE wdir taken from nearby site
                        )[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['u (m/s)'],
                        wind_dir_from=raw['ud MER (deg) *'], # NOTE wdir taken from nearby site
                        )[0],
            SWup   = raw['Kup (W/m2)'],
            LWup   = raw['Lup (W/m2)*'],
            Qle    = raw['HL (W/m2)'],
            Qh     = raw['H (W/m2)'],
            ###########################
            Qtau   = pipeline_functions.convert_ustar_to_qtau(
                        ustar = raw['u* (m/s)'],
                        temp = raw['Tair (degK)'],
                        pressure = raw['Ambient Pressure (kPa)']*1000.,
                        air_density = raw['Den. Air (kg/m3)']),
            # wdir = raw['ud (deg)'],
            # wdir_MER = raw['ud MER (deg) *'],
            )

    df = df.replace(-999,np.nan)

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    df.LWdown = np.nan # remove emperical model

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    return obs_ds


def import_obs_2006(sitedata,siteattrs):

    fpath_obs = f'{datapath}/{sitename}/Energy_fluxes_MILAGRO_2006.xls'

    # read data csv
    print('reading raw data file')
    raw = pd.read_excel(fpath_obs,na_values=-999,sheet_name='Sheet1')

    # get times from data and reindex
    times = pd.date_range(start='2006-03-17 13:00', end='2006-03-30 00:00', freq='30Min')
    raw.index = times
    raw.index.name='time'

    # create dataframe in ALMA format
    df2 = pd.DataFrame(index=times)
    df2 = df2.assign(
            SWdown = raw['Kdn'],
            LWdown = raw['Ldn'],
            Tair   = np.nan,
            Qair   = np.nan,
            PSurf  = np.nan,
            Rainf  = np.nan,
            Snowf  = np.nan,
            Wind_N = np.nan,
            Wind_E = np.nan,
            SWup   = raw['Kup'],
            LWup   = raw['Lup'],
            Qle    = raw['Qe'],
            Qh     = raw['Qh'],
            )

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df2 = pipeline_functions.convert_local_to_utc(df2,offset_from_utc)

    # convert pandas dataframe to xarray dataset
    obs_ds = df2.to_xarray()

    return obs_ds

################################################################################

################################################################################
## this plot shows misalignment between obs and ERA fro SWdown
## period 1: 2011-07-23 to 2011-09-14
## period 2: 2012-7-23 onwards
## would require deleting obs data to align, so not done

# flux = 'SWdown'

# clean = clean_ds[flux].to_series()
# era = era_ds[flux].sel(time=slice(clean.index[0],clean.index[-1])).to_series()

# months = clean.index.month.unique().tolist()
# c,e = {},{}

# for month in months:
#     c[month] = clean[clean.index.month.isin([month])].groupby(lambda x:x.time).mean()
#     e[month] = era[era.index.month.isin([month])].groupby(lambda x:x.time).mean()

# plt.close('all')
# fig,axes = plt.subplots(nrows=3,ncols=4, figsize=(12,10))

# for i,ax in enumerate(axes.flatten()):

#     c[i+1].plot(ax=ax,color='k',label='observations')
#     e[i+1].plot(ax=ax,color='r',label='era5')
#     ax.set_title(f'month {i+1}')
#     ax.set_xlabel('')

# plt.show()


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
