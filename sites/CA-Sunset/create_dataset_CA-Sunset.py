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

sitename = 'CA-Sunset'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = -8
long_sitename = 'Sunset, Vancouver, Canada'
obs_contact = 'Andreas Christen (andreas.christen@meteo.uni-freiburg.de) Albert-Ludwigs-Universitaet Freiburg'
obs_reference = 'Christen, Coops, Crawford, Kellett, Liss, Olchovski, Tooke, van der Laan, Voogt (2011): https://doi.org/10.1016/j.atmosenv.2011.07.040; Crawford and Christen (2015): https://doi.org/10.1007/s00704-014-1124-0'
obs_comment = 'Gapfilling for SWdown, LWdown, PSurf by Oliver Michels based on local observations and regressions. SoilTemp at 5cm depth.'
photo_source='A. Christen'
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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=14)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['CA001108463',    # VANCOUVER OAKRIDGE, BC CA       9.2333  -123.1167
                      'CA001106764',    # RICHMOND DALLYN 2, CA           49.1833  -123.0833
                      'CA001106PF7',    # RICHMOND NATURE PARK, CA        49.1667  -123.1000
                      'CA001108446',    # VANCOUVER HARBOUR CS            49.3000  -123.1167
                      'CA001108447']    # VANCOUVER INTERNATIONAL A, CA   49.2000  -123.1833

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

    print('reading raw EC files')

    tmp_list = []
    fpaths = sorted(glob.glob(f'{datapath}/{sitename}/EC/Vancouver-Sunset-EC*'))
        
    for fpath_obs in fpaths:
        print('loading',fpath_obs.split('/')[-1])
        # read data csv
        tmp = pd.read_csv(fpath_obs, comment='#', skipinitialspace=True)
        tmp_list.append(tmp)
        
    raw_ec = pd.concat(tmp_list)

    times = pd.date_range(start='2008-05-01 00:30:00',end='2017-10-01 00:00:00',freq='30Min')
    raw_ec.index = times

    # 00 STUST1 Vancouver Sunset (Tower) : Friction Velocity  28.80m [m / s]
    # 01 STTKE1 Vancouver Sunset (Tower) : Turbulent Kinetic Energy  28.80m [m2 / s2]
    # 02 STQHB1 Vancouver Sunset (Tower) : Sensible Heat Flux (Best available)  28.80m [W / m2]
    # 03 STQEB1 Vancouver Sunset (Tower) : Latent Heat Flux (Best available)  28.80m [W / m2]
    # 04 STFCB1 Vancouver Sunset (Tower) : Carbon Dioxide Flux (Best available)  28.80m [micromol / (m2 s)]    
    # 05 STUSP1 Vancouver Sunset (Tower) : Wind u-Component (Internal) No. of Spikes  28.80m [Count]
    # 06 STVSP1 Vancouver Sunset (Tower) : Wind v-Component (Internal) No. of Spikes  28.80m [Count]
    # 07 STWSP1 Vancouver Sunset (Tower) : Wind w-Component (Internal) No. of Spikes  28.80m [Count]
    # 08 STTSP1 Vancouver Sunset (Tower) : Acoustic Temperature No. of Spikes  28.80m [Count]
    # 09 STHSP1 Vancouver Sunset (Tower) : Water Vapour No. of Spikes  28.80m [Count]
    # 10 STCSP1 Vancouver Sunset (Tower) : Carbon Dioxide Molar Density No. of Spikes  28.80m [Count]
    # 11 STNST1 Vancouver Sunset (Tower) : Number of Samples (including invalid samples)  28.80m [No]
    # 12 STPQF1 Vancouver Sunset (Tower) : Flux Processing Quality Flag  28.80m [Code]

    print('reading raw MET files')
    tmp_list = []
    fpaths = sorted(glob.glob(f'{datapath}/{sitename}/MET/Vancouver-Sunset-MET*'))
    for fpath_obs in fpaths:
        print('loading',fpath_obs.split('/')[-1])
        # read data csv
        tmp = pd.read_csv(fpath_obs, comment='#', skipinitialspace=True)
        tmp_list.append(tmp)
    raw_met = pd.concat(tmp_list)

    times = pd.date_range(start='2008-05-01 00:05:00',end='2017-10-01 00:00:00',freq='5Min')
    raw_met.index = times

    raw_met_30min = raw_met.resample('30Min',closed='right',label='right').mean()

    # accumulation of rainfall over half hour from 5 min bins
    raw_met_30min['STPCT1'] = raw_met_30min['STPCT1']*6.
    
    # 00 STWVA1 Vancouver Sunset (Tower) : Wind Velocity (Horizontal Scalar Mean)  28.80m [m / s]
    # 01 STWVV1 Vancouver Sunset (Tower) : Wind Velocity (Horizontal Vector Mean)  28.80m [m / s]
    # 02 STWDA1 Vancouver Sunset (Tower) : Wind Direction  28.80m [degdeg]
    # 03 STICI1 Vancouver Sunset (Tower) : Wind Inclination (from Horiz. Plane)  28.80m [degdeg]
    # 04 STSDA2 Vancouver Sunset (Tower) : Shortwave Irradiance  26.20m [W / m2]
    # 05 STSUA2 Vancouver Sunset (Tower) : Shortwave Reflectance  26.20m [W / m2]
    # 06 STLDA2 Vancouver Sunset (Tower) : Longwave Downward Radiation  26.20m [W / m2]
    # 07 STLUA2 Vancouver Sunset (Tower) : Longwave Upward Radiation  26.20m [W / m2]
    # 08 STNRA2 Vancouver Sunset (Tower) : Net All-Wave Radiation  26.20m [W / m2]
    # 09 STPSA1 Vancouver Sunset (Tower) : Barometric Pressure  28.80m [kPa]
    # 10 STDTA1 Vancouver Sunset (Tower) : Air Temperature  26.00m [degC]
    # 11 STRHA1 Vancouver Sunset (Tower) : Relative Humidity  26.00m [%]
    # 12 SSSTA4 Vanocuver Sunset (SS4)   : Soil Temperature  -0.05m [degC]
    # 13 STPCT1 Vancouver Sunset (Tower) : Precipitation  1.00m [mm]
    # 14 STCVA1 Vancouver Sunset (Tower) : Carbon Dioxide Molar Mixing Ratio  28.80m [ppm]
    # 15 STCVN1 Vancouver Sunset (Tower) : Carbon Dioxide Molar Mixing Ratio Minimum  28.80m [ppm]
    # 16 STCVX1 Vancouver Sunset (Tower) : Carbon Dioxide Molar Mixing Ratio Maximum  28.80m [ppm]

    # create dataframe in ALMA format
    df = pd.DataFrame()
    df = df.assign(
            SWdown = raw_met_30min['STSDA2'],
            LWdown = raw_met_30min['STLDA2'],
            Tair   = raw_met_30min['STDTA1']+273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=raw_met_30min['STRHA1'],
                        temp=raw_met_30min['STDTA1']+273.15,
                        pressure=raw_met_30min['STPSA1']*1000.),
            PSurf  = raw_met_30min['STPSA1']*1000.,
            Rainf  = raw_met_30min['STPCT1']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw_met_30min['STWVA1'],
                        wind_dir_from=raw_met_30min['STWDA1'])[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw_met_30min['STWVA1'],
                        wind_dir_from=raw_met_30min['STWDA1'])[0],
            SWup   = raw_met_30min['STSUA2'],
            LWup   = raw_met_30min['STLUA2'],
            Qle    = raw_ec['STQEB1'],
            Qh     = raw_ec['STQHB1'],
            ####
            Qtau   = pipeline_functions.convert_ustar_to_qtau(
                        ustar = raw_ec['STUST1'],
                        temp = raw_met_30min['STDTA1']+273.15,
                        pressure = raw_met_30min['STPSA1']*1000.,
                        air_density =  pipeline_functions.calc_density(
                                temp= raw_met_30min['STDTA1']+273.15,
                                pressure= raw_met_30min['STPSA1']*1000.
                                )
                        ),
            SoilTemp = raw_met_30min['SSSTA4']+273.15 # at 5cm
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    #### GAP FILLED DATA ####
    fpath = f'{datapath}/{sitename}/Gap-Filled/VS_Gapfilled_2008_to_2017.csv'
    gapfilled = pd.read_csv(fpath)
    times = pd.date_range(start='2008-05-06 17:00',end='2017-05-04 13:00',freq='5Min')
    gapfilled.index = times
    gapfilled_30min = gapfilled.resample('30Min',closed='right',label='right').mean()
    # accumulation of rainfall over half hour from 5 min bins
    gapfilled_30min['rain'] = gapfilled_30min['rain']*6.

    fill = pd.DataFrame()
    fill['SWdown'] = gapfilled_30min['kdown']
    fill['LWdown'] = gapfilled_30min['ldown']
    fill['Tair']   = gapfilled_30min['Tair']+273.15
    fill['Qair']   = pipeline_functions.convert_rh_to_qair(
                        rh=gapfilled_30min['RH'],
                        temp=gapfilled_30min['Tair']+273.15,
                        pressure=gapfilled_30min['pres']*100.)
    fill['PSurf']  = gapfilled_30min['pres']*100.
    fill['Rainf']  = gapfilled_30min['rain']/1000
    fill['Wind_N'] = pipeline_functions.convert_wdir_to_uv(
                        speed=gapfilled_30min['U'].where(gapfilled_30min['U']>0.01),
                        wind_dir_from=raw_met_30min['STWDA1'])[1]
    fill['Wind_E'] = pipeline_functions.convert_wdir_to_uv(
                        speed=gapfilled_30min['U'].where(gapfilled_30min['U']>0.01),
                        wind_dir_from=raw_met_30min['STWDA1'])[0]

    # 00 ‘iy’ year
    # 01 ‘id’ DOY
    # 02 ‘it’ hour
    # 03 ‘iy’ minute
    # 04 ‘U’ Vancouver Sunset (Tower) : Wind Velocity (Horizontal, Scalar Mean)  26.30m [m / s]
    # 05 ‘Tair’ Vancouver Sunset (Tower) : Air Temperature  26.00m [degC]
    # 06 ‘RH’ Vancouver Sunset (Tower) : Relative Humidity  26.00m [%]
    # 07 ‘pres’ Vancouver Sunset (Tower) : Barometric Pressure  28.80m [kPa]
    # 08 ‘rain’ Vancouver Sunset (Tower) : Precipitation  1.00m [mm]
    # 09 ‘kdown’ Vancouver Sunset (Tower) : Shortwave Irradiance  26.20m [W / m2]
    # 10 ‘ldown’ Vancouver Sunset (Tower) : Longwave Downward Radiation  26.20m [W / m2]

    # use gap-filled data from Oliver Michels where available
    print('using gap-filled data from Oliver Michels for SWdown, LWdown, Tair, PSurf')
    for flux in ['SWdown','LWdown','Tair','PSurf']:
        tmp = fill[flux].where(df[flux].isna())
        tmp_idx = tmp.dropna().index
        df.loc[tmp_idx,f'{flux}_qc'] = 1
        df[flux] = df[flux].fillna(tmp)

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

    df = df.loc['2012':'2016']
    # convert pandas dataframe to xarray dataset
    df.index.name='time'
    obs_ds = df.to_xarray()

    # for flux in ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Wind_N','Wind_E']:
    # # for flux in ['SWdown']:
    #     plt.close('all')
    #     fig, ax = plt.subplots(figsize=(10,5))

    #     sdate,edate = df.index[0],df.index[-1]

    #     era_ds[flux].to_series()[sdate:edate].plot(ax=ax,color='g',label='era5 uncorrected')
    #     corr_ds[flux].to_series()[sdate:edate].plot(ax=ax,color='r',label='era5 corrected')
    #     df[flux].plot(ax=ax,color='royalblue',label='filled')
    #     clean_ds[flux].to_series().plot(ax=ax,color='k',label='clean obs')
    #     plt.legend()

    #     plt.title(flux)
    #     plt.show()

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

