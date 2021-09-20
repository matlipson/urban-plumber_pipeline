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

sitename = 'NL-Amsterdam'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 1.0
long_sitename = 'Amsterdam, The Netherlands'
obs_contact = 'Bert Heusinkveld (bert.heusinkveld@wur.nl) & Gert-Jan Steeneveld (gert-jan.steeneveld@wur.nl), Wageningen University'
obs_reference = 'Horst et al. (2021) in preparation'
obs_comment = 'Rainfall, air pressure and humidity observations from Schiphol Airport, with pressure corrected to tower height. Sensible and latent heat periods flagged 0 included. Given specific humidity results in RH>100, so using given RH (limited to 100) converted to Qair.'
photo_source='B. Heusinkveld'
history = 'v0.9 (2021-09-08): beta issue'

##########################################################################
# MAIN
##########################################################################

def main():

    print('calculating land cover fractions from figure')
    calc_fractions_from_figure()

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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=10)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        # start with Schiphol, as that is used as the site observed precipitation
        rain_sites = ['NLE00152485',  # SCHIPHOL, NL
                      'NLE00108974',  # AMSTERDAM, NL           52.3667     4.9167
                      'NLE00101920',  # SCHELLINGWOUDE, NL      52.3831     4.9667
                      'NLE00109368']  # ZAANDAM HEMBRUG, NL     52.4167     4.8331

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

    fpath_obs = f'{datapath}/{sitename}/Fluxdata_Amsterdam_Heusinkveld_30min_forMatthiasDemuzere.xls'

    # read data csv
    print('reading raw data file')
    raw = pd.read_excel(fpath_obs, na_values=-9999, sheet_name='Data', skiprows=[0,2])

    # get times from data and reindex
    times = pd.date_range(start='2019-01-01 00:00', end='2020-10-13 10:00', freq='30Min')
    raw.index = times
    raw.index.name='time'

    print('getting 1min radiation files')
    rad_df = combine_rad_files(siteattrs["sitepath"])
    rad_30min = rad_df.resample('30Min',closed='right',label='right').mean()

    print('getting pressure and rain from Schiphol AWS (https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens)')
    aws = get_schiphol_aws(sitedata)

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = rad_30min['SWdown'],
            LWdown = rad_30min['LWdown'],
            Tair   = raw['air_temperature'],
            # Qair   = raw['specific_humidity'], # givers RH >> 100
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=aws['RH'],
                        temp=aws['temperature'],
                        pressure=aws['corr_pressure'].loc[raw.index[0]:raw.index[-1]]),
            PSurf  = aws['corr_pressure'],    # from Schiphol
            Rainf  = aws['rain_rate'],        # from Schiphol
            Snowf  = np.nan,
            Wind_N = raw['v_unrot'],
            Wind_E = raw['u_unrot'],
            SWup   = rad_30min['SWup'],
            LWup   = rad_30min['LWup'],
            Qle    = raw['LE'].where(raw['qc_LE']==0),
            Qh     = raw['H'].where(raw['qc_H']==0),
            ###########################
            Qtau   = -raw['Tau'].where(raw['qc_Tau']==0),
            )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)
    # replace all rain & pressure flags with filled from obs (from nearby met station)
    df['PSurf_qc'] = np.where(df['PSurf_qc'].isna(),0,1)
    df['Rainf_qc'] = np.where(df['Rainf_qc'].isna(),0,1)
    df['Qair_qc'] = np.where(df['Rainf_qc'].isna(),0,1)

    # convert times
    # df = pipeline_functions.convert_local_to_utc(df,siteattrs['local_utc_offset_hours'])

    # convert pandas dataframe to xarray dataset
    obs_ds = df.to_xarray()

    return obs_ds


def get_schiphol_aws(sitedata):
    '''
    from: https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens
    HH: time (HH uur/hour, UT. 12 UT=13 MET, 14 MEZT. Hourly division 05 runs from 04.00 UT to 5.00 UT
    Temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
    Q:  Global radiation (in J/cm2) during the hourly division
    RH: Hourly sum of precipitation (in 0.1 mm) (-1 for <0.05 mm)
    P: Air pressure (in 0.1 hPa) reduced to mean sea level, at the time of observation
    U: Relative atmospheric humidity (in percents) at 1.50 m at the time of observation
    '''
    raw = pd.read_csv(f'{datapath}/{sitename}/uurgeg_240_2011-2020.txt',skiprows=33, usecols=[1,2,7,11,13,14,17], header=None, names=['YYYYMMDD','HH','T','Q','RH','P','U'])
    times = pd.date_range(start='2011-01-01 01:00', end='2021-01-01 00:00', freq='60Min')
    raw.index = times
    raw.index.name='time'
    # replace rain (RH) of -1 (<0.05mm) with half of limit (0.025 mm or 0.25 in .1mm units)
    raw['RH'].replace(-1,0.25,inplace=True)

    aws = pd.DataFrame(index=raw.index)
    aws['rain_rate'] = (raw['RH']/10)/3600 # from 0.1mm/hour
    aws['pressure'] = raw['P']*10.      # from 0.1 HPa
    aws['temperature'] = raw['T']/10 + 273.15   # from 0.1 degrees Celsius
    aws['RH'] = raw['U']

    # correct pressure for height using hypsometric equation (assuming constant temp)
    aws['corr_pressure'] = pipeline_functions.correct_pressure(  
                        ref_height=0.,
                        ref_temp=aws['temperature'],
                        ref_pressure=aws['pressure'],
                        local_temp=aws['temperature'],
                        local_height=sitedata['ground_height'] + sitedata['measurement_height_above_ground'],
                        mode=0).round(1)

    aws_30min = aws.resample('30Min').interpolate()

    return aws_30min

def combine_rad_files(sitepath):

    ############################################################
    #### COMBINES OBS OF OVERLAPPING PERIODS INTO ONE ####

    rad_files = sorted(glob.glob(f'{datapath}/{sitename}/EC_Amsterdam_Radiation_2018-2020/*'))

    names= ["RECORD","Ql_in_Avg","Ql_out_Avg","Qs_in_Avg","Qs_out_Avg","Qp_Avg","T1_Avg","T2_Avg"]
    times = pd.date_range(start='2018-05-01 11:43', end='2020-10-13 10:07', freq='1Min')
    rad_df = pd.DataFrame(columns=names,index=times)

    for fpath in rad_files:
        raw = pd.read_csv(fpath, skiprows=4,header=None, parse_dates=[0],index_col=0, names=names)
        rad_df = rad_df.fillna(raw)

    rad_df = rad_df.apply(pd.to_numeric)
    rad_df.index.name = 'Datetime'

    # ############################################################
    # # write out single file
    # formats = {
    #         'RECORD'    : '{:.0f}'.format,
    #         'Ql_in_Avg' : '{:.3f}'.format,
    #         'Ql_out_Avg': '{:.3f}'.format,
    #         'Qs_in_Avg' : '{:.3f}'.format,
    #         'Qs_out_Avg': '{:.3f}'.format,
    #         'Qp_Avg': '{:.3f}'.format,
    #         'T1_Avg': '{:.3f}'.format,
    #         'T2_Avg': '{:.3f}'.format,
    #         }

    # print('writing 30m radiation file')
    # fname = f'{sitepath}/raw_data/EC_Amsterdam_Radiation_2018-2020.csv'
    # with open(fname, 'w') as file:
    #     file.writelines(rad_df.to_string(formatters=formats))
    # ############################################################

    df = pd.DataFrame(index=times)
    df['SWdown'] = rad_df['Qs_in_Avg']
    df['SWup'] = rad_df['Qs_out_Avg']
    # correct LW for temperature of housing, per email from G-J Steenevel, 2021-03-25
    df['LWdown'] = rad_df['Ql_in_Avg'] + 5.670374419E-8*(rad_df['T1_Avg']+273.15)**4
    df['LWup']  = rad_df['Ql_out_Avg'] + 5.670374419E-8*(rad_df['T2_Avg']+273.15)**4

    return df

def calc_fractions_from_figure():
    '''
    This function calculates the land cover fractions in 8 wind sectors based on Figure 2 of 
    the draft manuscript Van der Horst describing the site.
    '''

    data = pd.read_csv(f'{datapath}/{sitename}/digitized_landcover.csv',header=None)

    alldata = {}
    data_df = pd.DataFrame()
    alldata['N']  = data.loc[0:4,1]
    alldata['NE'] = data.loc[5:9,1]
    alldata['E']  = data.loc[10:14,1]
    alldata['SE'] = data.loc[15:19,1]
    alldata['S']  = data.loc[20:24,1]
    alldata['SW'] = data.loc[25:29,1]
    alldata['W'] = data.loc[30:34,1]
    alldata['NW'] = data.loc[35:39,1]

    for key,item in alldata.items():

        # normalise
        item = item/item.iloc[-1]

        # unstack
        new_item = pd.Series(index=['buildings','water','vegetation','street','paving'],dtype=float)
        new_item.iloc[0] = item.iloc[0]
        new_item.iloc[1] = item.iloc[1] - item.iloc[0]
        new_item.iloc[2] = item.iloc[2] - item.iloc[1]
        new_item.iloc[3] = item.iloc[3] - item.iloc[2]
        new_item.iloc[4] = item.iloc[4] - item.iloc[3]

        data_df[key] = new_item

    data_df['mean'] = data_df.mean(axis=1)
    # find mean across sectors
    print(data_df.round(3))
    # data_df.round(3).to_csv('digitized_landcover_calculated.csv')

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
