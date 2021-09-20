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

sitename = 'FR-Capitole'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = 1
long_sitename = 'Capitole district, Toulouse, France'
obs_contact = 'Valéry Masson (valery.masson@meteo.fr)'
obs_reference = 'Masson, Gomes, Pigeon, Liousse, Pont, Lagouarde, Voogt, Salmond, Oke, Hidalgo, Legain, Garrouste, Lac, Connan, Briottet, Lachérade, Tulet (2008): https://doi.org/10.1007/s00703-008-0289-4; Goret, Masson, Schoetter, Moine (2019): https://doi.org/10.1016/j.aeaoa.2019.100042'
obs_comment = 'Observation height varied depending on windspeed. All observations at lowest tower position (~27m) excluded, with wind speed corrected using log laws, following Goret et al. (2019) https://doi.org/10.1016/j.aeaoa.2019.100042. Gap-filling from local observations by CNRM. ERA5 snowfall set to zero during analysis period per advice from CNRM.'
photo_source='V. Masson'
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

        # rain_sites = ['FR000007630']   # TOULOUSE BLAGNAC, FR       43.6208     1.3789
        rain_sites = ['FR000007630']   # TOULOUSE BLAGNAC, FR       43.6208     1.3789
                      
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

    # MTO_MAT and FLUX_CNRM files from  https://www.aeris-data.fr/catalogue/
    ############################################################################

    # times are in UTC (according to header), and timestamp is finishing block
    # this is checked by resampling 1min RN-ray (in FLUX_NCRM) to end on half hour and comparing with block averaged RN (=SWdown + LWdown - SWup - LWup in MTO_MAT)
    # those timestep and values then match
    # forcing, on the other hand, is 1/2 hour ahead of obs, so must be in local time (+1 UTC) AND be for the start of the averaging period

    # MTO_MAT: 
    col_names = ['date','time','air_pressure','air_temperature','relative_humidity','SWdown','SWup','LWdown','LWup','rain_rate_optical','rain_rate_bucket','wind_dir_top','wind_speed_top','wind_dir_intermediate','wind_speed_intermediate','co2','tower_position']
    #          JJ/MM/AAAA' 'HHMMSS.SSS UTC'    ''  'celsius'   'pourcent'      'watts/m2' watts/m2''watts/m2' 'watts/m2''millimeter/hour''mm/h''degree'    'meter/second' 'degree'  'meter/second''g/m3'  '1=full extension, 2=intermediate extension, 3=down' 

    raw_list = []

    fnames = [f'Ta2004{str(month).zfill(2)}_MAT_%60.asc' for month in range(2,13)]
    for fname in fnames:
        raw_list.append(pd.read_csv(f'{datapath}/{sitename}/MTO_MAT/{fname}', na_values='+9999', delim_whitespace=True, 
            header=None, names=col_names))
    fnames = [f'Ta2005{str(month).zfill(2)}_MAT_%60.asc' for month in range(1,3)]
    for fname in fnames:
        raw_list.append(pd.read_csv(f'{datapath}/{sitename}/MTO_MAT/{fname}', na_values='+9999', delim_whitespace=True, 
            header=None, names=col_names))

    raw1_1min = pd.concat(raw_list)

    times_1min = pd.date_range(start='2004-02-20 00:01:00', end='2005-03-01 00:00:00', freq='1Min')
    raw1_1min.index = times_1min

    raw1 = raw1_1min.resample('30Min',closed='right',label='right').mean()

    raw1['tower_position'] = raw1_1min['tower_position'].resample('30Min',closed='right',label='right').agg(lambda x:x.mode())
    raw1['tower_position'] = pd.to_numeric(raw1['tower_position'],errors='coerce')
    raw1['tower_position'] = raw1['tower_position'].fillna(9999)
    raw1['tower_position'] = raw1['tower_position'].astype(int)

    raw1['tower_height'] = raw1.apply(lambda row:set_height_correct(row), axis=1)
    
    ############################################################################

    # FLUX_CNRM: Ta_flx_V9
    col_names = ['DATE','HEURE','CO2-flx','HS_TAIR_HAUT-flx','HS_TSON_HAUT-flx','LE_TAIR_HAUT-flx','USTAR_HAUT-vit','HS_TSON_BAS-flx','USTAR_BAS-vit','RN-ray','RES_HAUT-flx','CO2-con']
    #       'JJ/MM/AAAA''HHMMSS.SSS''mg.m-2.s-1' 'W/m2'            'W/m2'               'W/m2'           'm/s'            'W/m2'           'm/s'       'W/m2'     'W/m2'      'gram/m3'
    #                                          'high'              'high'               'high'           'high'           'low'            'low'                  'high'     
    raw_list = []

    fnames = [f'Ta040{str(doy).zfill(3)}_flx1800-dat.txt' for doy in range(51,367)]
    for fname in fnames:
        raw_list.append(pd.read_csv(f'{datapath}/{sitename}/FLUX_CNRM/Ta_flx_V9/{fname}', na_values='+9999', delim_whitespace=True, 
            header=None, names=col_names))

    fnames = [f'Ta050{str(doy).zfill(3)}_flx1800-dat.txt' for doy in range(1,60)]
    for fname in fnames:
        raw_list.append(pd.read_csv(f'{datapath}/{sitename}/FLUX_CNRM/Ta_flx_V9/{fname}', na_values='+9999', delim_whitespace=True, 
            header=None, names=col_names))

    raw2 = pd.concat(raw_list)

    # get times from data and reindex
    times = pd.date_range(start='2004-02-20 00:30:00', end='2005-03-01 00:00:00', freq='30Min')
    raw2.index = times

    ############################################################################
    # observations at measurement height

    # create dataframe in ALMA format
    obs = pd.DataFrame(index=times)
    obs = obs.assign(
            SWdown = raw1['SWdown'],
            LWdown = raw1['LWdown'],
            Tair   = raw1['air_temperature']+273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=raw1['relative_humidity'],
                        temp=raw1['air_temperature']+273.15,
                        pressure=raw1['air_pressure']*100.),
            PSurf  = raw1['air_pressure']*100.,
            Rainf  = raw1['rain_rate_optical']/3600.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed = raw1['wind_speed_top'],
                        wind_dir_from = raw1['wind_dir_top'])[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed = raw1['wind_speed_top'],
                        wind_dir_from = raw1['wind_dir_top'])[0],
            SWup   = raw1['SWup'],
            LWup   = raw1['LWup'],
            Qle    = raw2['LE_TAIR_HAUT-flx'],
            Qh     = raw2['HS_TSON_HAUT-flx'],
            #####
            Qtau   = pipeline_functions.convert_ustar_to_qtau(
                        ustar=raw2['USTAR_HAUT-vit'],
                        temp=raw1['air_temperature']+273.15,
                        pressure=raw1['air_pressure']*100.,
                        air_density=pipeline_functions.calc_density(
                            temp=raw1['air_temperature']+273.15,
                            pressure=raw1['air_pressure']*100.)),
            )

    cnrm_forcing = pd.DataFrame(index=times)

    DIR_SW = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_DIR_SW.txt',header=None)
    SCA_SW = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_SCA_SW.txt',header=None)
    LW = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_LW.txt',header=None)
    PS = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_PS.txt',header=None)
    QA = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_QA.txt',header=None)
    TA = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_TA.txt',header=None)
    RAIN = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_RAIN.txt',header=None)
    WIND = pd.read_csv(f'{datapath}/{sitename}/Forc_CAPITOUL/Forc_WIND.txt',header=None)

    cnrm_forcing['SWdown'] = DIR_SW.values + SCA_SW.values
    cnrm_forcing['LWdown'] = LW.values
    cnrm_forcing['PSurf'] = PS.values
    cnrm_forcing['Qair'] = QA.values
    cnrm_forcing['Tair'] = TA.values
    cnrm_forcing['Rainf'] = RAIN.values
    cnrm_forcing['Wind'] = WIND.values
    cnrm_forcing['Wind_N'] = pipeline_functions.convert_wdir_to_uv(
                        speed = cnrm_forcing['Wind'],
                        wind_dir_from = raw1['wind_dir_top'])[1]
    cnrm_forcing['Wind_E'] = pipeline_functions.convert_wdir_to_uv(
                        speed = cnrm_forcing['Wind'],
                        wind_dir_from = raw1['wind_dir_top'])[0]

    # convert times
    cnrm_forcing = pipeline_functions.convert_local_to_utc(cnrm_forcing,0.5)

    ######################
    # HEIGHT CORRECTIONS
    # These observations are taken at 4 different heights (see appendix D of  Goret et al. (2019) https://doi.org/10.1016/j.aeaoa.2019.100042)

    corr = obs.copy()
    
    # remove analysis variables where tower position is down (=4) or unknown (=9999)
    corr = corr.assign(
                SWup = corr.SWup.where(raw1['tower_position'].isin([1,2,3])),
                LWup = corr.LWup.where(raw1['tower_position'].isin([1,2,3])),
                Qle  = corr.Qle.where(raw1['tower_position'].isin([1,2,3])), 
                Qh   = corr.Qh.where(raw1['tower_position'].isin([1,2,3])),
                Qtau = corr.Qtau.where(raw1['tower_position'].isin([1,2,3])),
                Wind_N = pipeline_functions.correct_wind(
                        ref_wind=obs['Wind_N'],
                        local_z0=sitedata['roughness_length_momentum'],
                        local_d0=sitedata['displacement_height'],
                        local_wind_hgt=48.05,
                        ref_wind_hgt=raw1['tower_height'],
                        ref_z0=sitedata['roughness_length_momentum'],
                        ref_d0=sitedata['displacement_height'],
                        mode=0).where(raw1['tower_position'].isin([1,2,3])), # log law with displacement
                Wind_E = pipeline_functions.correct_wind(
                        ref_wind=obs['Wind_E'],
                        local_z0=sitedata['roughness_length_momentum'],
                        local_d0=sitedata['displacement_height'],
                        local_wind_hgt=48.05,
                        ref_wind_hgt=raw1['tower_height'],
                        ref_z0=sitedata['roughness_length_momentum'],
                        ref_d0=sitedata['displacement_height'],
                        mode=0).where(raw1['tower_position'].isin([1,2,3])), # log law with displacement
                )

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in corr.columns:
        corr[f'{key}_qc'] = np.where(corr[key].isna(), 3, 0)

    # for tower position 2 (mid), 3 (mid), 4 (low), set qc flag to 'filled from obs'
    corr['Wind_N_qc'] = np.where(raw1['tower_position'].isin([2,3,4]), 1, corr['Wind_N_qc'])
    corr['Wind_E_qc'] = np.where(raw1['tower_position'].isin([2,3,4]), 1, corr['Wind_E_qc'])
    # reset flag for missing
    corr['Wind_N_qc'] = np.where(corr['Wind_N'].isna(), 3, corr['Wind_N_qc'])
    corr['Wind_E_qc'] = np.where(corr['Wind_N'].isna(), 3, corr['Wind_E_qc'])

    # fill remaining gaps with CNRM forcing
    for flux in ['SWdown','LWdown','PSurf','Qair','Tair','Rainf','Wind_N','Wind_E']:
        tmp = cnrm_forcing[flux].where(corr[flux].isna())
        tmp_idx = tmp.dropna().index
        corr.loc[tmp_idx,f'{flux}_qc'] = 1
        corr[flux] = corr[flux].fillna(tmp)

    # OTHER VARIABLES APPEAR NOT TO REQUIRE CORRECTION e.g. makes pressure jump, not corrected in original forcing

    # corr['Tair'] = pipeline_functions.correct_temperature(
    #                     ref_height=raw1['tower_height'],
    #                     ref_temp=raw1['air_temperature']+273.15,
    #                     local_height=48.05)

    # corr['PSurf'] = pipeline_functions.correct_pressure(
    #                     ref_height=raw1['tower_height'],
    #                     ref_temp=raw1['air_temperature']+273.15,
    #                     ref_pressure=raw1['air_pressure']*100.,
    #                     local_temp=corr['Tair'],
    #                     local_height=48.05,
    #                     mode=1)

    # corr['Qair'] = pipeline_functions.correct_humidity(
    #                     ref_temp=obs['Tair'],
    #                     ref_pressure=obs['PSurf'],
    #                     ref_qair=obs['Qair'],
    #                     local_temp=corr['Tair'],
    #                     local_pressure=corr['PSurf'],
    #                     mode=0)

    # corr['LWdown'] = pipeline_functions.correct_lwdown(
    #                     ref_lwdown=obs['LWdown'],
    #                     ref_temp=obs['Tair'],
    #                     ref_pressure=obs['PSurf'],
    #                     ref_qair=obs['Qair'],
    #                     local_temp=corr['Tair'],
    #                     local_pressure=corr['PSurf'],
    #                     local_qair=corr['Qair'])


    ### Already in UTC
    ### convert times
    # corr = pipeline_functions.convert_local_to_utc(corr,siteattrs['local_utc_offset_hours'])

    # convert pandas dataframe to xarray dataset
    corr.index.name='time'
    corr_ds = corr.to_xarray()

    return corr_ds

def plot_SW_diurnal():

    sdate_sub, edate_sub= forcing_ds.time_analysis_start,forcing_ds.time_coverage_end
    flux = 'SWdown'

    plt.close('all')
    fig,ax = plt.subplots(figsize=(10,5))

    # obs_day = obs[flux]['2004-09'].groupby(lambda x: x.time).mean()
    # fill_day = cnrm_forcing[flux]['2004-09'].groupby(lambda x: x.time).mean()
    # era_day = era_ds[flux].sel(time=slice(sdate_sub,edate_sub)).squeeze().to_series()['2004-09'].groupby(lambda x: x.time).mean()

    obs_day = obs[flux]['2004-07-28'].resample('1H',closed='right',label='right').mean()
    fill_day = cnrm_forcing[flux]['2004-07-28'].resample('1H',closed='right',label='right').mean()
    era_day = era_ds[flux].sel(time=slice(sdate_sub,edate_sub)).squeeze().to_series()['2004-07-28']

    obs_day.plot(ax=ax,label='obs',color='k')
    fill_day.plot(ax=ax,label='fill',color='g')
    era_day.plot(ax=ax,label='era',color='r')
    ax.set_title('SWdown on clear day (UTC) with solar noon marked')
    ax.axvline(pd.Timestamp('2004-07-28 12:00'),color='k',lw=1)
    plt.legend()
    plt.show()


def plot_obs_vs_forcing():

    sdate_sub, edate_sub= forcing_ds.time_analysis_start,forcing_ds.time_coverage_end

    forcing_ds['Wind'] = forcing_ds['Wind_N']
    forcing_ds['Wind_qc'] = forcing_ds['Wind_N_qc']
    forcing_ds['Wind'].values = np.sqrt(forcing_ds['Wind_N']**2 + forcing_ds['Wind_E']**2)
    forcing_ds['Wind'].attrs['long_name'] = 'Wind speed'


    for flux in cnrm_forcing.columns:

        plt.close('all')
        fig,ax = plt.subplots(figsize=(10,5))

        forcing_ds[flux].where(forcing_ds[f'{flux}_qc']==0).sel(time=slice(sdate_sub,edate_sub)).plot(
            ax=ax,color='k',label='orig obs',lw=1,ms=2,marker='o')
        forcing_ds[flux].where(forcing_ds[f'{flux}_qc']==1).sel(time=slice(sdate_sub,edate_sub)).plot(
            ax=ax,color='b',label='filled from obs',lw=1,ms=2,marker='o')
        forcing_ds[flux].where(forcing_ds[f'{flux}_qc']==2).sel(time=slice(sdate_sub,edate_sub)).plot(
            ax=ax,color='r',label='filled from era',lw=1,ms=2,marker='o')

        cnrm_forcing.loc[sdate_sub:edate_sub,flux].plot(
            ax=ax,color='g',label='orig cnrm forcing',lw=1,ms=2,marker='o',alpha=0.3)

        ax.legend(loc='upper right',fontsize=7)
        ax.set_title('CNRM and Urban-PLUMBER forcing for FR-Capitole')

        plt.show()
        fig.savefig(f'{sitepath}/processed/{flux}_forcing_comparison.png',bbox_inches='tight',dpi=150)

    return

def set_height_correct(row):

    '''NOTE flags in "README_format_MTO-MAT.en.html" for tower_position are WRONG.
    Have notified Masson (2021-03-09). Actual flags follow Goret et al. (2019)

    MTO_MAT   |  Goret et al.  |  height (m)  |   %    |
    ----------------------------------------------------
    1  full   |  1             |   48.05      | 39.67% |
    2  mid    |  2             |   38.98      | 3.46%  |
    -         |  3             |   38.23      | 35.02% |
    3  down   |  4             |   26.93      | 20.88% |
    +9999     |  n/a           |   ?????      | 0.96%  |
    '''

    if row['tower_position'] == 1:
        return 48.05
    elif row['tower_position'] == 2:
        return 38.98
    elif row['tower_position'] == 3:
        return 38.23
    elif row['tower_position'] == 4:
        return 26.93
    else:
        return np.nan


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

