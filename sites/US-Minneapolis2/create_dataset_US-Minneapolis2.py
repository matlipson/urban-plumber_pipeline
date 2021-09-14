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

sitename = 'US-Minneapolis2'
out_suffix = 'v0.9'
sitedata_suffix = 'v1'

local_utc_offset_hours = -6
long_sitename = 'KUOM Tall Tower, Minneapolis, Minnesota, United States'
obs_contact = 'Joe McFadden (mcfadden@ucsb.edu) University of California, Santa Barbara'
obs_reference = 'Peters, Hiller, McFadden (2011): https://doi.org/10.1029/2010JG001463; Menzer and McFadden (2017): https://doi.org/10.1016/j.atmosenv.2017.09.049'
obs_comment = 'Recreational sectors (180°-270°) of Minneapolis KUOM tower only. Air pressure (PSurf) from Minneapolis-Saint Paul International Airport. Radiation components (SWdown,SWup,LWdown,LWup) measured at 2 m above ground level, not from tower as other variables. Tsoil at 5cm depth. Tower fluxes from original dataset which are flagged 2 excluded.'
photo_source='J. McFadden'
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

        nearest = pipeline_functions.find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=6)
        print('nearest stations, see: https://www.ncdc.noaa.gov/cdo-web/search:\n',nearest)

        rain_sites = ['USC00218450',    # UNIVERSITY OF MN ST. PAUL, MN US   44.9903   -93.1800 
                      'USC00214884']    # LOWER ST. ANTHONY FALLS, MN US     44.9783   -93.2469

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

    fpath_obs = f'{datapath}/US-Minneapolis/kuom_data_extract_20150930.txt'

    # read data csv
    print('reading raw data file')
    raw = pd.read_csv(fpath_obs, delim_whitespace=True, skipfooter=1378, engine='python')

    # get times from data and reindex
    times = pd.date_range(start='2005-11-27 00:00', end='2009-05-29 7:00', freq='30Min')
    raw.index = times

    # # gives RH>100
    # esat = pipeline_functions.calc_esat(
    #             temp = raw['Tair_40m_kuom']+273.15,
    #             pressure = raw['Pair_kuom']*1000.,
    #             mode=0)
    # rh = 100 * (esat - raw['vpd_40m_kuom']*1000.)/esat

    # # LiDAR cloud of building heights
    # bh = pd.read_csv('raw_data/KUOM_lidar_building_height_above_ground_in_footprint.txt',
    #         header=None,names=['height']).sort_values(by='height',ignore_index=True)

    rho_air = pipeline_functions.calc_density(
                    temp=raw['Tair_40m_kuom']+273.15,
                    pressure=raw['Pair_kuom']*1000.)

    qair = (raw['h2o_40m_kuom']/1000)/rho_air

    # create dataframe in ALMA format
    df = pd.DataFrame(index=times)
    df = df.assign(
            SWdown = raw['Rs_in_turf'], # measured at 2m agl
            LWdown = raw['RL_in_turf'], # measured at 2m agl
            Tair   = raw['Tair_40m_kuom']+273.15,
            Qair   = pipeline_functions.convert_rh_to_qair(
                        rh=raw['rh_40m_kuom'],
                        temp=raw['Tair_40m_kuom']+273.15,
                        pressure=raw['Pair_kuom']*1000.).where(raw['rh_40m_kuom']<=100),
            # Qair   = (raw['h2o_40m_kuom']/1000)/rho_air, # wv concentration / air density
            PSurf  = raw['Pair_kuom']*1000.,
            Rainf  = raw['precip_turf']/1800.,
            Snowf  = np.nan,
            Wind_N = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['ws_40m_kuom'],
                        wind_dir_from=raw['wd_40m_kuom'], 
                        )[1],
            Wind_E = pipeline_functions.convert_wdir_to_uv(
                        speed=raw['ws_40m_kuom'],
                        wind_dir_from=raw['wd_40m_kuom'],
                        )[0],
            SWup   = raw['Rs_out_turf'], # measured at 2m agl
            LWup   = raw['RL_out_turf'], # measured at 2m agl
            Qle    = raw['LE_40m_kuom'].where(raw['qaqc_wh2o_40m_kuom']!=2), # remove data flagged 2
            Qh     = raw['H_40m_kuom'].where(raw['qaqc_wTs_40m_kuom']!=2),   # remove data flagged 2
            #######
            Qg = raw['G_turf'], # Ground heat flux in non-irrigated open turfgrass
            Qtau = pipeline_functions.convert_ustar_to_qtau(
                ustar=raw['ustar_40m_kuom'],
                temp=raw['Tair_40m_kuom']+273.15,
                pressure=raw['Pair_kuom']*1000.,
                air_density=rho_air).where(raw['qaqc_uw_40m_kuom']!=2),    # remove data flagged 2
            # Tair2m=raw['Tair_135cm_turf']+273.15,
            SoilTemp=raw['Tsoil_5cm_gf_turf']+273.15 # 5cm below ground
            )

    '''
    From Menzer et al. 2017
    The residential sector was defined to include observations to the northeast (wind direction between 0+ and 75+) and to the northwest of the tower (wind direction between 285+ and 360+). 
    The recreational sector included observations from the southwest over the nearby golf course, spanning wind directions from 195 to 270 .
    '''

    analysis_fluxes = ['Qh','Qle','Qtau']
    wind_dir = pd.Series(data=raw['wd_40m_kuom'],index=times)
    # remove residential wind sectors
    df[analysis_fluxes] = df[analysis_fluxes].where(~wind_dir.between(0,180))
    df[analysis_fluxes] = df[analysis_fluxes].where(~wind_dir.between(270,360))

    # reweight surface cover fractions to golf course
    wd = raw.where(wind_dir.between(180,270))

    weights = ['fp_K2d_frac_tree_dec','fp_K2d_frac_tree_eg','fp_K2d_frac_grass_irr','fp_K2d_frac_grass_nirr','fp_K2d_frac_impervious','fp_K2d_frac_water_pool','fp_K2d_frac_water_open','fp_K2d_frac_outside']
    print(wd[weights].describe())

    wd['trees'] = (wd['fp_K2d_frac_tree_dec'] + wd['fp_K2d_frac_tree_eg'])
    wd['grass'] = (wd['fp_K2d_frac_grass_irr'] + wd['fp_K2d_frac_grass_nirr']) 
    wd['water'] = (wd['fp_K2d_frac_water_pool'] + wd['fp_K2d_frac_water_open']) 
    wd['impervious'] = wd['fp_K2d_frac_impervious']

    fractions = wd[['trees','grass','water','impervious']].copy()

    fractions = fractions.divide(fractions.sum(axis=1),axis=0)

    print(fractions.mean().round(2))

    # remove early subset without radiation obs
    df = df.loc['2006-06-01 12:00':]

    # create qc flags, with 0=observed, 1=gap-filled by obsevations, 2=gap-filled by era-derived, 3=missing
    for key in df.columns:
        df[f'{key}_qc'] = np.where(df[key].isna(), 3, 0)

    df[f'PSurf_qc'] = np.where(df[f'PSurf_qc'].isna(), 3, 1)

    # convert times
    offset_from_utc = siteattrs['local_utc_offset_hours']
    df = pipeline_functions.convert_local_to_utc(df,offset_from_utc)

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
