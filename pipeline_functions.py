'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

__title__ = "Pipeline functions for processing site information in the Urban-PLUMBER project"
__version__ = "2021-09-20"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__description__ = 'When run on NCI:Gadi this script collects ERA5 and WFDE5 surface data and applies correction based on observations. Output intended for driving land surface models'

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import os
import sys
import glob
from datetime import datetime
from collections import OrderedDict
import statsmodels.api as sm

import qc_observations as qco

era5path = '/g/data/rt52/era5/single-levels/reanalysis'
wfdepath = '/g/data/rt52/era5-derived/wfde5/v1-1/cru'

##########################################################################
# USER INPUTS
##########################################################################

missing_float = -9999.
missing_int8  = -128
wind_hgt      = 10.     # era5 wind height variable (100m has poor diurnal pattern, use 10m)
img_fmt       = 'png'   # image format

##########################################################################
# main
##########################################################################

def main(datapath,sitedata,siteattrs,raw_ds,fullpipeline=True,qcplotdetail=False):
    '''
    This function is called from individual site scripts "create_dataset_{sitename}.py"
    Main sections are:
    1. CLEAN SITE OBSERVATIONS
    2. EXTRACT WFDE5
    3. EXTRACT ERA5
    4. CORRECT ERA5
    5. CORRECT ERA5 USING LINEAR REGRESSION
    6. FILL FORCING VARIABLES
    7. PREPEND FORCING VARIABLES WITH ERA5
    8. CREATE ANALYSIS FILES
    '''

    extract_wfde5 = False
    extract_era5 = False

    if fullpipeline:
        clean_obs_nc          = True
        correct_era5          = True
        correct_era5_linear   = True
        create_filled_forcing = True
        create_analysis       = True
        write_to_text         = True # can also use seperate program if necessary (convert_nc_to_text.py)
    else:
        clean_obs_nc          = False
        correct_era5          = False
        correct_era5_linear   = False
        create_filled_forcing = False
        create_analysis       = False
        write_to_text         = False

    ##############################################################

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']
    
    raw_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.nc')

    ##############################################################
    ######## CLEAN SITE OBSERVATIONS ########
    ##############################################################
    if clean_obs_nc:

        print('cleaning observational data\n')
        clean_ds = qco.main(raw_ds, sitedata, siteattrs, sitepath, plotdetail=qcplotdetail)
       
        print('setting clean attributes\n')
        clean_ds.attrs['time_analysis_start'] = raw_ds.attrs['time_coverage_start']
        clean_ds.attrs['timestep_number_analysis'] = raw_ds.attrs['timestep_number_analysis']
        clean_ds = set_global_attributes(clean_ds,siteattrs,ds_type='clean_obs')
        clean_ds = set_variable_attributes(clean_ds)
        # clean_ds = clean_ds.merge(assign_sitedata(siteattrs))

        print(f'writing cleaned observations to NetCDF\n')
        fpath = f'{sitepath}/timeseries/{sitename}_clean_observations_{siteattrs["out_suffix"]}.nc'
        write_netcdf_file(ds=clean_ds,fpath_out=fpath)
    else:
        print('loading clean observations from NetCDF\n')
        clean_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_clean_observations_{siteattrs["out_suffix"]}.nc')

    calc_midday_albedo(clean_ds,siteattrs['local_utc_offset_hours'])

    # observed time period
    syear = pd.to_datetime(clean_ds.time.values[0]).year - 10
    eyear = pd.to_datetime(clean_ds.time.values[-1]).year

    ##############################################################
    ######## EXTRACT WFDE5 ########
    ##############################################################
    if extract_wfde5:
        print('extracting WFDE5 data for site location')
        watch_ds = get_wfde5_data(sitename,sitedata,syear,eyear,wfdepath)

        # make directories if necessary
        if not os.path.exists(f'{datapath}/{sitename}'):
            os.makedirs(f'{datapath}/{sitename}')

        # write
        fpath = f'{datapath}/{sitename}/{sitename}_wfde5_v1-1.nc'
        write_netcdf_file(ds=watch_ds,fpath_out=fpath)
    else: 
        watch_ds = xr.open_dataset(f'{datapath}/{sitename}/{sitename}_wfde5_v1-1.nc')

    ##############################################################
    ######## EXTRACT ERA5 ########
    ##############################################################
    if extract_era5: # from GADI

        # era_ds = get_era5_main_rt52(sitename,sitedata,syear,eyear,era5path,sitepath)
        print('Using GADI/RT52')
        # native
        print('import ERA5 site native datafile and select time of spinup and observation')
        era_native = get_era5_data(sitename, sitedata, syear, eyear, era5path, sitepath)

        fpath = f'{sitepath}/{sitename}_era_native.nc'
        write_netcdf_file(ds=era_native,fpath_out=fpath)

        # ALMA
        print('convert ERA5 site raw datafile to ALMA standard variables')
        era_ds = convert_era5_to_alma(era_native,siteattrs,sitename)

        # make directories if necessary
        if not os.path.exists(f'{datapath}/{sitename}'):
            os.makedirs(f'{datapath}/{sitename}')
        # write
        fpath = f'{datapath}/{sitename}/{sitename}_era5_raw.nc'
        write_netcdf_file(ds=era_ds,fpath_out=fpath)
        add_era_vegtype(era_ds,fpath)

    else: # if era5 data already pulled use:
        print('loading ERA5 data for site location')
        era_ds = xr.open_dataset(f'{datapath}/{sitename}/{sitename}_era5_raw.nc')

    ##############################################################
    ######## CORRECT ERA5 ########
    ##############################################################
    if correct_era5:
        print('making corrections to ERA5 data based on local site information')
        corr_ds = calc_era5_corrections(era_ds,watch_ds,sitename,sitedata,sitepath,plot_bias='all',obs_ds=clean_ds)
        corr_ds = set_global_attributes(corr_ds,siteattrs,ds_type='era5_corrected')
        corr_ds = set_variable_attributes(corr_ds)
        # write
        fpath = f'{sitepath}/timeseries/{sitename}_era5_corrected_{siteattrs["out_suffix"]}.nc'
        print(f'writing corrected era5 data to {fpath.split("/")[-1]}')
        write_netcdf_file(ds=corr_ds,fpath_out=fpath)
    else:
        print('loading corrected ERA5 data for site location')
        corr_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_era5_corrected_{siteattrs["out_suffix"]}.nc')

    ##############################################################
    ######## CORRECT ERA5 USING LINEAR REGRESSION ########
    ##############################################################
    if correct_era5_linear:
        lin_ds = calc_era5_linear_corrections(era_ds,watch_ds,clean_ds,siteattrs,sitedata)
        lin_ds = set_global_attributes(lin_ds,siteattrs,ds_type='era5_linear')
        lin_ds = set_variable_attributes(lin_ds)
        # write
        fpath = f'{sitepath}/timeseries/{sitename}_era5_linear_{siteattrs["out_suffix"]}.nc'
        print(f'writing linearly debiased era5 data to {fpath.split("/")[-1]}')
        write_netcdf_file(ds=lin_ds,fpath_out=fpath)
    else:
        print('loading corrected ERA5 data for site location')
        lin_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_era5_linear_{siteattrs["out_suffix"]}.nc')

    ##############################################################
    ######## FILL FORCING VARIABLES ########
    ##############################################################
    if create_filled_forcing:
        print('gap filling forcing')

        # get dataframes for filling
        forcing_fluxes = ['SWdown', 'LWdown', 'Tair', 'Qair', 'PSurf', 'Rainf', 'Snowf', 'Wind_N','Wind_E']
        filling = clean_ds.squeeze().to_dataframe()[forcing_fluxes]

        qc_keys = [f'{x}_qc' for x in forcing_fluxes]
        filling_qc = clean_ds.to_dataframe()[qc_keys]

        corr_df = corr_ds[forcing_fluxes].sel(time=slice(str(syear),str(eyear+1))).to_dataframe()[forcing_fluxes]

        instant_flux =  ['Tair', 'Qair', 'PSurf',  'Wind_N','Wind_E']
        average_flux = ['SWdown', 'LWdown','Rainf', 'Snowf']
        if clean_ds.timestep_interval_seconds == 1800.:
            print('resampling era5 data to 30Min')
            era_to_fill = corr_df.resample('30Min').asfreq()
            era_to_fill[instant_flux] = era_to_fill[instant_flux].interpolate()
            era_to_fill[average_flux] = era_to_fill[average_flux].backfill()
            max_gap = 4
        elif clean_ds.timestep_interval_seconds == 3600.:
            era_to_fill = corr_df.copy()
            max_gap = 2
        else:
            print('timesteps must be 30min or 60min, check netcdf')

        # 0=observed, 1=gapfilled_from_obs, 2=gapfilled_from_era5, 3=missing
        for key in forcing_fluxes:

            print(f'updating {key} qc flags')
            filling[f'{key}_qc'] = clean_ds[f'{key}_qc'].to_series()

            ####################################################################
            #### Step 1) Contemporaneous and nearby flux tower or weather observing sites (where available and provided by observing groups)
            #### already done during site data import

            ####################################################################
            #### Step 2) Small gaps (≤ 2 hours) are linearly interpolated from the adjoining observations

            # fill small gaps linearly, qc = 1 (filled from obs)
            ser_filled,qc = linearly_fill_gaps(filling[key],max_gap=max_gap,qc_flag=1)
            filling[key] = ser_filled

            # update qc flag with non-nan values
            filling[f'{key}_qc'].update(qc)

            ####################################################################
            #### Step 3) Larger gaps (and a 10-year spin-up period) are filled using corrected ERA5 data

            # fill remaining gaps with era, qc = 2 (filled from era)
            ser_filled,qc = era_fill_gaps(filling[key],era_to_fill[key],qc_flag=2)
            filling[key] = ser_filled

            # update qc flag with non-nan values
            filling[f'{key}_qc'].update(qc)

            assert filling[key].isna().sum() == 0, f'ERROR: gaps still remaining in {key}'

        print('done filling gaps\n')

        filled_ds = filling.to_xarray()

        # use ERA5 for Snowf (all sites except JP-Yoyogi which includes snow)
        # if no snow must correct observed rainf which is likely to include melted snow
        if clean_ds.Snowf.to_series().count()==0:
            # set observed rain and snow partitioning from era5 snowfall data
            print('partitioning observed precipitation into snowfall and rainfall from era5')
            obs_partition = filled_ds.to_dataframe()[['Rainf','Snowf']]
            era_partition = era_to_fill.loc[clean_ds.time_coverage_start:clean_ds.time_coverage_end,['Rainf','Snowf']]

            new_partition = pd.DataFrame(index=obs_partition.index, columns=['Rainf','Snowf'])
            era_snowf, obs_precip = era_partition.Snowf, obs_partition.Rainf

            # maintain mass balance by removing era5 snow from observed rain
            new_partition[['Rainf','Snowf']],swe = partition_precip_to_snowf_rainf(era_snowf,obs_precip)

            filled_ds['Rainf'].values = new_partition['Rainf'].values
            filled_ds['Snowf'].values = new_partition['Snowf'].values
        else:
            print('not partitioning precipitation (using snow from site observations)')

        print('setting attributes\n')
        filled_ds.attrs['time_analysis_start'] = clean_ds.attrs['time_coverage_start']
        filled_ds.attrs['timestep_number_analysis'] = clean_ds.attrs['timestep_number_analysis']
        filled_ds = set_global_attributes(filled_ds,siteattrs,ds_type='forcing')
        filled_ds = set_variable_attributes(filled_ds)

        # print(f'writing filled observations to NetCDF\n')
        # fpath = f'{sitepath}/timeseries/{sitename}_filled_observations_{siteattrs["out_suffix"]}.nc'
        # write_netcdf_file(ds=filled_ds,fpath_out=fpath)

        ##############################################################
        ######## PREPEND FORCING VARIABLES WITH ERA5 ########
        ##############################################################

        print('combining era5 and obs data')

        # if clean_ds.timestep_interval_seconds == 1800.:
        #     print('resampling era5 data to 30Min')
        #     rsmp_ds = corr_ds.sel(time=slice(str(syear),str(eyear))).resample(time='30Min').interpolate()
        # else:
        #     rsmp_ds = corr_ds.sel(time=slice(str(syear),str(eyear))).copy()

        era_to_fill_ds = era_to_fill.to_xarray()

        # add qc flag = 2 for all notnull values in era, else = 3
        for key in forcing_fluxes:
            era_to_fill_ds['%s_qc' %key] = xr.DataArray(
            data   = np.where(era_to_fill_ds[key].notnull(),2,3),
            dims   =['time'],
            coords = {'time': era_to_fill_ds.time.values})

        # combine forcing and era-corrected dataset, using forcing where there is overlap
        final_period = clean_ds.time_coverage_end
        forcing_ds = filled_ds.combine_first(era_to_fill_ds).sel(time=slice(None, final_period)) 
        forcing_ds = build_new_dataset_forcing(forcing_ds)

        print('setting forcing attributes\n')
        forcing_ds.attrs['time_analysis_start'] = pd.to_datetime(raw_ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
        forcing_ds.attrs['timestep_number_analysis'] = len(raw_ds.time)
        forcing_ds = set_global_attributes(forcing_ds,siteattrs,ds_type='forcing')
        forcing_ds = set_variable_attributes(forcing_ds)
        forcing_ds = forcing_ds.merge(assign_sitedata(siteattrs))

        # write
        fpath = f'{sitepath}/timeseries/{sitename}_metforcing_{siteattrs["out_suffix"]}.nc'
        write_netcdf_file(ds=forcing_ds,fpath_out=fpath)

    else:
        # print('loading filled observations from NetCDF\n')
        # filled_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_filled_observations_{siteattrs["out_suffix"]}.nc')

        print('loading forcing from NetCDF\n')
        forcing_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_metforcing_{siteattrs["out_suffix"]}.nc')

    if write_to_text:

        print('writing raw obs to text file')
        fpath = f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.txt'
        write_netcdf_to_text_file(ds=raw_ds,fpath_out=fpath)

        print('writing clean obs to text file')
        fpath = f'{sitepath}/timeseries/{sitename}_clean_observations_{siteattrs["out_suffix"]}.txt'
        write_netcdf_to_text_file(ds=clean_ds,fpath_out=fpath)

        print('writing era5 corrected to text file')
        fpath = f'{sitepath}/timeseries/{sitename}_era5_corrected_{siteattrs["out_suffix"]}.txt'
        write_netcdf_to_text_file(ds=corr_ds,fpath_out=fpath)

        print('writing met forcing to text file')
        fpath = f'{sitepath}/timeseries/{sitename}_metforcing_{siteattrs["out_suffix"]}.txt'
        write_netcdf_to_text_file(ds=forcing_ds,fpath_out=fpath)

    # ##############################################################
    # ######## CREATE ANALYSIS FILES (for modelevaluation.org) ########
    # ##############################################################

    # if create_analysis:

    #     data = clean_ds.squeeze().to_dataframe()
    #     times = forcing_ds.time.to_series().index
    #     data = data.reindex(times)

    #     analysis_ds = build_new_dataset_analysis(data,sitedata)

    #     print('setting analysis attributes\n')
    #     analysis_ds.attrs['time_analysis_start'] = pd.to_datetime(raw_ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
    #     analysis_ds.attrs['timestep_number_analysis'] = len(raw_ds.time)
    #     analysis_ds = set_global_attributes(analysis_ds,siteattrs,ds_type='analysis')

    #     print('writing analysis NetCDF')
    #     fpath = f'{sitepath}/timeseries/{sitename}_analysis_{siteattrs["out_suffix"]}.nc'
    #     write_netcdf_file(ds=analysis_ds,fpath_out=fpath)

    # else:
    #     print('loading analysis from NetCDF\n')
    #     analysis_ds = xr.open_dataset(f'{sitepath}/timeseries/{sitename}_analysis_{siteattrs["out_suffix"]}.nc')

    return raw_ds, clean_ds, watch_ds, era_ds, corr_ds, lin_ds, forcing_ds


###############################################################################
# Shared functions
###############################################################################

def prep_site(sitename, sitepath, out_suffix, sitedata_suffix, long_sitename, 
        local_utc_offset_hours, obs_contact, obs_reference, obs_comment,
        history, photo_source, get_globaldata, datapath):
    '''
    prepare site folders, attributes and show local characteristics from global datasets
    run at start of create_dataset_{sitename}.py main function
    '''

    for dirname in ['processing','precip_plots','era_correction','obs_plots','images','timeseries']:
        # make directories if necessary
        if not os.path.exists(f'{sitepath}/{dirname}'):
            print(f'making {dirname} dir')
            os.makedirs(f'{sitepath}/{dirname}')

    print('loading site parameters for %s' %sitename)
    fpath = f'{sitepath}/{sitename}_sitedata_{sitedata_suffix}.csv'
    sitedata_full = pd.read_csv(fpath, index_col=1, delimiter=',')
    sitedata = pd.to_numeric(sitedata_full['value'])

    check_area_fractions(sitedata)

    if get_globaldata:
        print('loading global paramaters for site:\n')
        globalpath = f'{datapath}/global_datasets'
        _,_,_ = get_global_soil(globalpath,sitedata)
        _,_   = get_global_qanth(globalpath,sitedata)
        _,_   = get_climate(globalpath,sitedata)

    # site attrs
    print('setting attributes\n')
    siteattrs = pd.Series(name='siteattrs',dtype='object')
    siteattrs['sitename'] = sitename
    siteattrs['sitepath'] = sitepath
    siteattrs['out_suffix'] = out_suffix
    siteattrs['sitedata_suffix'] = sitedata_suffix
    siteattrs['long_sitename'] = long_sitename
    siteattrs['local_utc_offset_hours'] = local_utc_offset_hours
    siteattrs['obs_contact'] = obs_contact
    siteattrs['obs_reference'] = obs_reference
    siteattrs['obs_comment'] = obs_comment
    siteattrs['history'] = history
    siteattrs['photo_source'] = photo_source

    fpath = f'{sitepath}/{sitename}_siteattrs_{out_suffix}.csv'
    siteattrs.to_csv(fpath,header=True,index=True)

    siteattrs = pd.read_csv(fpath,index_col=0,squeeze=True)
    # reformat loaded utc_offset as float
    siteattrs['local_utc_offset_hours'] = float(siteattrs['local_utc_offset_hours'])

    return sitedata,siteattrs

def set_raw_attributes(raw_ds, siteattrs):

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']

    print('setting raw attributes\n')
    raw_ds.attrs['time_analysis_start'] = pd.to_datetime(raw_ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
    raw_ds.attrs['timestep_number_analysis'] = len(raw_ds.time)
    raw_ds = set_global_attributes(raw_ds,siteattrs,ds_type='raw_obs')
    raw_ds = set_variable_attributes(raw_ds)

    print(f'writing raw observations to NetCDF\n')
    fpath = f'{sitepath}/timeseries/{sitename}_raw_observations_{siteattrs["out_suffix"]}.nc'
    write_netcdf_file(ds=raw_ds,fpath_out=fpath)

    return raw_ds

def post_process_site(sitedata,siteattrs,datapath,
        raw_ds,forcing_ds,clean_ds,era_ds,watch_ds,corr_ds,lin_ds,
        forcingplots,create_outofsample_obs):
    '''
    final site data plotting and error calculation
    runs after pipeline main from create_dataset_{sitename}.py
    '''

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']

    # make website
    create_markdown_observations(forcing_ds,siteattrs)

    # compare corrected, era5 and wfde5 (watch) errors
    compare_corrected_errors(clean_ds,era_ds,watch_ds,corr_ds,lin_ds,sitename,sitepath,'all')

    if forcingplots:
        plot_forcing(datapath,siteattrs,forcing_ds,with_era=False)
    
    if create_outofsample_obs:
        in_ds, out_ds = test_out_of_sample(clean_ds,era_ds,watch_ds,sitedata,siteattrs)

    # Snow partitioning plot
    plot_snow_partitioning(raw_ds,forcing_ds,era_ds,sitepath,sitename)

    return

def get_era5_data(sitename,sitedata,syear,eyear,era5path,sitepath):
    '''
    get native era5 netcdf variables from raijin and combine into xarray dataset
    '''

    # era5 variables to collect from gadi
    vz = '%iv' %wind_hgt # default v10
    uz = '%iu' %wind_hgt # default u10

    # ncvars = ['msdwswrf','msnswrf','msdwlwrf','msnlwrf','2t','2d','sp',vz,uz,'mtpr','msr','msshf','mslhf']
    ncvars = ['msdwswrf','msdwlwrf','2t','2d','sp',vz,uz,'mtpr','msr']

    if sitename == 'SG-TelokKurau': # nearest era tile over water, move to land
        sitedata['latitude'] = sitedata['latitude'] + 0.25
        sitedata['longitude'] = sitedata['longitude'] - 0.25

    # longitude correction for era5 using 0<lon<360
    lat = sitedata['latitude']
    lon = sitedata['longitude']
    if lon < 0.:
        lon = lon + 360.
    assert (0 <= lon < 360), 'longitude in era5 needs to be 0<lon<360'
    assert (-90 <= lat < 90), 'latutude in era5 needs to be -90<lat<90'

    ds = xr.Dataset()
    years = [str(year) for year in range(syear,eyear+1)]

    # loop through variables
    for ncvar in ncvars:
        print('collecting %s data' %ncvar)

        files = []
        # get list of files in path using glob wildcard
        for year in years:
            files = sorted(glob.glob('%s/%s/%s/*' %(era5path,ncvar,year)))

            assert len(files)>0, 'no files found in %s/%s/%s/*' %(era5path,ncvar,year)

            for file in files:
                print('opening %s' %file)
                tmp = xr.open_dataset(file).sel(latitude=sitedata['latitude'],
                                                    longitude=sitedata['longitude'],
                                                    method='nearest')

                ds = xr.merge([ds,tmp])

        longname = tmp[list(tmp.keys())[0]].long_name.lower()
        print('done merging %s (%s)' %(longname,ncvar))

    ############################################################################

    ds.attrs['source'] = era5path

    # get static information (veg frac, type, soil, geopotential, land/sea mask, roughness)
    for ncvar in ['cvl','cvh','tvl','tvh','slt','z','lsm','fsr']:
        static = xr.open_dataset(f'{era5path}/{ncvar}/2000/{ncvar}_era5_oper_sfc_20000101-20000131.nc').sel(
                latitude=sitedata['latitude'],longitude=sitedata['longitude'],time='2000-01-01 00:00', method='nearest')

        ds[ncvar] = static[ncvar]

    return ds

############################################################################

def convert_era5_to_alma(era5,siteattrs,sitename):
    '''
    opens ERA5 native datafile from site and converts to alma standard variables     
    '''

    # convert dewtemp to specific humidity data
    Qdata_np = convert_dewtemp_to_qair(
        dewtemp  = era5['d2m'].values,
        temp     = era5['t2m'].values,
        pressure = era5['sp'].values)
    Qair_xr = xr.DataArray(Qdata_np, coords=[era5.time.values], dims='time')

    # convert era5 to alma form
    ds = xr.Dataset()
    ds = ds.assign( time      = era5['time'])
    ds = ds.assign( SWdown    = era5['msdwswrf'],
                    LWdown    = era5['msdwlwrf'],
                    Wind_N    = era5['v%i' %wind_hgt],
                    Wind_E    = era5['u%i' %wind_hgt],
                    Wind      = (era5['v%i' %wind_hgt]**2 + era5['u%i' %wind_hgt]**2)**(1/2),
                    PSurf     = era5['sp'],
                    Tair      = era5['t2m'],
                    Qair      = Qair_xr,
                    Rainf     = era5['mtpr'] - era5['msr'],
                    Snowf     = era5['msr'],
                    era_wind_hgt = wind_hgt)

    for ncvar in ['cvl','cvh','tvl','tvh','slt','z','lsm','fsr']:
        ds[ncvar] = era5[ncvar]

    # # setting unphysical negative values to zero
    ds.Rainf.values = ds.Rainf.where(ds.Rainf>1E-9,0.).values
    ds.Snowf.values = ds.Snowf.where(ds.Snowf>1E-9,0.).values
    ds.SWdown.values = ds.SWdown.where(ds.SWdown>1E-9,0.).values

    ds = set_variable_attributes(ds)

    ds = set_global_attributes(ds,siteattrs,ds_type='era5_raw')

    return ds

###############################################################################

def correct_wind(ref_wind,local_z0,local_d0,local_wind_hgt,ref_wind_hgt,ref_z0,ref_d0,mode):
    '''correct wind speed assuming log wind profile assuming all neutral conditions

    Parameters
    ----------
    ref_wind   [m/s]   reanalysis grid scalar wind speed at ref_wind_hgt
    local_z0    [m]     local zero-plane displacment 
    local_d0    [m]     local roughness length (assumed constant)
    local_wind_hgt  [m]     local site measurement height for wind
    ref_wind_hgt   [m]     reanlysis site measurement height for wind
    ref_z0     [m]     reanalysis grid roughness length (assumed constant)
    ref_d0     [m]     reanalysis grid zero-plane displacment 
    mode        [0,1]   two different methods to calculate:
        mode=0: including displacement height from Goret et al. 2019 Eq 9 (https://doi.org/10.1016/j.aeaoa.2019.100042)
        mode=1: excluding displacement height from https://websites.pmc.ucsc.edu/~jnoble/wind/extrap/
    Method results depend on assumptions about grid z_0 and d_0, 
    method=1 results in higher wind speeds overall
    '''

    # ref_wind_hgt = 10. # basis of measurement height for era5

    if mode == 0: # log law with displacement height
        local_wind = ref_wind*( ( np.log((local_wind_hgt-local_d0)/local_z0) )/( np.log((ref_wind_hgt-ref_d0)/ref_z0) ) )

    if mode == 1: # log law described at: https://websites.pmc.ucsc.edu/~jnoble/wind/extrap/
        local_wind = ref_wind*(np.log(local_wind_hgt/local_z0))/(np.log(ref_wind_hgt/ref_z0))

    return local_wind

def correct_pressure(ref_height,ref_temp,ref_pressure,local_temp,local_height,mode=1):
    '''correct pressure to local site based on height difference

    Parameters
    ----------
    ref_height     [m]     reference (converted from) height above sea level (asl)
    ref_temp       [K]     reference (converted from) 2m air temperature
    ref_pressure   [Pa]    reference (converted from) surface air pressure
    local_temp      [K]    local site 2m air temperature
    local_height    [m]    local site height for correction
    mode            [0,1]  two different methods to calculate:
        - 0: Hypsometric equation (assuming constant temperature)
        - 1: Barometric equation (includes lapse rate)
        - 2: Hydrostatic equation P = rho * grav * h_diff
        - 3: WATCH method from Weeedon et al. (2010)
    Negligible difference between methods
    '''

    rd = 287.04             # Gas constant for dry air [J K^-1 kg^-1]
    grav = 9.80616          # gravity [m s^-2]
    env_lapse = 6.5/1000.   # environmental lapse rate [K m^-1]

    if mode == 0: # hypsometric equation assumes constant temperature
        local_pressure = ref_pressure*np.exp( (grav*(ref_height-local_height))/(rd*ref_temp) )

    elif mode == 1: # barometric equation with varying temperature
        local_pressure = ref_pressure*(ref_temp/local_temp)**(grav/(-env_lapse*rd))

    elif mode == 2: # hydrostatic equation
        air_density = 1.2
        local_pressure = ref_pressure + air_density * grav * (ref_height - local_height)

    elif mode ==3: # WATCH method
        # weedon et al (2010) eq. 2
        ref_temp_sea_level = ref_temp + ref_height * env_lapse

        # weedon et al (2010) eq. 7
        ref_pressure_sea_level = ref_pressure / ( ref_temp/ ref_temp_sea_level )**(grav/(-env_lapse*rd)) 

        # weedon et al (2010) eq. 9
        local_temp_sea_level = local_temp + ref_height * env_lapse

        # weedon et al (2010) eq. 10
        local_pressure = ref_pressure_sea_level * ( local_temp/local_temp_sea_level )**(grav/(-env_lapse*rd))

    else:
        raise SystemExit(0)

    local_pressure = np.where(local_pressure < 90000, 100000, local_pressure)

    return local_pressure

def calc_esat(temp,pressure,mode=0):
    '''Calculates vapor pressure at saturation

    From Weedon 2010, through Buck 1981: 
    New Equations for Computing Vapor Pressure and Enhancement Factor, Journal of Applied Meteorology
    ----------
    temp        [K]     2m air temperature
    pressure    [Pa]    air pressure
    mode        [0,1]   two different methods to calculate:
        mode=0: from Wheedon et al. 2010
        mode=1: from Ukkola et al., 2017
    NOTE: mode 0 and 1 nearly identical
          Ukkola et al uses the ws=qs approximation (which is not used here, see Weedon 2010)
    '''
   
    # constants
    Rd = 287.05  # specific gas constant for dry air
    Rv = 461.52  #specific gas constant for water vapour
    Epsilon = Rd/Rv  # = 0.622...
    Beta = (1.-Epsilon) # = 0.378 ...

    temp_C = temp - 273.15  # temperature conversion to [C]

    if mode == 0: # complex calculation from Weedon et al. 2010

        # values when over:         water,  ice
        A = np.where( temp_C > 0., 6.1121,  6.1115 )
        B = np.where( temp_C > 0., 18.729,  23.036 )
        C = np.where( temp_C > 0., 257.87,  279.82 )
        D = np.where( temp_C > 0., 227.3,   333.7 )
        X = np.where( temp_C > 0., 0.00072, 0.00022 )
        Y = np.where( temp_C > 0., 3.2E-6,  3.83E-6 )
        Z = np.where( temp_C > 0., 5.9E-10, 6.4E-10 )

        esat = A * np.exp( ((B - (temp_C/D) ) * temp_C)/(temp_C + C))

        enhancement = 1. + X + pressure/100. * (Y + (Z*temp_C**2))

        esat = esat*enhancement*100.

    elif mode == 1: 
        '''simpler calculation from Ukkola et al., 2017
        From Jones (1992), Plants and microclimate: A quantitative approach 
        to environmental plant physiology, p110 '''

        esat = 613.75*np.exp( (17.502*temp_C)/(240.97+temp_C) )

    else:
        raise SystemExit(0)

    return esat

def calc_qsat(esat,pressure):
    '''Calculates specific humidity at saturation

    Parameters
    ----------
    esat        [Pa]    vapor pressure at saturation
    pressure    [Pa]    air pressure

    Returns
    -------
    qsat        [g/g] specific humidity at saturation

    '''
    # constants
    Rd = 287.05  # specific gas constant for dry air
    Rv = 461.52  #specific gas constant for water vapour
    Epsilon = Rd/Rv  # = 0.622...
    Beta = (1.-Epsilon) # = 0.378 ...

    qsat = (Epsilon*esat)/(pressure - Beta*esat)

    return qsat

def calc_density(temp,pressure):
    '''https://www.omnicalculator.com/physics/air-density
    '''

    Rd = 287.05  # specific gas constant for dry air
    Rv = 461.52  #specific gas constant for water vapour

    esat = calc_esat(temp,pressure)

    density = (pressure / (Rd * temp) ) + ( esat / (Rv * temp) )

    return density

###############################################################################
################################# conversions #################################
###############################################################################

def convert_dewtemp_to_qair(dewtemp,temp,pressure,mode=0):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation vapor pressure at dewpoint
    esat_dpt = calc_esat(dewtemp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate specific humidity
    qair = qsat * (esat_dpt/esat)

    return qair

def convert_dewtemp_to_rh(dewtemp,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation vapor pressure at dewpoint
    esat_dpt = calc_esat(dewtemp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate relative humidity
    rh = esat_dpt/esat*100

    assert rh.where(rh<=105).all(), 'relative humidity values > 105. check input'
    assert rh.where(rh>0).any(), 'relative humidity values < 0. check input'
    assert rh.max()>1., 'relative humidity values betwen 0-1 (should be 0-100)'

    return rh


def convert_rh_to_qair(rh,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    assert rh.where(rh<=105).all(), 'relative humidity values > 105. check input'
    assert rh.where(rh>0).any(), 'relative humidity values < 0. check input'
    assert rh.max()>1., 'relative humidity values betwen 0-1 (should be 0-100)'

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate specific humidity
    qair = qsat*rh/100.

    return qair

def convert_qair_to_rh(qair,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate relative humidity
    rh = 100.*qair/qsat

    assert rh.where(rh<=105).all(), 'relative humidity values > 105. check input'
    assert rh.where(rh>0).any(), 'relative humidity values < 0. check input'
    assert rh.max()>1., 'relative humidity values betwen 0-1 (should be 0-100)'

    return rh

def convert_uv_to_wdir(u,v):
    ''' Converts 2 component wind velocity to wind direction scalar
    NOTE: wdir is wind direction FROM

    Parameters
    ----------
    u (np array):    eastward wind component     [m/s]
    v (np array):    northward wind component    [m/s]

    Returns
    ------
    wdir (np array): wind direction from North  [deg]
    '''

    wdir = (180./np.pi)*np.arctan2(u, v) - 180

    # convert negative angles to positive
    wdir = np.where(wdir<0, wdir+360., wdir)

    return wdir

def convert_uv_to_wind(u,v):
    ''' Converts 2 component wind velocity to wind speed scalar
    Parameters
    ----------
    u (np array):    eastward wind component     [m/s]
    v (np array):    northward wind component    [m/s]

    Returns
    ------
    wind (np array): wind speed scalar          [m/s]
    '''

    wind = np.sqrt(u**2 + v**2)

    return wind

def convert_wdir_to_uv(speed,wind_dir_from):
    ''' Converts 2 component wind velocity to wind direction scalar
    NOTE: wdir is wind direction FROM by default
    see: https://unidata.github.io/MetPy/latest/_modules/metpy/calc/basic.html#wind_components

    Parameters
    ----------
    u (np array):    eastward wind component     [m/s]
    v (np array):    northward wind component    [m/s]

    Returns
    ------
    wdir (np array): wind direction from North  [deg]
    '''

    wdir_rad = wind_dir_from*np.pi/180.

    u = -speed * np.sin(wdir_rad)
    v = -speed * np.cos(wdir_rad)

    return u,v

def convert_ustar_to_qtau(ustar,temp,pressure,air_density):

    qtau = ustar**2/air_density

    return qtau

def convert_qtau_to_ustar(qtau,temp,pressure,air_density):

    ustar = np.sqrt(qtau*air_density)

    return ustar

def calc_site_roughness(bldhgt_ave,sigma_h,lambda_p,lambda_pv,lambda_f,bldhgt_max=None,mode=0):
    '''estimate urban site roughness length for momentum and zero-plane displacement based on various methods 
    as described in Grimmond and Oke (1999): Aerodynamic Properties of Urban Areas Derived from Analysis of Surface Form .

    modes: four different morphometric methods to calculate roughness and displacement:
        - 0: Macdonald et al. 1998: An improved method for the estimation of surface roughness of obstacle arrays
        - 1: Kent et al 2017: Aerodynamic roughness parameters in cities: Inclusion of vegetation
        - 2: Millward-Hopkins et al., (2011) per Kent et al. 2017 eq 11 and 12 # TYPO IN EQ 12 OF KENT PAPER
        - 3: Kanda et al., 2013

    Inputs
    ------
    bldhgt_ave [m] : building mean height
    sigma_h [m]    : building height standard deviation
    lambda_p [0-1] : building plan area fraction
    lambda_pv [0-1]: tree plan area fraction
    lambda_f [1]   : wall frontal area fraction
    bldhgt_max [m] : maximum building height (if none assume 1.25 times SD)


    Returns
    -------
    zd [m] zero-plane displacement
    z0 [m] rougness length for momentum

    Warning
    -------
    in mode=0 wall frontal area (assuming random canyon orientation) calculated from Porsen et al 2010 (DOI:10.1002/qj.668)
    in mode=1 (including vegetation), vegetation frontal area index assumed to be equal to tree plan fraction (from Table 1 in Kent et al 2017)
    in mode=3 (Kanda) maximum building height assumed to be 2 standard deviations
    '''

    vonkarm = 0.4
    dragbld = 1.2

    if mode==0:
        print('zd,z0 from Macdonald et al. 1998')

        alpha   = 4.43 # for staggered arrays
        beta    = 1.0  # for staggered arrays

        zd = (1. + alpha**(-lambda_p)*(lambda_p - 1.))*bldhgt_ave    # eq 23 from Mac1998
        z0 = ((1. - zd/bldhgt_ave) * np.exp( -1.*(0.5*beta*(dragbld/vonkarm**2)*(1.-zd/bldhgt_ave)*lambda_f)**(-0.5) ))*bldhgt_ave  # eq 26 from Mac1998

    if mode == 1: # with veg from Kent et al (2018) https://doi.org/10.1007/s11252-017-0710-1
        print('zd,z0 from Macdonald with vegetation from Kent et al. 2017 and 2018')
        # Aerodynamic roughness variation with vegetation: analysis in a suburban neighbourhood and a city park

        alpha   = 4.43 # for staggered arrays
        beta    = 1.0  # for staggered arrays

        lambda_fv = lambda_pv # estimated equality based on Table 1 of Kent et al 2017 for different sites
        P_3D = 0.4   # leaf-on porosity

        lambda_tot = (lambda_p + lambda_pv*(1. - P_3D)) # eq 4 from Kent et al (2018)
        Pv  = (-1.251*P_3D**2 + 0.489*P_3D + 0.803)/dragbld  # eq 7 from Kent et al (2018)
        weighted_frontal_area = lambda_f + lambda_fv*Pv

        zd = (1. + alpha**(-lambda_tot)*(lambda_tot-1.))*bldhgt_ave # eq 5
        z0 = (1. - zd/bldhgt_ave)*np.exp( -1.*(0.5*beta*(dragbld/vonkarm**2)*(1.-zd/bldhgt_ave)*weighted_frontal_area)**(-0.5) )*bldhgt_ave # eq 6

    if mode == 2: # Millward-Hopkins et al. (2011), per Kent et al. 2017 eq 11 and 12
        print('zd,z0 from Millward-Hopkins et al. 2011, per Kent et al. 2017 eq 11 and 12')

        MhoUzd_on_Hav_A = (19.2*lambda_p - 1. + np.exp(-19.2*lambda_p))/(19.2*lambda_p*(1.- np.exp(-19.2*lambda_p)))
        MhoUzd_on_Hav_B = (117*lambda_p + (187.2*lambda_p**3 - 6.1)*(1.- np.exp(-19.2*lambda_p)) )/( (1.+114*lambda_p + 187*lambda_p**3)*(1.-np.exp(-19.2*lambda_p)) )
        MhoUzd_on_Hav = np.where( lambda_p >= 0.19, MhoUzd_on_Hav_A,  MhoUzd_on_Hav_B )

        Mho_exp = np.exp( -1.*(0.5*dragbld*vonkarm**(-2)*lambda_f)**(-0.5) )
        MhoUz0_on_Hav = ( (1. - MhoUzd_on_Hav)*Mho_exp)

        zd = bldhgt_ave*(MhoUzd_on_Hav + ((0.2375 * np.log(lambda_p) + 1.1738)*(sigma_h/bldhgt_ave)) )

        # z0 = bldhgt_ave*(MhoUz0_on_Hav + (np.exp(0.8867*lambda_f) - 1.)*(sigma_h/bldhgt_ave)**(np.exp(2.3271*lambda_f))) # TYPO IN KENT ET AL PAPER
        z0 = bldhgt_ave*(MhoUz0_on_Hav + np.exp(0.8867*lambda_f - 1.)*(sigma_h/bldhgt_ave)**(np.exp(2.3271*lambda_f))) # based on UMEP

    if mode == 3:
        print('zd,z0 Kanda et al 2013')

        alpha   = 4.43 # for staggered arrays
        beta    = 1.0  # for staggered arrays

        if bldhgt_max==None:
            bldhgt_max = 1.5*sigma_h + bldhgt_ave # assume bldhgt_max
            # bldhgt_max = 12.51*sigma_h**0.77 # eq 3 from Kanda et al 2013

        # first calculate macdonald et al 1998 (for later scaling)
        mac_zd = (1. + (lambda_p - 1.)*alpha**(-lambda_p))*bldhgt_ave
        mac_z0 = ((1. - mac_zd/bldhgt_ave) * np.exp( -1.*(0.5*beta*(dragbld/vonkarm**2)*(1.-mac_zd/bldhgt_ave)*lambda_f)**(-0.5) ))*bldhgt_ave

        # then Kanda et al 2013 scaling parameters
        a0,b0,c0 = 1.29, 0.36, -0.17
        a1,b1,c1 = 0.71, 20.21, -0.77

        X = (sigma_h + bldhgt_ave)/bldhgt_max  # eq. 10b
        Y = (lambda_p*sigma_h)/bldhgt_ave        # eq. 12b

        zd = bldhgt_max*(c0*X**2 + (a0*lambda_p**b0 - c0)*X) # eq 10a 
        z0 = mac_z0*(a1 + b1*Y**2 + c1*Y) # eq 12a

    return np.round(zd,4),np.round(z0,4)

def calc_rougness_from_ustar(ustar,wind,sitedata):
    try:
        # estimate roughness assuming displacment fraction of average roughness elements
        vonkarmon = 0.4
        max_roughness = max(sitedata['building_mean_height'],sitedata['tree_mean_height'])
        # max_roughness = sitedata['building_mean_height']
        zd = 0.67*max_roughness # rule of thumb
        # kent et al 2017 eq 23
        z0 = ((sitedata['measurement_height_above_ground'] - zd)*np.exp(-(wind*vonkarmon)/ustar))
    except Exception as e:
        print('roughness could not be calculated from available observations')
        z0=np.nan

    return z0

def calc_roughness_from_clean(clean_ds,sitedata):

    try:
        # estimate roughness assuming displacment fraction of average roughness elements
        vonkarmon = 0.4
        max_roughness = max(sitedata['building_mean_height'],sitedata['tree_mean_height'])
        zd = 0.67*max_roughness # rule of thumb
        wind = np.sqrt(clean_ds.Wind_N.to_series()**2 + clean_ds.Wind_E.to_series()**2)
        ustar = convert_qtau_to_ustar(
                    qtau=clean_ds.Qtau,
                    temp=clean_ds.Tair,
                    pressure=clean_ds.PSurf,
                    air_density=calc_density(
                        temp=clean_ds.Tair,
                        pressure=clean_ds.PSurf))

        # kent et al 2017 eq 23
        z0 = ((sitedata['measurement_height_above_ground'] - zd)*np.exp(-(wind*vonkarmon)/ustar)).values
    except Exception as e:
        print('roughness could not be calculated from available observations')
        z0=np.nan

    return z0

def rolling_hourly_bias_correction(flux,sitepath,era,obs,plot_bias='all',window=60,publication=False):
    '''
    This function appies a bias correction to era5 data based on available observations.
    The mean for each hour in a rolling [60] day window is used for bias correcting era data.
    Where observations are not available, bias is linearly interpolated between available data.

    PROCESS
    -------
    0. retain era periods only where obs available
    1. calculate metric for each day of year for era and obs seperately
    2. loop era and obs metric 3 times to avoid end effects
    3. calculate rolling metric for era and obs seperately
    4. calculate comparison metric
    5. interpolate missing values
    6. select central year
    7. apply correction to era
    '''
    # #### TESTING
    # flux = 'LWdown'
    # obs = clean_ds[flux].to_series()
    # era = era_ds[flux].to_series()

    sitename = sitepath.split('/')[-1]

    min_periods = 30
    obs_days = (obs.dropna().index[-1] - obs.dropna().index[0]).days
    if obs_days < min_periods:
        print(f'WARNING: updating min_periods in bias correction to {obs_days}')
        min_periods = obs_days

    # extend obs dataframe to 366 days if necessary (to apply bias correction to full year)
    if (obs.index[-1]-obs.index[0]).days < 366:
        edate = obs.index[0] + pd.Timedelta(days=366)
        freq = obs.index[1]-obs.index[0]
        obs = pd.Series(data = obs, index=pd.date_range(start=obs.index[0],end=edate,freq=freq))

    # 0. match observed and reanalysis periods
    obs_clean = obs.resample('H',closed='right',label='right').mean() 
    era_clean = era[obs_clean.index[0]:obs_clean.index[-1]].where(obs_clean.notna())

    # 1. find mean for each hour of year
    obs_doy_hour = obs_clean.groupby([obs_clean.index.dayofyear,obs_clean.index.hour]).mean().rename_axis(index=['doy','hour'])
    era_doy_hour = era_clean.groupby([era_clean.index.dayofyear,era_clean.index.hour]).mean().rename_axis(index=['doy','hour'])

    last_day = 366
    if len(obs_doy_hour.groupby(level=0)) < last_day:
        doys = [last_day]*24
        hours = [x for x in range(24)]
        leap = pd.Series(index=pd.MultiIndex.from_arrays([doys,hours]),dtype=float)
        obs_doy_hour = pd.concat([obs_doy_hour,leap])
        era_doy_hour = pd.concat([era_doy_hour,leap])

    # 2. repeat 3x in order to loop central year before calculating rolling mean
    obs_doy_hour3 = pd.concat([obs_doy_hour]*3)
    era_doy_hour3 = pd.concat([era_doy_hour]*3)
    doys = np.array([[x]*24 for x in range(1,last_day*3 + 1)]).flatten()
    hours = np.tile(np.arange(1,25),last_day*3)
    obs_doy_hour3.index = pd.MultiIndex.from_arrays([doys,hours])
    era_doy_hour3.index = pd.MultiIndex.from_arrays([doys,hours])
    obs_doy_hour3 = obs_doy_hour3.rename_axis(index=['doy','hour'])
    era_doy_hour3 = era_doy_hour3.rename_axis(index=['doy','hour'])

    # 3. calculate rolling mean at each hour across 3 x year loops with minimum periods in window, with repeat for greater smoothing
    era3_rolling = era_doy_hour3.groupby(level='hour').apply(lambda x: x.rolling(window,min_periods=min_periods,center=True).mean())
    era3_rolling = era3_rolling.groupby(level='hour').apply(lambda x: x.rolling(window,min_periods=min_periods,center=True).mean())

    obs3_rolling = obs_doy_hour3.groupby(level='hour').apply(lambda x: x.rolling(window,min_periods=min_periods,center=True).mean())
    obs3_rolling = obs3_rolling.groupby(level='hour').apply(lambda x: x.rolling(window,min_periods=min_periods,center=True).mean())

    # 4. calculate bias for each hour of year
    bias3_rolling = obs3_rolling - era3_rolling

    # 5. fill gaps in yearly bias correction by linear interpolation through each hour seperately
    new_group_list = []
    for hour, group in bias3_rolling.groupby(level='hour'):
        new_group = group.interpolate()
        new_group_list.append(new_group)
    new_bias3_rolling = pd.concat(new_group_list).sort_index()

    # 6. slice central year of rolling mean
    new_bias = pd.Series(new_bias3_rolling.loc[last_day+1:last_day*2].values, index=era_doy_hour.index)
    # fillna with mean if necessary (only for very short obs periods)
    new_bias = new_bias.unstack().fillna(new_bias.unstack().mean(axis=1)).stack()

    # calculate bias corrected era data
    new_group_list = []
    for (doy,hour), group in era.groupby([era.index.dayofyear,era.index.hour]):
        new_group = group + new_bias.loc[doy,hour]
        new_group_list.append(new_group)
    corr = pd.concat(new_group_list).sort_index()

    # if plot_bias:
    if flux == 'Tair':
        units = 'K'
    elif flux == 'PSurf':
        units = 'Pa'
    elif flux == 'Qair':
        units = 'kg/kg'
    elif flux == 'LWdown':
        units = 'W/m2'
    else:
        units = None
        print('WARNING: hourly correction units not recognised')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,5))
    new_bias.unstack(level=1).plot(ax=ax,cmap='jet',alpha=0.8)
    
    ax.set_xlabel('Time (day of year)')
    ax.set_ylabel(f'Correction ({units})')
    ax.set_xlim([0,366])
    ax.grid(lw=0.5,color='0.8',zorder=-1)
    ax.axhline(y=0,color='k',linewidth=1)
    ax.legend(loc='upper left',fontsize=7,ncol=1,bbox_to_anchor=(1.01,0.99),title='hour')

    if publication:
        ax.set_title('a)',loc='left')
    else:
        ax.set_title(f'{sitename} {flux}: hourly bias correction from ERA5')

    if (publication) and (flux=='Tair'):
        fig.savefig(f'{sitepath}/era_correction/{sitename}_{flux}_hourlycorrection_{plot_bias}_fig4a.{img_fmt}',bbox_inches='tight',dpi=150)
    else:
        fig.savefig(f'{sitepath}/era_correction/{sitename}_{flux}_hourlycorrection_{plot_bias}.{img_fmt}',bbox_inches='tight',dpi=150)
    plt.close('all')

    print(f'saving {flux} bias correction to ERA5')
    new_bias.name = flux
    new_bias.to_csv(f'{sitepath}/era_correction/{sitename}_erabias_{flux}.csv')

    return corr

def rolling_daily_range_correction(era,obs,window=30):
    '''
    This function appies a bias correction to era5 data based on available observations.
    The mean daily range in a rolling [30] day window is used to bias correct era data.
    Where observations are not available, bias is linearly interpolated between adjacent data.

    PROCESS
    -------
    0. retain era periods only where obs available
    1. calculate metric for each day of year for era and obs seperately
    2. loop era and obs metric 3 times to avoid end effects
    3. calculate rolling metric for era and obs seperately
    4. calculate comparison metric
    5. interpolate missing values
    6. select central year
    7. apply correction to era
    '''
 
    min_periods = 15

    era = corr_ds['SWdown'].to_series()
    obs = clean_ds['SWdown'].to_series()

    # 0. match observed and reanalysis periods
    obs_clean = obs.resample('1H',closed='right',label='right').mean()
    era_clean = era.where(obs.notna())

    #### 30 day rolling mean ####

    # 1. find max for each day of year
    obs_doy = obs_clean.groupby([lambda x: x.dayofyear]).mean().rename_axis(index=['doy'])
    era_doy = era_clean.groupby([lambda x: x.dayofyear]).mean().rename_axis(index=['doy'])

    # 2. repeat 3x in order to loop central year before calculating rolling mean
    obs3_doy = pd.concat([obs_doy]*3)
    obs3_doy.index = np.arange(1, len(obs3_doy) + 1)
    obs3_doy = obs3_doy.rename_axis(index=['doy'])

    era3_doy = pd.concat([era_doy]*3)
    era3_doy.index = np.arange(1, len(era3_doy) + 1)
    era3_doy = era3_doy.rename_axis(index=['doy'])

    # 3. calculate rolling mean for each day acroos 3 x year with minimum periods in window
    obs3_rolling = obs3_doy.rolling(window,min_periods=min_periods,center=True).mean()
    era3_rolling = era3_doy.rolling(window,min_periods=min_periods,center=True).mean()

    # 4. calculate bias for each day of year
    bias3 = obs3_rolling - era3_rolling

    # 5. fill gaps 3 x year bias correction by linear interpolation
    bias3_filled = bias3.interpolate()

    # 6. slice central year of rolling mean
    last_day = era_doy.index[-1]
    new_bias = pd.Series(bias3_filled.loc[last_day+1:last_day*2].values, index=era_doy.index)

    #### 30 day rolling diurnal range ####

    # 1. find max for each day of year
    obs_doy_max = obs_clean.groupby([lambda x: x.dayofyear]).max().rename_axis(index=['doy'])
    era_doy_max = era_clean.groupby([lambda x: x.dayofyear]).max().rename_axis(index=['doy'])

    # 2. repeat 3x in order to loop central year before calculating rolling mean
    obs3_max = pd.concat([obs_doy_max]*3)
    obs3_max.index = np.arange(1, len(obs3_max) + 1)
    obs3_max = obs3_max.rename_axis(index=['doy'])

    era3_max = pd.concat([era_doy_max]*3)
    era3_max.index = np.arange(1, len(era3_max) + 1)
    era3_max = era3_max.rename_axis(index=['doy'])

    # 3. calculate rolling mean for each day acroos 3 x year with minimum periods in window
    obs3_rolling = obs3_max.rolling(window,min_periods=min_periods,center=True).mean()
    era3_rolling = era3_max.rolling(window,min_periods=min_periods,center=True).mean()

    # 4. ratio of obs to era
    ratio3 = obs3_rolling/era3_rolling

    # 5. fill gaps 3 x year bias correction by linear interpolation
    ratio3_filled = ratio3.interpolate()

    # 6. slice central year of rolling mean
    last_day = era_doy_max.index[-1]
    new_ratio = pd.Series(ratio3_filled.loc[last_day+1:last_day*2].values, index=era_doy_max.index)

    # 7. calculate bias corrected era data
    new_group_list = []
    for doy, group in era.groupby([lambda x: x.dayofyear]):
        new_group = group * new_ratio.loc[doy] + new_bias.loc[doy]
        new_group_list.append(new_group)
    corr = pd.concat(new_group_list).sort_index()

    corr_clean = corr.where(obs.notna())

    return corr

def partition_precip_to_snowf_rainf(era_snowf,obs_precip):
    ''' partitition observed total precipitation into rain and snow using era5 snowfall'''

    # iteratively record snow water equivalent (swe) before melting
    swe = [0.]
    rain_adj = [0.]

    for i in range(1,len(obs_precip)):

        # take last swe value, add new snowfall, remove observed 'precipitation' (assumed snowmelt)
        new_swe = swe[i-1] + era_snowf.iloc[i] - obs_precip.iloc[i]

        # record rainfall adjustement where 
        if ( (swe[-1] > 1E-7) and (new_swe < 0) ):
            rain_adj.append(-swe[i-1])
        else:
            rain_adj.append(0.)

        swe.append( max(0, new_swe) )

    swe = pd.Series(data=swe,index=obs_precip.index)
    rain_adj = pd.Series(data=rain_adj,index=obs_precip.index)

    # create new partitioned snow and rain dataframe
    partitioned = pd.DataFrame(index=obs_precip.index, columns=['Rainf','Snowf'])
    partitioned['Snowf'] = era_snowf
    partitioned['Rainf'] = (obs_precip + rain_adj - era_snowf).where(swe<=0, 0)

    return partitioned,swe

def find_ghcnd_closest_stations(syear,eyear,sitedata,datapath,nshow=6):

    fpath = f'{datapath}/global_datasets/ghcnd-inventory_prcp.txt'
    df = pd.read_csv(fpath,header=None,names=['station','latitude','longitude','precip','syear','eyear'])

    dist = 2
    # select stations within dist
    df = df[(df['latitude'] >= sitedata['latitude'] - dist) & (df['latitude'] <= sitedata['latitude'] + dist)]
    df = df[(df['longitude'] >= sitedata['longitude'] - dist) & (df['longitude'] <= sitedata['longitude'] + dist)]
    # select stations within forcing period
    df = df[(df['syear'] <= eyear) & (df['eyear'] >= syear)]

    # pythag to nearest stations
    df['dist'] = np.sqrt( (sitedata['longitude']-df['longitude'])**2 + (sitedata['latitude']-df['latitude'])**2 )
    df = df.sort_values('dist')

    return df.head(nshow)

def get_ghcnd_precip(sitepath,datapath,syear,eyear,rain_sites):
    ''' Using GHCN-D daily summary for nearest site available during forcing period.
    see: https://www.ncdc.noaa.gov/cdo-web/search
    select: Daily Summaries
    date range: 1st Jan -10 yrs from first observation
    search for: stations with siteid

    click on symbol to see coverage in that period

    In checkout, choose METRIC not STANDARD units
    '''

    sitename = sitepath.split('/')[-1]

    missing={}
    syear,eyear = str(syear),str(eyear+1)

    ser = pd.Series(data=np.nan,index=pd.date_range(syear,eyear,freq='1D'),name='ghcnd_filled')

    fpath = f'{datapath}/{sitename}/GHCND_{rain_sites[0]}.csv'
    raw = pd.read_csv(fpath, parse_dates=['DATE'], index_col='DATE')
    ser = ser.fillna(raw.loc[syear:eyear,'PRCP'])
    missing[raw['NAME'].iloc[0]] = f'precip in primary site {rain_sites[0]} ({raw.NAME[0]}) missing: {len(ser) - ser.count()} ({100 - 100*ser.count()/len(ser):.1f} %)'
    print(missing[raw['NAME'].iloc[0]])

    if len(ser)==ser.count():
        plot_ghcnd_cumulative(sitepath,ser,1,missing)
        return ser

    for i,site in enumerate(rain_sites[1:]):

        fpath = f'{datapath}/{sitename}/GHCND_{site}.csv'
        raw = pd.read_csv(fpath, parse_dates=['DATE'], index_col='DATE')
        ser = ser.fillna(raw.loc[syear:eyear,'PRCP'])
        missing[raw['NAME'].iloc[0]] = f'precip with filler site {site} ({raw.NAME[0]}) missing: {len(ser) - ser.count()} ({100 - 100*ser.count()/len(ser):.1f} %)'
        print(missing[raw['NAME'].iloc[0]])

        if len(ser)==ser.count():
            plot_ghcnd_cumulative(sitepath,ser,i+2,missing)
            return ser 

    # special fill for Escandon (outside of obs period)
    if sitename == 'MX-Escandon':
        ser.loc['2013-01-01'] = 0

    plot_ghcnd_cumulative(sitepath,ser,'all',missing)
    assert ser.count() == len(ser), f'GHCND rain observations still has {len(ser) - ser.count()} missing days, download more sites'

    # ser.name = 'ghcnd'

    return ser

def plot_ghcnd_cumulative(sitepath,ser,s,missing):

    print('plotting ghcnd cumulative')

    sitename = sitepath.split('/')[-1]

    plt.close('all')
    fig,ax = plt.subplots()

    # GHCND plot
    ser.cumsum().plot(ax=ax,color='k',label='GHCND')

    try:
        # raw site obs plot
        fname = glob.glob(f'{sitepath}/timeseries/{sitename}_raw_observations*.nc')[-1]
        raw_ds = xr.open_dataset(fname)
        ts = raw_ds.timestep_interval_seconds
        cumsum_start = ser[:raw_ds.time_coverage_start].sum()
        rainf = raw_ds.Rainf.squeeze().to_series()*ts
        (rainf.cumsum()+cumsum_start).plot(ax=ax,color='red',label='raw site obs')
        rainf_day = rainf.resample('1D').sum()
        precip_yr = rainf_day.sum()/(rainf_day.index.date[-1]-rainf_day.index.date[0]).days * 365.25
        ax.text(0.22,0.94, f'{precip_yr:.2f} mm/year from raw site observation',color='r',transform=ax.transAxes, fontsize=7, va='top')
    except Exception:
        print('site observed rainfall not found for ghcnd plot')
        pass

    # yearly annotation
    precip_yr = ser.sum()/(ser.index.date[-1]-ser.index.date[0]).days * 365.25
    ax.text(0.22,0.98, f'{precip_yr:.2f} mm/year from {s} ghcnd sites',transform=ax.transAxes, fontsize=7, va='top')

    ax.set_title(f'{sitename}: cumulative rainfall from nearby GHCND sites')
    ax.set_xlabel('')
    ax.set_ylabel(f'cumulative rain (mm)')
    ax.set_ylim((0,None))

    for i,key in enumerate(missing.keys()):
        ax.text(0.99,0.01 + 0.03*i, missing[key],transform=ax.transAxes, fontsize=7, va='bottom',ha='right')

    # ax.legend(loc='upper center',fontsize=7,bbox_to_anchor=(0.5,-0.05),ncol=2)
    ax.legend(loc='upper left',fontsize=7)

    fig.savefig(f'{sitepath}/precip_plots/{sitename}_ghcnd_cumulative_precip.{img_fmt}',bbox_inches='tight',dpi=150)

    return

def write_ghcnd_precip(sitepath,sitename,ser):

    assert len(ser) == ser.count(), 'precip still missing, add more sites'
    assert ser.count() == len(ser), 'nan in rain obs'
    assert any(ser.index.duplicated()) == False, 'rain obs has duplicate days'

    ser.to_csv(f'{sitepath}/timeseries/{sitename}_ghcnd_precip.csv',header=True)
    precip_yr = ser.sum()/(ser.index.date[-1]-ser.index.date[0]).days * 365.25
    print('nearby met stations precip: %.2f mm/year' %(precip_yr))

    return

def plot_snow_partitioning(obs_ds,forcing_ds,era_ds,sitepath,sitename):

    # sdate,edate = '2013-01-15', '2013-02-16'
    sdate,edate = obs_ds.time_coverage_start,obs_ds.time_coverage_end
    ts = obs_ds.timestep_interval_seconds

    obs = obs_ds.sel(time=slice(sdate,edate))[['Rainf','Snowf','Tair']].squeeze().to_dataframe()
    # obs = obs.resample('1H',closed='right',label='right').mean()
    obs['precip'] = obs['Rainf']
    # add snow to precip if recorded
    idx = obs['Snowf'].dropna().index
    obs.loc[idx,'precip'] = obs.loc[idx,'Rainf'] + obs.loc[idx,'Snowf']

    fill = forcing_ds.sel(time=slice(sdate,edate))[['Rainf','Snowf','Tair']].squeeze().to_dataframe()
    # fill = fill.resample('1H',closed='right',label='right').mean()
    fill['precip'] = fill['Rainf'] + fill['Snowf']

    era = era_ds.sel(time=slice(sdate,edate)).squeeze().to_dataframe()[['Rainf','Snowf','Tair']]
    era = era.resample('30Min').asfreq()
    era[['Rainf','Snowf']] = era[['Rainf','Snowf']].backfill()
    era[['Tair']] = era[['Tair']].interpolate()
    era['precip'] = era['Rainf'] + era['Snowf']

    ###############

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_title(f'Precipitation in {sitename}: corrected')
    ax.set_ylabel('Water fluxes [mm]')

    (obs['precip']*ts).cumsum().plot(ax=ax, color='k', lw=2, ls='solid',label='raw obs all precip')
    (fill['precip']*ts).cumsum().plot(ax=ax, color='r', lw=1, ls='solid',label='forcing all precip')
    (fill['Rainf']*ts).cumsum().plot(ax=ax, color='r', lw=1, ls='dashed',label='forcing Rainf')
    (fill['Snowf']*ts).cumsum().plot(ax=ax, color='r', lw=1, ls='dotted',label='forcing Snowf')
    # (era['precip']*ts).cumsum().plot(ax=ax, color='royalblue', lw=1, ls='solid',label='ERA5 all precip')

    try:
        fname = f'{sitepath}/timeseries/{sitename}_ghcnd_precip.csv'
        ghcnd = pd.read_csv(fname,index_col=0,parse_dates=True)[sdate:edate]
        ghcnd.rename(columns={'ghcnd':'GHCND all precip'},inplace=True)
        ghcnd.cumsum().plot(ax=ax, color='purple', lw=1)
    except:
        print('GHCND data not found')
        pass

    ax2 = ax.twinx()
    ax2.set_ylabel('Air temperature [°C]')
    (fill['Tair'] - 273.15).plot(ax=ax2, color='0.75', ls='solid',lw=1,label='obs temperature')

    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='center right', fontsize=8)

    ax.set_xlabel('')
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False) # hide the 'canvas'
    ax.set_ylim(0,None)

    # plt.show()
    fig.savefig(f'{sitepath}/precip_plots/{sitename}_snow_correction.{img_fmt}', dpi=150,bbox_inches='tight')

    # plt.show()

    plt.close('all')

    return

def calc_MAE(sim,obs):
    '''Calculate Mean Absolute Error from Best et al 2015'''

    metric = abs(sim-obs).mean()

    return metric

def calc_MBE(sim,obs):
    '''Calculate Mean Bias Error from Best et al 2015'''

    metric = np.mean(sim-obs)

    return metric

def calc_NSD(sim,obs):
    '''calculate normalised standard deviation'''

    metric = sim.std()/obs.std()

    return metric

def calc_NSD(sim,obs):
    '''calculate normalised standard deviation'''

    metric = sim.std()/obs.std()

    return metric

def calc_R(sim,obs):
    '''cacluate normalised correlation coefficient (pearsons)'''

    metric = sim.corr(obs, method='pearson')

    return metric

###############################################################################

def calc_era5_linear_corrections(era_ds,watch_ds,obs_ds,siteattrs,sitedata):

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']

    lin_ds = era_ds.copy()

    min_obs = 10

    print('\ncorrecting Wind linearly')

    if len(obs_ds['Wind_N'].to_series().dropna().unique()) > min_obs:

        obs_wind = np.sqrt(obs_ds['Wind_N'].to_series()**2 + obs_ds['Wind_E'].to_series()**2)
        era_wind = era_ds['Wind'].to_series()

        lin_ds['Wind'].values = linear_debiasing('Wind',sitepath,era_wind,obs_wind)

        print('')
        print(f'mean observed wind speed: {obs_wind.mean():.2f} m/s')
        print(f'mean wind speed change from {era_wind.mean():.2f} to {lin_ds.Wind.to_series().mean():.2f} m/s')

        era_wdir = convert_uv_to_wdir(era_ds['Wind_E'],era_ds['Wind_N'])
        lin_ds['Wind_E'].values = convert_wdir_to_uv(lin_ds['Wind'].values,era_wdir)[0]
        lin_ds['Wind_N'].values = convert_wdir_to_uv(lin_ds['Wind'].values,era_wdir)[1]

    ################################################################################

    for key in ['Tair','PSurf','Qair','SWdown']:
        print(f'\ncorrecting {key} linearly')
        if len(obs_ds[key].to_series().dropna().unique()) > min_obs:
            era = era_ds[key].to_series()
            obs = obs_ds[key].to_series()
            lin_ds[key].values = linear_debiasing(key,sitepath,era,obs)

    ################################################################################

    # fill NaN values in corrected dataset with zero
    lin_ds['SWdown'].values = lin_ds['SWdown'].fillna(0.).values
    # set negative values to zero
    lin_ds['SWdown'].values = lin_ds['SWdown'].where(lin_ds['SWdown']>=0., 0.).values

    # # setting very small values to zero
    lin_ds.Rainf.values = lin_ds.Rainf.where(lin_ds.Rainf>1E-8,0.).values
    lin_ds.Snowf.values = lin_ds.Snowf.where(lin_ds.Snowf>1E-8,0.).values

    if sitename in ['JP-Yoyogi']:
        lin_ds['Qair'].values = lin_ds['Qair'].where(lin_ds['Qair']>=0.0001, 0.0001).values

    ################################################################################

    key = 'LWdown'
    print(f'\ncorrecting {key} linearlly')
    if sitename == 'MX-Escandon': # MX-Escandon has no LW in 2011 obs period, using 2006 observed data for bias correction
        print('Using LWdown from 2006 for bias correction at MX-Escandon')
        # obs_LWdown = xr.open_dataset(f'{sitepath}/MX-Escandon_era5_corr_v2006.nc')['LWdown']
        obs_LWdown = xr.open_dataset(f'{sitepath}/timeseries/MX-Escandon_raw2006_observations.nc')['LWdown']
    else:
        obs_LWdown = obs_ds['LWdown']

    try:
        print('remove spurious LWdown era5 value at 2010-11-27 09:00 (at many sites)')
        before = era_ds['LWdown'].loc[dict(time='2010-11-27 08:00')]
        after = era_ds['LWdown'].loc[dict(time='2010-11-27 10:00')]
        era_ds['LWdown'].loc[dict(time='2010-11-27 09:00')] = 0.5*(before+after)
    except Exception:
        print('No correction done to ERA5 for LW on 2010-11-27 09:00 (not found)')

    if len(obs_LWdown.to_series().dropna().unique()) > min_obs:
        era = era_ds[key].to_series()
        obs = obs_LWdown.to_series()
        lin_ds[key].values = linear_debiasing(key,sitepath,era,obs)

    print(f'mean Tair change from {era_ds["Tair"].mean().values-273.15:.1f} to {lin_ds["Tair"].mean().values-273.15:.1f} °C')
    print(f'mean PSurf change from {era_ds["PSurf"].mean().values:.1f} to {lin_ds["PSurf"].mean().values:.1f} Pa')
    print(f'mean Qair change from {era_ds["Qair"].mean().values:.4f} to {lin_ds["Qair"].mean().values:.4f} kg/kg')
    print(f'mean SWdown change from {era_ds["SWdown"].mean().values:.1f} to {lin_ds["SWdown"].mean().values:.1f} W/m2')
    print(f'mean LWdown change from {era_ds["LWdown"].mean().values:.1f} to {lin_ds["LWdown"].mean().values:.1f} W/m2')

    ################################################################################

    # setting 
    print('\nchecking corrections are within ALMA ranges...')
    alma_ranges = pd.DataFrame({
        'SWdown' : (0,1360),
        'LWdown' : (0,750),
        'Tair'   : (213,333),
        'Qair'   : (0,0.03),
        'PSurf'  : (5000,110000),
        'Rainf'  : (0,0.02),
        'Snowf'  : (0,0.0085),
        'Wind_N' : (-75,75),
        'Wind_E' : (-75,75),
        },index=('min','max'))

    for key in alma_ranges.columns:
        assert (lin_ds[key].values >= alma_ranges.loc['min',key]).all() and (lin_ds[key].values <= alma_ranges.loc['max',key]).all(), f'corrected {key} outside ALMA physical range: {float(lin_ds[key].min())}'

    ########## ANNOTATIONS ###########

    lin_ds['era_lat'] = era_ds.latitude
    lin_ds['era_lon'] = era_ds.longitude
    lin_ds['era_wind_hgt'] = era_ds.era_wind_hgt

    print('done correcting ERA5 data with obs')

    return lin_ds

def linear_debiasing(flux,sitepath,era,obs):
    '''
    This function does bias correction using linear regression technique from Vichard and Papale 2015 (FLUXNET)
    DOI: 10.5194/essd-7-157-2015
    '''

    # # #### TESTING
    # flux = 'LWdown'
    # obs = clean_ds[flux].to_series()
    # era = era_ds[flux].to_series()

    # 0. match observed and reanalysis periods
    obs_clean = obs.resample('H',closed='right',label='right').mean() 
    era_clean = era[obs_clean.index[0]:obs_clean.index[-1]].where(obs_clean.notna())

    x = era_clean.dropna().values
    y = obs_clean.dropna().values

    # 1. allow intercept unless global radiation or wind (per Vichard & Papale 2015)
    if flux not in ['SWdown','Wind_N','Wind_E','Wind']:
        x = sm.add_constant(x)

        # 2. calculate regression parameters
        model = sm.OLS(y,x)
        intercept = model.fit().params[0]
        slope = model.fit().params[1]

        # 3. bias correct era5 per eq 4 in V&P2015
        corr = slope*era + intercept

        print(f'{flux} slope factor: {slope:.3f}, intercept: {intercept:.3f}')

    else:
        # 2. calculate regression parameters
        model = sm.OLS(y,x)
        slope = model.fit().params[0]

        # 3. bias correct era5 per eq 4 in V&P2015
        corr = slope*era

        print(f'{flux} slope factor: {slope:.3f}')

    return corr

def calc_era5_corrections(era_ds,watch_ds,sitename,sitedata,sitepath,
    plot_bias,obs_ds=None,ref_wind_hgt=wind_hgt):

    '''takes era5 data (in alma format) and makes corrections based on site data '''

    corr_ds = era_ds.copy()

    print('\ncorrecting Rainf and Snowf')
    try:
        rain_obs = pd.read_csv(f'{sitepath}/timeseries/{sitename}_ghcnd_precip.csv',index_col=0,parse_dates=True)
        # use controlled 10 yr period prior to observation
        sdate = pd.Timestamp(obs_ds.time_coverage_end) - pd.DateOffset(years=10)
        edate = pd.Timestamp(obs_ds.time_coverage_end)
        # sdate, edate = rain_obs.index[0], rain_obs.index[-1]
        span_years = (edate-sdate)/np.timedelta64(1, 'Y')

        obs_total_precip_mm = rain_obs.loc[sdate:edate].values.sum()
        print('')
        print(f'available ghcnd years: {span_years:.2f}')
        print('obs total precip %.2f mm (%s - %s) %.2f mm per year' %(obs_total_precip_mm, sdate.year,edate.year,obs_total_precip_mm/span_years))

        # set ERA5 snowfall to zero during analysis period for Capitole (according to advice from Masson)
        if sitename in ['FR-Capitole']:
            era_ds['Snowf'].loc[dict(time=slice(obs_ds.time_coverage_start,obs_ds.time_coverage_end))] = 0.
            corr_ds['Snowf'].loc[dict(time=slice(obs_ds.time_coverage_start,obs_ds.time_coverage_end))] = 0.

        # tmp = era_ds.Rainf.sel(time=sdate),era_ds.Rainf.sel(time=edate)
        era_total_rain_mm = era_ds.Rainf.sel(time=slice(sdate,edate)).values.sum()*3600
        era_total_snow_mm = era_ds.Snowf.sel(time=slice(sdate,edate)).values.sum()*3600
        era_total_precip_mm = era_total_snow_mm + era_total_rain_mm
        print('')
        print('era total rain %.2f mm (%s - %s) %.2f mm per year' %(era_total_rain_mm, sdate.year,edate.year,era_total_rain_mm/span_years))
        print('era total snow %.2f mm (%s - %s) %.2f mm per year' %(era_total_snow_mm, sdate.year,edate.year,era_total_snow_mm/span_years))
        print('era total precip %.2f mm (%s - %s) %.2f mm per year' %(era_total_precip_mm, sdate.year,edate.year,era_total_precip_mm/span_years))

        try:
            # tmp = watch_ds.Rainf.sel(time=sdate),watch_ds.Rainf.sel(time=edate)
            watch_total_rain_mm = watch_ds.Rainf.sel(time=slice(sdate,edate)).values.sum()*3600
            watch_total_snow_mm = watch_ds.Snowf.sel(time=slice(sdate,edate)).values.sum()*3600
            watch_total_precip_mm = watch_total_snow_mm + watch_total_rain_mm
            print('')
            print('watch total rain %.2f mm (%s - %s) %.2f mm per year' %(watch_total_rain_mm, sdate.year,edate.year,watch_total_rain_mm/span_years))
            print('watch total snow %.2f mm (%s - %s) %.2f mm per year' %(watch_total_snow_mm, sdate.year,edate.year,watch_total_snow_mm/span_years))
            print('watch total precip %.2f mm (%s - %s) %.2f mm per year' %(watch_total_precip_mm, sdate.year,edate.year,watch_total_precip_mm/span_years))
        except Exception:
            print('no watch data found')

        precip_corr_ratio = obs_total_precip_mm/era_total_precip_mm
        print('era_corr = era x %.2f' %(precip_corr_ratio))

        corr_ds['Rainf'].values = era_ds['Rainf'].values * precip_corr_ratio
        corr_ds['Snowf'].values = era_ds['Snowf'].values * precip_corr_ratio
    except Exception as e:
        print('rain correction error:',e)
        print('no GHCND precipitation file found, not undertaking bias correction')

    # # setting very small values to zero
    corr_ds.Rainf.values = corr_ds.Rainf.where(corr_ds.Rainf>1E-8,0.).values
    corr_ds.Snowf.values = corr_ds.Snowf.where(corr_ds.Snowf>1E-8,0.).values

    ################################################################################

    min_obs = 10

    print('\ncorrecting Wind using log laws')

    if len(obs_ds['Wind_N'].to_series().dropna().unique()) > min_obs:

        print('finding effective era5 z0')

        obs_wind = np.sqrt(obs_ds['Wind_N'].to_series()**2 + obs_ds['Wind_E'].to_series()**2)
        era_wind = era_ds["Wind"].to_series().where(obs_wind.notna())
        cor_wind = corr_ds["Wind"].to_series().where(obs_wind.notna())

        eff_z0 = era_ds.fsr.values
        bias = 0.5
        print(f'mean observed wind speed: {obs_wind.mean():.2f} m/s')

        # loop until mean corrected wind is close to obs wind
        while abs(bias)>0.01:
            print(f'trying z0: {eff_z0}')

            cor_wind_eff = pd.Series(correct_wind(
                    ref_wind     = cor_wind,
                    local_z0     = sitedata['roughness_length_momentum'],
                    local_d0     = sitedata['displacement_height'],
                    local_wind_hgt = sitedata['measurement_height_above_ground'],
                    ref_wind_hgt = era_ds.era_wind_hgt.values,
                    ref_z0       = eff_z0,
                    ref_d0       = 0,
                    mode         = 0),index=cor_wind.index)

            bias = cor_wind_eff.mean() - obs_wind.mean()

            print(f'z0=eff: mean wind speed change from {era_wind.mean():.2f} to {cor_wind_eff.mean():.2f} m/s')
            print(f'BIAS: {bias:.2f},MAE: {calc_MAE(sim=cor_wind_eff,obs=obs_wind):.2f} m/s')

            eff_z0 = round(eff_z0 - bias/5,3)

        print('')
        print(f'done finding effecting era5 z0: {eff_z0}')

        print(f'Converting ERA5 wind at {era_ds.era_wind_hgt.values}m height with grid {era_ds.fsr.values:.2f}m roughness, effective {eff_z0}m roughness')
        print(f' to site {sitedata["measurement_height_above_ground"]}m height with {sitedata["roughness_length_momentum"]}m roughness, {sitedata["displacement_height"]:.2f}m displacement')

        corr_ds['Wind_N'].values = correct_wind(
                ref_wind     = era_ds['Wind_N'].values,
                local_z0     = sitedata['roughness_length_momentum'],
                local_d0     = sitedata['displacement_height'],
                local_wind_hgt = sitedata['measurement_height_above_ground'],
                ref_wind_hgt = era_ds.era_wind_hgt.values,
                ref_z0       = eff_z0,
                ref_d0       = 0,
                mode         = 0)

        corr_ds['Wind_E'].values = correct_wind(
                ref_wind     = era_ds['Wind_E'].values,
                local_z0     = sitedata['roughness_length_momentum'],
                local_d0     = sitedata['displacement_height'],
                local_wind_hgt = sitedata['measurement_height_above_ground'],
                ref_wind_hgt = era_ds.era_wind_hgt.values,
                ref_z0       = eff_z0,
                ref_d0       = 0,
                mode         = 0)

        corr_ds['Wind'].values = np.sqrt(corr_ds['Wind_N'].values**2 + corr_ds['Wind_E'].values**2)
        print('')
        print(f'mean observed wind speed: {obs_wind.mean():.2f} m/s')
        print(f'mean wind speed change from {era_wind.mean():.2f} to {corr_ds.Wind.to_series().mean():.2f} m/s')

        if plot_bias=='all':
            #####
            cor_wind_N_fsr = pd.Series(correct_wind(
                    ref_wind     = era_ds['Wind_N'].values,
                    local_z0     = sitedata['roughness_length_momentum'],
                    local_d0     = sitedata['displacement_height'],
                    local_wind_hgt = sitedata['measurement_height_above_ground'],
                    ref_wind_hgt = era_ds.era_wind_hgt.values,
                    ref_z0       = era_ds.fsr.values,
                    ref_d0       = -era_ds.fsr.values,
                    mode         = 0),index=cor_wind.index)

            cor_wind_E_fsr = pd.Series(correct_wind(
                    ref_wind     = era_ds['Wind_E'].values,
                    local_z0     = sitedata['roughness_length_momentum'],
                    local_d0     = sitedata['displacement_height'],
                    local_wind_hgt = sitedata['measurement_height_above_ground'],
                    ref_wind_hgt = era_ds.era_wind_hgt.values,
                    ref_z0       = era_ds.fsr.values,
                    ref_d0       = -era_ds.fsr.values,
                    mode         = 0),index=cor_wind.index)

            cor_wind_fsr = np.sqrt(cor_wind_N_fsr**2 + cor_wind_E_fsr**2).where(obs_wind.notna())

            wname = f'{sitename}_u{int(wind_hgt)}'
            windstats = pd.Series(name=wname,dtype=float)
            windstats['site_roughness'] = f'{sitedata["roughness_length_momentum"]:.3f}'
            windstats['era_roughness'] = f'{era_ds.fsr.values:.3f}'
            windstats['eff_roughness'] = f'{eff_z0:.3f}'
            windstats['obs_wind_mean'] = f'{obs_wind.mean():.3f}'
            windstats['era_wind_mean'] = f'{era_wind.mean():.3f}'
            windstats['cor_wind_fsr_mean'] = f'{cor_wind_fsr.mean():.3f}'
            windstats['cor_wind_eff_mean'] = f'{cor_wind_eff.mean():.3f}'
            windstats['era_wind_mae'] = '%.3f' %(calc_MAE(sim=era_wind,obs=obs_wind))
            windstats['cor_wind_fsr_mae'] = '%.3f' %(calc_MAE(sim=cor_wind_fsr,obs=obs_wind))
            windstats['cor_wind_eff_mae'] = '%.3f' %(calc_MAE(sim=cor_wind_eff,obs=obs_wind))

            windstats.to_csv(f'{sitepath}/processing/{wname}.csv')

    ################################################################################

    for key in ['Tair','PSurf','Qair']:
        print(f'\ncorrecting {key}')
        if len(obs_ds[key].to_series().dropna().unique()) > min_obs:
            tmp = rolling_hourly_bias_correction(
                flux      = key,
                sitepath  = sitepath,
                era       = era_ds[key].to_series(),
                obs       = obs_ds[key].to_series(),
                plot_bias = plot_bias)

            corr_ds[key].values = tmp.values


    print(f'mean Tair change from {era_ds["Tair"].mean().values-273.15:.1f} to {corr_ds["Tair"].mean().values-273.15:.1f} °C')
    print(f'mean PSurf change from {era_ds["PSurf"].mean().values:.1f} to {corr_ds["PSurf"].mean().values:.1f} Pa')

    if sitename in ['US-WestPhoenix','MX-Escandon','JP-Yoyogi','GR-HECKOR','FI-Torni','KR-Ochang','KR-Jungnang','US-Baltimore']:
        corr_ds['Qair'].values = corr_ds['Qair'].where(corr_ds['Qair']>=0.0001, 0.0001).values

    print(f'mean Qair change from {era_ds["Qair"].mean().values:.4f} to {corr_ds["Qair"].mean().values:.4f} kg/kg')

    ################################################################################

    print('\ncorrecting LWdown')
    if sitename == 'MX-Escandon': # MX-Escandon has no LW in 2011 obs period, using 2006 observed data for bias correction
        print('Using LWdown from 2006 for bias correction at MX-Escandon')
        # obs_LWdown = xr.open_dataset(f'{sitepath}/MX-Escandon_era5_corr_v2006.nc')['LWdown']
        obs_LWdown = xr.open_dataset(f'{sitepath}/timeseries/MX-Escandon_raw2006_observations.nc')['LWdown']
    else:
        obs_LWdown = obs_ds['LWdown']

    # fix spurious LWdown era5 value at 2010-11-27 09:00 (at many sites)
    try:
        before = era_ds['LWdown'].loc[dict(time='2010-11-27 08:00')]
        after = era_ds['LWdown'].loc[dict(time='2010-11-27 10:00')]
        era_ds['LWdown'].loc[dict(time='2010-11-27 09:00')] = 0.5*(before+after)
    except Exception:
        print('No correction done to ERA5 for LW on 2010-11-27 09:00 (spurious low reading at many sites)')

    if len(obs_LWdown.to_series().dropna().unique()) > min_obs:
        tmp = rolling_hourly_bias_correction(
            flux      = 'LWdown',
            sitepath  = sitepath,
            era       = era_ds['LWdown'].to_series(),
            obs       = obs_LWdown.to_series(),
            plot_bias = plot_bias)

        corr_ds['LWdown'].values = tmp.values

    print(f'mean LWdown change from {era_ds["LWdown"].mean().values:.1f} to {corr_ds["LWdown"].mean().values:.1f} W/m2')

    ################################################################################

    print('\ncorrecting SWdown')
    # fill NaN values in corrected dataset with zero
    corr_ds['SWdown'].values = corr_ds['SWdown'].fillna(0.).values
    # set negative values to zero
    corr_ds['SWdown'].values = corr_ds['SWdown'].where(corr_ds['SWdown']>=0., 0.).values

    print(f'mean SWdown change from {era_ds["SWdown"].mean().values:.1f} to {corr_ds["SWdown"].mean().values:.1f} W/m2')

    ################################################################################

    # setting 
    print('\nchecking corrections are within ALMA ranges...')
    alma_ranges = pd.DataFrame({
        'SWdown' : (0,1360),
        'LWdown' : (0,750),
        'Tair'   : (213,333),
        'Qair'   : (0,0.03),
        'PSurf'  : (5000,110000),
        'Rainf'  : (0,0.02),
        'Snowf'  : (0,0.0085),
        'Wind_N' : (-75,75),
        'Wind_E' : (-75,75),
        },index=('min','max'))

    for key in alma_ranges.columns:
        assert (corr_ds[key].values >= alma_ranges.loc['min',key]).all() and (corr_ds[key].values <= alma_ranges.loc['max',key]).all(), f'corrected {key} outside ALMA physical range: {float(corr_ds[key].min())}'

    ########## ANNOTATIONS ###########

    corr_ds['era_lat'] = era_ds.latitude
    corr_ds['era_lon'] = era_ds.longitude
    corr_ds['era_wind_hgt'] = era_ds.era_wind_hgt

    print('done correcting ERA5 data with obs')

    return corr_ds


def linearly_fill_gaps(ser_to_fill : pd.Series, max_gap=3, qc_flag=1) -> pd.Series:
    ''' linearly fill gaps where gap is smaller than max_gap'''

    new_group_list = []

    ser_test = ser_to_fill.copy()

    # break series into groups (unless series is shorter than max_gap)
    if max_gap < len(ser_test):

        # find break points
        isna = pd.Series( np.where(ser_test.isna(), 1, np.nan), index=ser_test.index )
        isna_sum = isna
        for n in range(1,max_gap+1):
            isna_sum = isna_sum + isna.shift(n)
        break_idxs = isna_sum.dropna().index

        # # add start series
        prev_break = ser_test.index[0]
        
        for next_break in break_idxs:
            group = ser_test[prev_break:next_break]

            # skip to next loop if no values in group (for efficiency)
            if group.count() == 0:
                continue

            new_group = group.interpolate(method='linear',limit=max_gap, limit_area='inside')
            new_group_list.append(new_group)

            prev_break = next_break

        # append final group without interpolation
        group = ser_test[prev_break:]

    else: #simply group entire series
        group = ser_test

    new_group = group.interpolate(method='linear',limit=max_gap, limit_area='inside')
    new_group_list.append(new_group)

    # concatenate all groups
    filled = pd.concat(new_group_list).sort_index()
    filled = filled[~filled.index.duplicated(keep='first')]

    # flag where fill occured
    qc = filled.where(ser_test.isna())
    qc = qc.where(qc.isna(),qc_flag)

    # print(pd.concat([ser_to_fill,filled,qc],axis=1))

    assert len(filled) == len(ser_to_fill), 'length of filled series is different to original'
    print('values filled linearly: %s ' %(filled.count() - ser_to_fill.count()))

    return filled, qc

def adjacent_fill_gaps(ser_to_fill : pd.Series, delta='1D', min_data=0.5, scale = False, bias=False, qc_flag=1):

    ser_test = ser_to_fill.copy()

    # split into period (day), and define before and after period to fill from.
    new_group_list = []
    for period, group in ser_test.groupby(pd.Grouper(freq='1D')):

        # if data missing more than minimum, continue (default = 50% data minimum)
        if group.count()/len(group) < min_data:
            final_group = group
        else:
            prev_period = period - pd.Timedelta(delta)
            next_period = period + pd.Timedelta(delta)

            prev_group = ser_test.loc[prev_period:period - pd.Timedelta(minutes=1)]
            next_group = ser_test.loc[period:next_period - pd.Timedelta(minutes=1)]

            # base series for gap filling
            ser_fill_base = ser_test.loc[prev_period:next_period - pd.Timedelta(minutes=1)]
            new_group = ser_fill_base.groupby(ser_fill_base.index.time).mean().reset_index(drop=True)
            new_group = new_group[:len(group)]
            new_group.index = group.index

            if bias:
                # de-bias based on mean of current/adjacent days in observed periods.
                prev_idx = group.dropna().index - pd.Timedelta(delta)
                next_idx = group.dropna().index + pd.Timedelta(delta)

                bias = ( np.nanmean(np.array([prev_group[prev_idx].mean(), next_group[next_idx].mean()])) ) - group.mean()
                new_group = new_group + bias

            if scale:
                # rescale group
                scale_fill = min(2, max( 0.5, (new_group.std() / group.std())))
                new_group = new_group / scale_fill

            final_group = group.fillna(new_group)

        new_group_list.append(final_group)

    filled = pd.concat(new_group_list).sort_index()

    qc = filled.where(ser_to_fill.isna())
    qc = qc.where(qc.isna(),qc_flag)

    assert len(filled) == len(ser_to_fill), 'length of filled series is different to original'
    print('values filled with adjacent days: %s ' %(filled.count() - ser_to_fill.count()))

    return filled, qc

def era_fill_gaps(ser_to_fill, era, qc_flag=2):
    '''
    Fills gaps in forcing variable using another timeseries (corrected ERA5) of same frequency

    ser_to_fill: (pd.Series)    observed timeseries
    era: (pd.Series)            corrected era variable timeseries
    qc_flag: (int)              qc flag to use in affected periods
    '''

    freq1 = ser_to_fill.index[1] - ser_to_fill.index[0]
    freq2 = era.index[1] - era.index[0]

    assert freq1==freq2, 'obs and era frequencies do not match'

    ser_filled = ser_to_fill.fillna(era)

    qc = ser_filled.where(ser_to_fill.isna())
    qc = qc.where(qc.isna(),qc_flag)

    assert len(ser_filled) == len(ser_to_fill), 'length of filled series is different to original'
    print('values filled with era: %s ' %(ser_filled.count() - ser_to_fill.count()))

    return ser_filled, qc


def plot_forcing(datapath,siteattrs,forcing_ds, with_era=False,
    fluxes = ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Snowf','Wind_N','Wind_E']):
    '''Plots gap filled data for forcing variables (as in Fig 3. of manuscript).
    Called from each site create_dataset_*.py script

    siteattrs:  pandas dataframe for site information from {sitename}_siteattrs_v1.csv
    forcing_ds: xarray dataset of forcing variables
    with_era:   option to plot with original era5 data (uncorrected)
    '''

    import matplotlib.patheffects as path_effects
    from matplotlib import dates as mdates

    timestep = forcing_ds.timestep_interval_seconds
    edate = forcing_ds.time_coverage_end

    ocol = 'tab:blue'
    ecol = 'orangered'
    outline = [path_effects.withStroke(linewidth=0.8, foreground='black')]

    sitepath = siteattrs["sitepath"]
    sitename = siteattrs["sitename"]

    era5_raw = xr.open_dataset(f'{datapath}/{sitename}/{sitename}_era5_raw.nc')

    if with_era:
        era5_corr = xr.open_dataset(f'{sitepath}/{sitename}_era5_corr_{siteattrs["out_suffix"]}.nc')
        wfde5 = xr.open_dataset(f'{datapath}/{sitename}/{sitename}_wfde5_v1-1.nc')

    for flux in fluxes:

        print(f'plotting {flux}')

        plt.close('all')
        fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,8))

        for i,ax in enumerate(axes.flatten()):

            if i == 0:
                rsmp_period = '7D'
                sdate       = forcing_ds.time_coverage_start
                f           = forcing_ds[flux].sel(time=slice(sdate,edate)).copy()
                f_rsmp      = f.squeeze().to_series().resample(rsmp_period).mean()
                f_min       = f.min().values - 0.07*(f.max().values - f.min().values)
                f_max       = f.max().values + 0.06*(f.max().values - f.min().values)
                e           = era5_raw[flux].sel(time=slice(sdate,edate)).to_series()

            if i == 1:
                rsmp_period = '7D'
                sdate       = forcing_ds.time_analysis_start
                span        = (pd.Timestamp(edate) - pd.Timestamp(sdate)).days
                f           = forcing_ds[flux].sel(time=slice(sdate,edate)).copy()
                f_rsmp      = f_rsmp[sdate:]
                f_min       = f.min().values - 0.07*(f.max().values - f.min().values)
                f_max       = f.max().values + 0.06*(f.max().values - f.min().values)
                e           = e[sdate:]

            # setup
            ax.text(0.99,0.99,f'timesteps: {f.count().values} at {int(timestep)}s', 
                color='black', va='top', ha='right',fontsize=7, transform=ax.transAxes)
            # expand plot with single invisable value
            ax.plot([f.time[0].values],[f_max],marker='x',color='white',zorder=-1)

            # observed values
            f0 = f.where(forcing_ds[f'{flux}_qc']==0)
            f0.plot(ax=ax, label='observed', color='k', lw=0.0, marker='o', ms=2.5, mew=0.0, alpha=1.0, zorder=1)
            f0.where(f0.isnull(),f_min).plot(ax=ax, color='0.9', ms=3, mew=0.5, marker='|', lw=0, zorder=1)
            ax.text(0.01,0.99,'observed: %s (%.2f%%)' %(f0.count().values, (100*f0.count().values/f.count().values) ),
                color='black', va='top',ha='left',fontsize=7, transform=ax.transAxes)

            # gap-filled from obs values
            f1 = f.where(forcing_ds[f'{flux}_qc']==1)
            f1.plot(ax=ax, label='gap-filled, from obs', color=ocol, lw=0.0, marker='o', ms=2.5, mew=0.0, zorder=2)
            f1.where(f1.isnull(),f_min).plot(ax=ax, color=ocol, ms=3, mew=0.5, marker='|', lw=0, zorder=3)
            ax.text(0.01,0.95,'gap-filled from obs: %s (%.2f%%)' %(f1.count().values, (100*f1.count().values/f.count().values) ),
                color=ocol, va='top',ha='left',fontsize=7, transform=ax.transAxes)

            # gap-filled from era5 values
            f2 = f.where(forcing_ds[f'{flux}_qc']==2)
            f2.plot(ax=ax, label='gap-filled, derived from era5', color=ecol, lw=0.0, marker='o', ms=2.5, mew=0.0, alpha=1.0, zorder=3)
            f2.where(f2.isnull(),f_min).plot(ax=ax, color=ecol, ms=3, mew=0.5, marker='|', lw=0, zorder=2)
            ax.text(0.01,0.91,'gap-filled from era5: %s (%.2f%%)' %(f2.count().values, (100*f2.count().values/f.count().values) ),
                color=ecol, va='top',ha='left',fontsize=7, transform=ax.transAxes)

            # era
            # e.plot(ax=ax, label='era5 raw', color='g', lw=0.0, marker='o', ms=2.5, mew=0.0, alpha=0.20, zorder=0)

            ax.plot(f_rsmp.index,f_rsmp.values,color='white', lw=0.75, zorder=5, label=None) #,path_effects=outline)

            if with_era:
                e5 = era5_raw[flux].sel(time=slice(sdate,edate)).squeeze().to_series()
                ax.plot(e5.index,e5.values, color='green', alpha=0.5, label='era5')

                e5c = era5_corr[flux].sel(time=slice(sdate,edate)).squeeze().to_series()
                ax.plot(e5c.index,e5c.values, color='red', alpha=0.5, label='era5 site corrected')

                if flux not in ['Wind_N','Wind_E']:
                    w5 = wfde5[flux].sel(time=slice(sdate,edate)).squeeze().to_series().shift(1)
                    ax.plot(w5.index,w5.values, color='magenta', alpha=0.5, label='wfde5')

            if i == 0:
                ax.set_title('')
                ax.set_title(f'{sitename} {flux} in full period', fontsize=12, loc='left')
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(axis='x', reset=True, length=2,top=False)

            if i == 1:
                ax.set_title('')
                ax.set_title(f'{sitename} {flux} in analysis period', fontsize=12, loc='left')
                ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1,13,1) if span<(365*2.5) else range(1,13,6)))
                # ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
                ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                ax.tick_params(axis='x', reset=True, length=2, top=False)

            # annotations
            ax.legend(loc='lower right', ncol=4, fontsize=7, bbox_to_anchor=(1.005,1.005))
            ax.grid(axis='y', which='major', color='0.85', lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel('')
            gaps = 100*(f1.count().values + f2.count().values)/f.count().values
            ax.annotate('gap-filled (%.2f%%)' %gaps, xy=(f.time[0].values, f_min), 
                xytext=(0,2), textcoords='offset points',
                va='bottom',ha='left',fontsize=7,color='black')

            f_max = f.squeeze().to_series().groupby(lambda x: x.date).max()
            f_min = f.squeeze().to_series().groupby(lambda x: x.date).min()
            ax.fill_between(x=f_max.index, y1=f_min, y2=f_max,
                color='k', lw=0.0, alpha=0.15, zorder=0)

        # plt.show()

        if with_era:
            plt.show()
        else:
            fig.savefig(f'{sitepath}/obs_plots/{flux}_gapfilled_forcing.{img_fmt}',bbox_inches='tight',dpi=150)

        plt.close('all')

    return

def test_out_of_sample(clean_ds,era_ds,watch_ds,sitedata,siteattrs,outfrac=0.2):
    '''splits input dataset into two parts (in and out) based on outfrac (fraction of period in out-of-sample)'''

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']

    print('splitting available obs for out-of-sample testing')

    splitpoint = (pd.Timestamp(clean_ds.time_coverage_end) - pd.Timestamp(clean_ds.time_coverage_start)).days*(1-outfrac)
    split = pd.Timestamp(clean_ds.time_coverage_start) + pd.Timedelta(days=int(splitpoint))
    in_ds = clean_ds.sel(time=slice(None,split))
    in_ds.attrs['time_analysis_start'] = pd.to_datetime(in_ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
    in_ds.attrs['timestep_number_analysis'] = len(in_ds.time)
    in_ds = set_global_attributes(in_ds,siteattrs,ds_type='raw_obs')
    in_ds = set_variable_attributes(in_ds)

    out_ds = clean_ds.sel(time=slice(split,None))
    out_ds.attrs['time_analysis_start'] = pd.to_datetime(out_ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
    out_ds.attrs['timestep_number_analysis'] = len(out_ds.time)
    out_ds = set_global_attributes(out_ds,siteattrs,ds_type='raw_obs')
    out_ds = set_variable_attributes(out_ds)

    print('creating new bias-corrected era5 using out of sample only')
    part_corr_ds = calc_era5_corrections(era_ds,watch_ds,sitename,sitedata,sitepath,
        plot_bias='out',obs_ds=in_ds )

    print('creating new bias-corrected era5 using out of sample only (linear regression method)')
    part_lin_ds = calc_era5_linear_corrections(era_ds,watch_ds,in_ds,siteattrs,sitedata)

    # # check correction and plot
    # print('in sample corrections')
    # compare_corrected_errors(in_ds,era_ds,watch_ds,part_corr_ds,part_lin_ds,sitename,sitepath,'in-sample')

    print('out-of-sample corrections')
    compare_corrected_errors(out_ds,era_ds,watch_ds,part_corr_ds,part_lin_ds,sitename,sitepath,'out-of-sample')

    return in_ds, out_ds

def compare_corrected_errors(clean_ds,era_ds,watch_ds,corr_ds,lin_ds,sitename,sitepath,sample='in-sample'):

    resample_obs_to_era=True

    fluxes1 = ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Wind_N','Wind_E']
    fluxes2 = ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Wind']
    
    if resample_obs_to_era:
        print('resampling')
        clean = clean_ds[fluxes1].to_dataframe().resample('60Min',closed='right',label='right').mean()
        sdate, edate = clean.index[0], clean.index[-1]
        # compare daytime SWdown only
        clean['SWdown'] = clean['SWdown'].where(clean['SWdown']>0)
        # create wind speed from components for comparison with WATCH
        clean['Wind'] = np.sqrt(clean['Wind_N']**2 + clean['Wind_E']**2)
        # match continuous datasets to observation periods
        corr = corr_ds[fluxes2].sel(time=slice(sdate,edate)).to_dataframe().where(clean.notna())
        era = era_ds[fluxes2].sel(time=slice(sdate,edate)).to_dataframe().where(clean.notna())
        watch = watch_ds[fluxes2].sel(time=slice(sdate,edate)).to_dataframe().where(clean.notna())
        tmp = lin_ds[fluxes2].sel(time=slice(sdate,edate)).to_dataframe()
        lin = tmp.where(clean.notna())
        print('done resampling')

    else: # resample era to match obs (not used)
        clean = clean_ds[fluxes1].to_dataframe()
        ts = (clean.index[1] - clean.index[0]).seconds
        sdate, edate = clean.index[0], clean.index[-1]
        clean['Wind'] = np.sqrt(clean['Wind_N']**2 + clean['Wind_E']**2)
        corr = corr_ds[fluxes2].to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        era = era_ds[fluxes2].to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        watch = watch_ds[fluxes2].to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        lin = lin_ds[fluxes2].to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())

    for metric in ['mae','mbe','nsd','r']:

        # # special case for Escandon using 2006 LW observations
        # if (sitename=='MX-Escandon') and (sample in ['in-sample','out-of-sample']):
        #     df = pd.read_csv(f'{sitepath}/processing/{metric}_{sample}.csv',index_col=0)
        # else:
        #     df = pd.DataFrame()

        df = pd.DataFrame()

        if clean['SWdown'].where(clean['SWdown']>0).dropna().count() > 1:
            flux,units,shift = 'SWdown','W/m2',1
            df['SWdown'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['LWdown'].dropna().count() > 1:
            flux,units,shift = 'LWdown','W/m2',1
            df['LWdown'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['Tair'].dropna().count() > 1:
            flux,units,shift = 'Tair','K',0
            df['Tair'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['Qair'].dropna().count() > 1:
            flux,units,shift = 'Qair','kg/kg',0
            df['Qair'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['PSurf'].dropna().count() > 1:
            flux,units,shift = 'PSurf','Pa',0
            df['PSurf'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['Rainf'].dropna().count() > 1:
            flux,units,shift = 'Rainf','kg/m2/s',1
            df['Rainf'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        if clean['Wind'].dropna().count() > 1:
            flux,units,shift = 'Wind','m/s',0
            df['Wind'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        print(f'{sample} {metric}: ')
        print(df)

        df.to_csv(f'{sitepath}/processing/{metric}_{sample}.csv')

    return

def compare_corrected_errors_escandon(clean_ds,era_ds,watch_ds,corr_ds,lin_ds,sitename,sitepath,sample='in-sample'):

    resample_obs_to_era=True
    
    if resample_obs_to_era:
        clean = clean_ds.to_dataframe().resample('60Min',closed='right',label='right').mean()
        sdate, edate = clean.index[0], clean.index[-1]
        clean['Wind'] = np.sqrt(clean['Wind_N']**2 + clean['Wind_E']**2)
        corr = corr_ds.to_dataframe()[sdate:edate].where(clean.notna())
        era = era_ds.to_dataframe()[sdate:edate].where(clean.notna())
        watch = watch_ds.to_dataframe()[sdate:edate].where(clean.notna())
        lin = lin_ds.to_dataframe()[sdate:edate].where(clean.notna())

    else: # resample era to match obs (not used)
        clean = clean_ds.to_dataframe()
        ts = (clean.index[1] - clean.index[0]).seconds
        sdate, edate = clean.index[0], clean.index[-1]
        clean['Wind'] = np.sqrt(clean['Wind_N']**2 + clean['Wind_E']**2)
        corr = corr_ds.to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        era = era_ds.to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        watch = watch_ds.to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())
        lin = lin_ds.to_dataframe().resample('%sS' %ts).interpolate()[sdate:edate].where(clean.notna())

    for metric in ['mae','mbe','nsd','r']:

        df = pd.read_csv(f'{sitepath}/processing/{metric}_{sample}.csv',index_col=0)

        if clean['LWdown'].dropna().count() > 1:
            flux,units,shift = 'LWdown','W/m2',1
            df['LWdown'] = plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric)

        print(f'{sample} {metric}: ')
        print(df)

        df.to_csv(f'{sitepath}/processing/{metric}_{sample}.csv')

    return

def plot_corr_comparison(clean,corr,era,watch,lin,shift,flux,units,sitename,sitepath,sample,metric,plotwatch=True,publication=False):

    if sitename in ['GR-HECKOR','NL-Amsterdam','JP-Yoyogi','KR-Jungnang']:
        plotwatch=False

    ts = (clean.index[1] - clean.index[0]).seconds

    if metric == 'mae':
        if flux in ['Qair','Rainf']:
            s1 = '%.7f %s' %(calc_MAE(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.7f %s'  %(calc_MAE(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.7f %s' %(calc_MAE(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.7f %s' %(calc_MAE(sim=lin[flux],obs=clean[flux]),units)
        else:
            s1 = '%.2f %s' %(calc_MAE(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.2f %s'  %(calc_MAE(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.2f %s' %(calc_MAE(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.2f %s' %(calc_MAE(sim=lin[flux],obs=clean[flux]),units)
    elif metric == 'mbe':
        if flux in ['Qair','Rainf']:
            s1 = '%.7f %s' %(calc_MBE(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.7f %s'  %(calc_MBE(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.7f %s' %(calc_MBE(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.7f %s' %(calc_MBE(sim=lin[flux],obs=clean[flux]),units)
        else:
            s1 = '%.2f %s' %(calc_MBE(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.2f %s'  %(calc_MBE(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.2f %s' %(calc_MBE(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.2f %s' %(calc_MBE(sim=lin[flux],obs=clean[flux]),units)
    elif metric == 'nsd':
        if flux in ['Qair','Rainf']:
            s1 = '%.4f %s' %(calc_NSD(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.4f %s'  %(calc_NSD(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.4f %s' %(calc_NSD(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.4f %s' %(calc_NSD(sim=lin[flux],obs=clean[flux]),units)
        else:
            s1 = '%.4f %s' %(calc_NSD(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.4f %s'  %(calc_NSD(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.4f %s' %(calc_NSD(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.4f %s' %(calc_NSD(sim=lin[flux],obs=clean[flux]),units)
    elif metric == 'r':
        if flux in ['Qair','Rainf']:
            s1 = '%.4f %s' %(calc_R(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.4f %s'  %(calc_R(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.4f %s' %(calc_R(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.4f %s' %(calc_R(sim=lin[flux],obs=clean[flux]),units)
        else:
            s1 = '%.4f %s' %(calc_R(sim=corr[flux],obs=clean[flux]),units)
            s2 = '%.4f %s'  %(calc_R(sim=era[flux],obs=clean[flux]),units)
            s3 = '%.4f %s' %(calc_R(sim=watch[flux].shift(shift),obs=clean[flux]),units)
            s4 = '%.4f %s' %(calc_R(sim=lin[flux],obs=clean[flux]),units)
    else:
        print(f'input metric not recognised: {metric}')

    ser = pd.Series(data=[float(x.split()[0]) for x in [s1,s2,s3,s4]], index=['corr','era5','wfde5','lin'], name=flux)

    if metric == 'mae':
        sdate,edate = clean.index[0], clean.index[-1]
        psdate,pedate = clean.index[-1] - pd.Timedelta(days=7), clean.index[-1]

        if flux == 'Rainf':

            # cumulative sum plot
            fig2, ax2 = plt.subplots(1,1,figsize=(8,5))
            ax2.set_title(f'{sitename}: {flux} {sample}')

            ax2.text(0.01,0.99, f'period: {sdate} - {edate}', 
                va='top', ha='left', fontsize=8, transform=ax2.transAxes, color='black')

            clean_sum = clean[flux].cumsum()*ts
            corr_sum = corr[flux].cumsum()*3600.
            era_sum = era[flux].cumsum()*3600.
            watch_sum = watch[flux].shift(shift).cumsum()*3600.

            clean_sum.plot(ax=ax2, label=f'observed ({clean_sum.iloc[-1]:.0f} mm)', 
                lw= 1.5, color='black', ls='dashed', zorder=4)
            era_sum.plot(ax=ax2, label=f'ERA5 ({era_sum.iloc[-1]:.0f} mm)', 
                lw= 1., color='blue', ls='solid', zorder=2)
            if plotwatch:
                watch_sum.plot(ax=ax2, label=f'WFDE5 ({watch_sum.iloc[-1]:.0f} mm)', 
                    lw= 1., color='green', ls='solid', zorder=1)
            corr_sum.plot(ax=ax2, label=f'UP: site correction ({corr_sum.iloc[-1]:.0f} mm)',
                lw= 1., color='red', ls='solid', zorder=3)

            ax2.grid(lw=0.5,color='0.8',zorder=-1)
            ax2.legend(loc='upper left',fontsize=7, bbox_to_anchor=(0.01,0.96))
            ax2.set_xlabel(None)
            ax2.set_ylabel('cumulative rainfall (mm)')

            fig2.savefig(f'{sitepath}/precip_plots/{sitename}_{flux}_{sample}_cumulative.{img_fmt}',bbox_inches='tight',dpi=150)
            plt.close('all')

        if sample in ['in-sample']: # do not create plots
            return ser

        # timeseries
        plt.close('all')
        fig, ax = plt.subplots(1,1,figsize=(8,5))
        ax.set_title(f'{sitename}: {flux} {sample}')

        clean[flux][psdate:pedate].plot(ax=ax, label=f'observed (MAE)', color='black', 
            lw=0.5, marker='s',ms=2, zorder=0)
        era[flux][psdate:pedate].plot(ax=ax, label=f'ERA5 ({s2})', color='blue', 
            lw=1, ls='solid', zorder=2)
        if plotwatch:
            watch[flux][psdate:pedate].shift(shift).plot(ax=ax, label=f'WFDE5 ({s3})', color='green', 
                lw=1, ls='solid', zorder=1)
        corr[flux][psdate:pedate].plot(ax=ax, label=f'UP: site correction ({s1})', color='red', 
            lw=1, ls='solid', zorder=3)
        lin[flux][psdate:pedate].plot(ax=ax, label=f'LN: linear correction ({s4})', color='orange', 
            lw=1, ls='solid', zorder=2)

        ax.text(0.01,0.99, f'period: {sdate} - {edate}', 
            va='top', ha='left', fontsize=8, transform=ax.transAxes, color='black')

        ax.legend(loc='upper right',fontsize=7)

        if flux == 'Rainf':

            ax.text(0.01,0.96, 'obs total rain (mm): %.1f' %(clean[flux][psdate:pedate].sum()*ts), 
                va='top', ha='left', fontsize=8, transform=ax.transAxes, color='black')
            ax.text(0.01,0.93, 'ERA total rain (mm): %.1f' %(era[flux][psdate:pedate].sum()*3600.), 
                va='top', ha='left', fontsize=8, transform=ax.transAxes, color='blue')
            ax.text(0.01,0.90, 'UP: site correction total rain (mm): %.1f' %(corr[flux][psdate:pedate].sum()*3600.), 
                va='top', ha='left', fontsize=8, transform=ax.transAxes, color='red')
            ax.text(0.01,0.87, 'LN: linear correction total rain (mm): %.1f' %(lin[flux][psdate:pedate].sum()*3600.), 
                va='top', ha='left', fontsize=8, transform=ax.transAxes, color='orange')
            if plotwatch:
                ax.text(0.01,0.84, 'WFDE5 total rain (mm): %.1f' %(watch[flux][psdate:pedate].sum()*3600.), 
                    va='top', ha='left', fontsize=8, transform=ax.transAxes, color='green')

        ax.grid(lw=0.5,color='0.8',zorder=-1)
        ax.set_xlabel(None)

        # plt.show()
        fig.savefig(f'/{sitepath}/era_correction/{sitename}_{flux}_{sample}_timeseries.{img_fmt}',bbox_inches='tight',dpi=150)

        ########################################

        # diurnal
        plt.close('all')
        

        if publication:
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.set_title('b)',loc='left')
        else:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
            ax.set_title(f'{sitename}: mean diurnal {flux} ({sample})', fontsize=10)

        ax.text(0.99,0.99, f'period: {sdate} - {edate}', 
            va='top', ha='right', fontsize=8, transform=ax.transAxes, color='black')

        clean[flux].groupby(lambda x: x.time).mean().plot(ax=ax, label=f'observed (MAE)', 
            lw= 1.5, color='black', ls='dashed', dashes=(2,2), zorder=4)
        era[flux].groupby(lambda x: x.time).mean().plot(ax=ax, label=f'ERA5 ({s2})', 
            lw= 1., color='blue', ls='solid', zorder=2)
        if plotwatch:
            watch[flux].shift(shift).groupby(lambda x: x.time).mean().plot(ax=ax, label=f'WFDE5 ({s3})', 
                lw= 1., color='green', ls='solid', zorder=1)
        corr[flux].groupby(lambda x: x.time).mean().plot(ax=ax, label=f'UP: site correction ({s1})', 
            lw= 1., color='red', ls='solid', zorder=3)
        lin[flux].groupby(lambda x: x.time).mean().plot(ax=ax, label=f'LN: linear correction ({s4})', 
            lw= 1., color='orange', ls='solid', zorder=2)

        ax.set_xticks(['%s:00' %str(x).zfill(2) for x in range(0,24,2)])
        ax.set_xlim(['00:00','23:59:59'])
        ax.set_xlabel('time (UTC)')
        ax.set_ylabel(f'{flux} [{units}]')

        ax.grid(lw=0.5,color='0.8',zorder=-1)
        ax.legend(loc='upper left',fontsize=7)

        # plt.show()
        if (publication) and (flux=='Tair'):
            fig.savefig(f'/{sitepath}/era_correction/{sitename}_{flux}_{sample}_diurnal_fig4b.{img_fmt}',bbox_inches='tight',dpi=150)
        else:
            fig.savefig(f'/{sitepath}/era_correction/{sitename}_{flux}_{sample}_diurnal.{img_fmt}',bbox_inches='tight',dpi=150)


        plt.close('all')

    return ser

def set_global_attributes(ds,siteattrs,ds_type):

    sitename = siteattrs['sitename']
    long_sitename = siteattrs['long_sitename']

    if ds_type == 'raw_obs':
        title = f'Flux tower observations from {sitename} (before qc)'
        summary = f'Flux tower observations for {long_sitename}, before quality control. Provided for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'
    
    elif ds_type == 'clean_obs':
        title = f'Flux tower observations from {sitename} (after qc)'
        summary = f'Quality controlled flux tower observations for {long_sitename}. Developed for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'

    elif ds_type == 'forcing':
        title = f'Continuous meterological forcing from {sitename}'
        summary = f'Flux tower observations from {long_sitename} after quality control, with gap filling from bias corrected ERA5 surface meteorological data. Developed for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'

    elif ds_type == 'analysis':
        title = f'Flux tower observations from {sitename} (after qc, for model analysis)'
        summary = f'Flux tower observations for {long_sitename}. Used for analysis of land surface models on modelevaluation.org. Do not distribute.'

    elif ds_type == 'era5_raw':
        title = f'Continuous ERA5 meterological data for the grid nearest {sitename}'
        summary = f'ERA5 reanalysis data (single level) for the grid nearest {long_sitename} (1990-2020). Developed for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'

    elif ds_type == 'era5_corrected':
        title = f'Continuous bias corrected ERA5 data for {sitename} (Urban-PLUMBER methods)'
        summary = f'ERA5 reanalysis data (single level), bias corrected for {long_sitename} (1990-2020). Developed for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'

    elif ds_type == 'era5_linear':
        title = f'Continous bias corrected ERA5 data for {sitename} (linear regression)'
        summary = f'ERA5 reanalysis data (single level), bias corrected using linear regression for {long_sitename} (1990-2020). Developed for use in the Urban-PLUMBER model evaluation project. Attribute any use to "Harmonized, gap-filled dataset from 20 urban flux tower sites"'

    else:
        print('WARNING: timeseries ds_type not recognised')
        title = ''
        summary = ''

    # licence information
    unrestricted = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
                    'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','NL-Amsterdam','UK-KingsCollege',
                    'UK-Swindon','US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

    if sitename in unrestricted:
        license = 'CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/'
    else:
        license = 'Restricted dataset - contact data owner - do not distribute'

    # use existing attributes for analysis start date if present
    try:
        time_analysis_start = ds.time_analysis_start
        timestep_number_analysis = ds.timestep_number_analysis
    except Exception:
        time_analysis_start = pd.to_datetime(ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
        timestep_number_analysis = str(len(ds.time))

    ds.attrs = OrderedDict([])

    # global attributes
    ds.attrs['title']                     = title
    ds.attrs['summary']                   = summary
    ds.attrs['sitename']                  = siteattrs['sitename']
    ds.attrs['long_sitename']             = siteattrs['long_sitename']
    ds.attrs['version']                   = siteattrs['out_suffix']
    ds.attrs['keywords']                  = 'urban, flux tower, eddy covariance, observations'
    ds.attrs['conventions']               = 'ALMA, CF, ACDD-1.3'
    ds.attrs['featureType']               = 'timeSeries'
    ds.attrs['license']                   = license
    ds.attrs['time_coverage_start']       = pd.to_datetime(ds.time[0].values).strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['time_coverage_end']         = pd.to_datetime(ds.time[-1].values).strftime('%Y-%m-%d %H:%M:%S')
    if ds_type in ['forcing','analysis']:
        ds.attrs['time_analysis_start']   = time_analysis_start
    ds.attrs['time_shown_in']             = 'UTC'
    ds.attrs['local_utc_offset_hours']    = siteattrs['local_utc_offset_hours']
    ds.attrs['timestep_interval_seconds'] = (ds.time[1].values - ds.time[0].values)/np.timedelta64(1, 's')
    if ds_type in ['forcing','analysis']:
        ds.attrs['timestep_number_spinup']= str(len(ds.time) - int(timestep_number_analysis))
    ds.attrs['timestep_number_analysis']  = str(timestep_number_analysis)
    ds.attrs['project']                   = 'Urban-PLUMBER: "Harmonized, gap-filled dataset from 20 urban flux tower sites"'
    ds.attrs['project_contact']           = 'Mathew Lipson (m.lipson@unsw.edu.au), Sue Grimmond (c.s.grimmond@reading.ac.uk), Martin Best (martin.best@metoffice.gov.uk)'
    ds.attrs['observations_contact']      = siteattrs['obs_contact']
    ds.attrs['observations_reference']    = siteattrs['obs_reference']
    ds.attrs['date_created']              = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['source']                    = 'https://github.com/matlipson/urban-plumber_pipeline'
    if ds_type in ['forcing','analysis','era5_raw','era5_corrected','era5_linear']:
        ds.attrs['other_references']      = 'ERA5: Copernicus Climate Change Service (C3S) (2017): https://cds.climate.copernicus.eu/cdsapp#!/home NCI Australia: http://doi.org/10.25914/5f48874388857'
        ds.attrs['acknowledgements']      = 'Contains modified Copernicus Climate Change Service Information 2021 (ERA5 hourly data on single levels). Data from replica hosted by NCI Australia.'
    if ds_type in ['forcing','analysis','raw_obs','clean_obs']:
        ds.attrs['comment']               = siteattrs['obs_comment']
    ds.attrs['history']                   = siteattrs['history']

    return ds

def write_netcdf_file(ds,fpath_out):
    '''Writes ds to file and corrects xarray time units string problem

    ds (dataset): dataset to write to file
    fpath_out (str): file path for writing'''

    encoding = {var: {'zlib':True} for var in ds.data_vars}
    ds.to_netcdf(fpath_out, format='NETCDF4', mode='w', encoding=encoding) # do not use unlimited dimensions (increases file size x10)

    # manually add seconds if time units fall on midnight
    if ds.time.encoding['units'][-8:] == '00:00:00':
        '''Corrects xarray issue which strips 00:00:00 from time.units
        i.e. "seconds from 1993-01-01" -> "seconds since 1993-01-01 00:00:00" '''
        f = netCDF4.Dataset(fpath_out, 'r+')
        f.variables['time'].units = f.variables['time'].units + ' 00:00:00'
        f.close()

    return

def write_netcdf_to_text_file(ds,fpath_out):
    ''' Writes text file from xarray dataset

    ds (dataset): dataset to write to file
    fpath_out (str): file path for writing'''

    print('creating %s from nc file' %fpath_out.split('/')[-1])

    formats = {
            'SWdown'  : '{:10.3f}'.format,
            'LWdown'  : '{:10.3f}'.format,
            'Wind_E'  : '{:10.3f}'.format,
            'Wind_N'  : '{:10.3f}'.format,
            'PSurf'   : '{:10.1f}'.format,
            'Tair'    : '{:10.3f}'.format,
            'Qair'    : '{:12.8f}'.format,
            'Rainf'   : '{:12.8f}'.format,
            'Snowf'   : '{:12.8f}'.format,

            'SWup'    : '{:10.3f}'.format,
            'LWup'    : '{:10.3f}'.format,
            'Qle'     : '{:10.3f}'.format,
            'Qh'      : '{:10.3f}'.format,

            'Qtau'    : '{:10.3f}'.format,
            'SoilTemp': '{:10.3f}'.format,
            'Tair2m'  : '{:10.3f}'.format,

            'SWdown_qc'  : '{:10d}'.format,
            'LWdown_qc'  : '{:10d}'.format,
            'Wind_E_qc'  : '{:10d}'.format,
            'Wind_N_qc'  : '{:10d}'.format,
            'PSurf_qc'   : '{:10d}'.format,
            'Tair_qc'    : '{:10d}'.format,
            'Qair_qc'    : '{:10d}'.format,
            'Rainf_qc'   : '{:10d}'.format,
            'Snowf_qc'   : '{:10d}'.format,

            'SWup_qc'    : '{:10d}'.format, 
            'LWup_qc'    : '{:10d}'.format,
            'Qle_qc'     : '{:10d}'.format,
            'Qh_qc'      : '{:10d}'.format, 

            'Qtau_qc'    : '{:10d}'.format, 
            'SoilTemp_qc': '{:10d}'.format, 
            'Tair2m_qc'  : '{:10d}'.format, 
            }

    var_list = list(ds.data_vars.keys())
    var_list = [x for x in var_list if x in list(formats.keys())]

    # get units information
    var_units = {}
    for key in var_list:
        try:
            var_units[key] = ds[key].units
        except Exception:
            var_units[key] = '-'

    # select dataframe
    df = ds.squeeze().to_dataframe()[var_list]

    df.index.name = None

    print(f'writing out text file')
    print(df.head())
    with open(fpath_out, 'w') as file:
        file.writelines(df.to_string(formatters=formats,header=True,index=True))

    print('add header comments and metainfo')
    with open(fpath_out, 'r') as f1: # read file
        origdata = f1.read()
        with open(fpath_out, 'w') as f2: # re-write with header info

            for key,item in ds.attrs.items():
                f2.write(f'# {key} = {item}\n')

            f2.write(f'# see sitedata csv for site characteristics\n')
            f2.write(f'# units = {", ".join([f"{key}: {values}" for key,values in var_units.items()])}\n')
            f2.write(f'# quality control (qc) flags: 0:observed, 1:gapfilled_from_obs, 2:gapfilled_derived_from_era5, 3:missing\n') 
            f2.write(f'# \n')
            f2.write(f'#     Date     Time {origdata[20:]}')

    return

def get_wfde5_data(sitename,sitedata,syear,eyear,wfdepath):
    '''get wfdei netcdf variables from gadi and combine into xarray dataset

    Parameters:
    ----------
    ncvars      [list of strings]   era5 variable short names
    lat         [float]             local site latitude
    lon         [float]             local site longitude
    syear       [integer]           start year
    eyear       [integer]           end year

    Returns:
    ----------
    sitedata    [xr DataSet]        all requested era5 variables and years at single lat/lon point

    '''

    # ncvars = ['Tair']
    ncvars = ['LWdown','PSurf','Qair','Rainf','Snowf','SWdown','Tair','Wind']

    ds = xr.Dataset()
    years = [str(year) for year in range(syear,eyear+1)]

    # loop through variables
    for ncvar in ncvars:
        print('collecting %s for %s - %s' %(ncvar,syear,eyear))


        files = []
        # get list of files in path using glob wildcard
        for year in years:
            fname = '%s/%s/%s_*_%s*' %(wfdepath,ncvar,ncvar,year)
            print('finding %s' %fname)
            files = files + sorted(glob.glob(fname))
        print(f'concatenating {len(files)} monthly WFDE5 {ncvar} files')
        # open datasets, select nearest grid to site, then concatenate on time
        tmp_ds = xr.concat( [xr.open_dataset(file).sel(lat=sitedata['latitude'],
                                                       lon=sitedata['longitude'],
                                                       method='nearest') for file in files], dim='time' )

        # merge variables into single file
        ds = xr.merge([ds,tmp_ds])
        longname = tmp_ds[list(tmp_ds.keys())[0]].long_name.lower()
        print('done merging %s (%s)' %(longname,ncvar))


    # get site elevation information
    data = xr.open_dataset('%s/ASurf/ASurf_era5_wfde5_v1-1_cru.nc' %wfdepath).sel(
                lat=sitedata['latitude'], lon=sitedata['longitude'], method='nearest')
    ds['ASurf'] = data['ASurf']

    ds.attrs['title'] = 'WATCH Forcing Data methodology applied to ERA5 data for %s' %sitename
    ds.attrs['reference'] = data.reference

    ds = set_variable_attributes(ds)

    return ds 

def assign_sitedata(siteattrs):

    # loading sitedata
    fpath = f'{siteattrs["sitepath"]}/{siteattrs["sitename"]}_sitedata_{siteattrs["sitedata_suffix"]}.csv'
    sitedata_full = pd.read_csv(fpath, index_col=1, delimiter=',')
    sitedata = pd.to_numeric(sitedata_full['value'])

    dims2D = ['y','x']
    coords2D = {'y':[1], 'x':[1]}

    # create dataset
    ds = xr.Dataset({
        ###########################################################################
        ########################### sitedata parameters ###########################
        # 1
        'latitude': xr.DataArray(
            data   = [[sitedata['latitude']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Latitude',
                'standard_name' : 'latitude',
                'units'         : 'degrees_north',
                'source'        : sitedata_full.loc['latitude','source'],
                'source_doi'    : sitedata_full.loc['latitude','doi'],
                }
            ),
        # 2
        'longitude': xr.DataArray(
            data   = [[sitedata['longitude']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Longitude',
                'standard_name' : 'longitude',
                'units'         : 'degrees_east',
                'source'        : sitedata_full.loc['longitude','source'],
                'source_doi'    : sitedata_full.loc['longitude','doi'],
                }
            ),
        # 3
        'ground_height': xr.DataArray(
            data   = [[sitedata['ground_height']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Ground height above sea level',
                'units'         : 'm',
                'source'        : sitedata_full.loc['ground_height','source'],
                'source_doi'    : sitedata_full.loc['ground_height','doi'],
                }
            ),
        # 4
        'measurement_height_above_ground': xr.DataArray(
            data   = [[sitedata['measurement_height_above_ground']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Measurement height above ground',
                'units'         : 'm',
                'source'        : sitedata_full.loc['measurement_height_above_ground','source'],
                'source_doi'    : sitedata_full.loc['measurement_height_above_ground','doi'],
                }
            ),
        # 5
        'impervious_area_fraction': xr.DataArray(
            data   = [[sitedata['impervious_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Impervious area (built) fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['impervious_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['impervious_area_fraction','doi'],
                }
            ),
        # 6
        'tree_area_fraction': xr.DataArray(
            data   = [[sitedata['tree_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Tree area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['tree_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['tree_area_fraction','doi'],
                }
            ),
        # 7
        'grass_area_fraction': xr.DataArray(
            data   = [[sitedata['grass_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Grass area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['grass_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['grass_area_fraction','doi'],
                }
            ),
        # 8
        'bare_soil_area_fraction': xr.DataArray(
            data   = [[sitedata['bare_soil_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Bare soil area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['bare_soil_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['bare_soil_area_fraction','doi'],
                }
            ),
        # 9
        'water_area_fraction': xr.DataArray(
            data   = [[sitedata['water_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Water area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['water_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['water_area_fraction','doi'],
                }
            ),
        ###########################################################################
        # 10
        'roof_area_fraction': xr.DataArray(
            data   = [[sitedata['roof_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Roof area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['roof_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['roof_area_fraction','doi'],
                }
            ),
        # 11
        'road_area_fraction': xr.DataArray(
            data   = [[sitedata['road_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Road area fraction',
                'units'         : '1',
                'source'        : sitedata_full.loc['road_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['road_area_fraction','doi'],
                }
            ),
        # 12
        'other_paved_area_fraction': xr.DataArray(
            data   = [[sitedata['other_paved_area_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Paved area fraction (other than roads)',
                'units'         : '1',
                'source'        : sitedata_full.loc['other_paved_area_fraction','source'],
                'source_doi'    : sitedata_full.loc['other_paved_area_fraction','doi'],
                }
            ),
        # 13
        'building_mean_height': xr.DataArray(
            data   = [[sitedata['building_mean_height']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Building mean height',
                'units'         : 'm',
                'source'        : sitedata_full.loc['building_mean_height','source'],
                'source_doi'    : sitedata_full.loc['building_mean_height','doi'],
                }
            ),
        # 14
        'tree_mean_height': xr.DataArray(
            data   = [[sitedata['tree_mean_height']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Tree mean height',
                'standard_name' : 'canopy_height',
                'units'         : 'm',
                'source'        : sitedata_full.loc['tree_mean_height','source'],
                'source_doi'    : sitedata_full.loc['tree_mean_height','doi'],
                }
            ),
        # 15
        'roughness_length_momentum': xr.DataArray(
            data   = [[sitedata['roughness_length_momentum']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Roughness length for momentum as reported in literature',
                'standard_name' : 'surface_roughness_length_for_momentum_in_air',
                'units'         : 'm',
                'source'        : sitedata_full.loc['roughness_length_momentum','source'],
                'source_doi'    : sitedata_full.loc['roughness_length_momentum','doi'],
                }
            ),
        # 16
        'displacement_height': xr.DataArray(
            data   = [[sitedata['displacement_height']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Zero-plane displacement height as reported in literature',
                'units'         : 'm',
                'source'        : sitedata_full.loc['displacement_height','source'],
                'source_doi'    : sitedata_full.loc['displacement_height','doi'],
                }
            ),
        # 17
        'canyon_height_width_ratio': xr.DataArray(
            data   = [[sitedata['canyon_height_width_ratio']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Street canyon height to width ratio',
                'units'         : '1',
                'source'        : sitedata_full.loc['canyon_height_width_ratio','source'],
                'source_doi'    : sitedata_full.loc['canyon_height_width_ratio','doi'],
                }
            ),
        # 18
        'wall_to_plan_area_ratio': xr.DataArray(
            data   = [[sitedata['wall_to_plan_area_ratio']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Ratio between walls in contact with the atmosphere and the horizontal surface',
                'units'         : '1',
                'source'        : sitedata_full.loc['wall_to_plan_area_ratio','source'],
                'source_doi'    : sitedata_full.loc['wall_to_plan_area_ratio','doi'],
                }
            ),
        # 19
        'average_albedo_at_midday': xr.DataArray(
            data   = [[sitedata['average_albedo_at_midday']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Average surface albedo at midday',
                'units'         : '1',
                'source'        : sitedata_full.loc['average_albedo_at_midday','source'],
                'source_doi'    : sitedata_full.loc['average_albedo_at_midday','doi'],
                }
            ),
        # 20
        'resident_population_density': xr.DataArray(
            data   = [[sitedata['resident_population_density']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Population density of residents',
                'units'         : 'person/km2',
                'source'        : sitedata_full.loc['resident_population_density','source'],
                'source_doi'    : sitedata_full.loc['resident_population_density','doi'],
                }
            ),
        # 21
        'anthropogenic_heat_flux_mean': xr.DataArray(
            data   = [[sitedata['anthropogenic_heat_flux_mean']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Annual mean of heat fluxes due to anthropogenic energy consumption',
                'units'         : 'W/m2',
                'source'        : sitedata_full.loc['anthropogenic_heat_flux_mean','source'],
                'source_doi'    : sitedata_full.loc['anthropogenic_heat_flux_mean','doi'],
                }
            ),
        # 22
        'topsoil_clay_fraction': xr.DataArray(
            data   = [[sitedata['topsoil_clay_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Clay fraction at topsoil by weight',
                'units'         : '1',
                'source'        : sitedata_full.loc['topsoil_clay_fraction','source'],
                'source_doi'    : sitedata_full.loc['topsoil_clay_fraction','doi'],
                }
            ),
        # 23
        'topsoil_sand_fraction': xr.DataArray(
            data   = [[sitedata['topsoil_sand_fraction']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Sand fraction at topsoil by weight',
                'units'         : '1',
                'source'        : sitedata_full.loc['topsoil_sand_fraction','source'],
                'source_doi'    : sitedata_full.loc['topsoil_sand_fraction','doi'],
                }
            ),
        # 24
        'topsoil_bulk_density': xr.DataArray(
            data   = [[sitedata['topsoil_bulk_density']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Soil bulk density at topsoil',
                'units'         : 'kg/m3',
                'source'        : sitedata_full.loc['topsoil_bulk_density','source'],
                'source_doi'    : sitedata_full.loc['topsoil_bulk_density','doi'],
                }
            ),
        # 25
        'building_height_standard_deviation': xr.DataArray(
            data   = [[sitedata['building_height_standard_deviation']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Standard deviation of building heights',
                'units'         : 'm',
                'source'        : sitedata_full.loc['building_height_standard_deviation','source'],
                'source_doi'    : sitedata_full.loc['building_height_standard_deviation','doi'],
                }
            ),
        # 26
        'roughness_length_momentum_mac': xr.DataArray(
            data   = [[sitedata['roughness_length_momentum_mac']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Roughness length for momentum from Macdonald morphometric method',
                'units'         : 'm',
                'source'        : sitedata_full.loc['roughness_length_momentum_mac','source'],
                'source_doi'    : sitedata_full.loc['roughness_length_momentum_mac','doi'],
                }
            ),
        # 27
        'displacement_height_mac': xr.DataArray(
            data   = [[sitedata['displacement_height_mac']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Zero-plane displacement height from Macdonald morphometric method',
                'units'         : 'm',
                'source'        : sitedata_full.loc['displacement_height_mac','source'],
                'source_doi'    : sitedata_full.loc['displacement_height_mac','doi'],
                }
            ),
        # 28
        'roughness_length_momentum_kanda': xr.DataArray(
            data   = [[sitedata['roughness_length_momentum_kanda']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Roughness length for momentum from Kanda morphometric method',
                'units'         : 'm',
                'source'        : sitedata_full.loc['roughness_length_momentum_kanda','source'],
                'source_doi'    : sitedata_full.loc['roughness_length_momentum_kanda','doi'],
                }
            ),
        # 29
        'displacement_height_kanda': xr.DataArray(
            data   = [[sitedata['displacement_height_kanda']]],
            dims   = dims2D,
            coords = coords2D,
            attrs  = {
                'long_name'     : 'Zero-plane displacement height from Kanda morphometric method',
                'units'         : 'm',
                'source'        : sitedata_full.loc['displacement_height_kanda','source'],
                'source_doi'    : sitedata_full.loc['displacement_height_kanda','doi'],
                }
            ),
        })

    return ds

def build_new_dataset_forcing(data):
    '''
    Builds xarray dataset in form complying witih Urban-PLUMBER protocol.
    Update sections with "<-enter data here" 

    Input:
    ------
    data (dataset):         data to use as xarray dataset

    Output
    ------
    ds (dataset):           new dataset construct
    '''

    ntimes = len(data.time)
    ndims = 1

    ###########################################################################

    dims1D = ['time']
    coords1D = {'time': data.time.values}

    dims2D = ['y','x']
    coords2D = {'y':[1], 'x':[1]}

    reshape3D = lambda x: x.reshape(ntimes, ndims, ndims)
    dims3D   = ['time','y','x']
    coords3D = {'time': data.time.values,'y':[1], 'x':[1]}

    # 4D variables (CABLE )
    reshape4D = lambda x: x.reshape(ntimes, ndims, ndims, ndims)
    dims4D   = ['time','z','y','x']
    coords4D = {'time': data.time.values, 'z':[1], 'y':[1], 'x':[1]}

    # create dataset
    ds = xr.Dataset({
        ###########################################################################
        ########################## forcing data variables #########################
        'SWdown': xr.DataArray(
                    data   = reshape3D(np.where(data.SWdown.values>0.01, data.SWdown.values, 0)),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'LWdown': xr.DataArray(
                    data   = reshape3D(data.LWdown.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Tair': xr.DataArray(
                    data   = reshape3D(data.Tair.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Qair': xr.DataArray(
                    data   = reshape3D(data.Qair.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'PSurf': xr.DataArray(
                    data   = reshape3D(data.PSurf.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Rainf': xr.DataArray(
                    data   = reshape3D(np.where(data.Rainf.values>0., data.Rainf.values, 0)),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Snowf': xr.DataArray(
                    data   = reshape3D(np.where(data.Snowf.values>0., data.Snowf.values, 0)),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Wind_N': xr.DataArray(
                    data   = reshape3D(data.Wind_N.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Wind_E': xr.DataArray(
                    data   = reshape3D(data.Wind_E.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        #######################################################################
        'SWdown_qc': xr.DataArray(
                    data   = data.SWdown_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'LWdown_qc': xr.DataArray(
                    data   = data.LWdown_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Tair_qc': xr.DataArray(
                    data   = data.Tair_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Qair_qc': xr.DataArray(
                    data   = data.Qair_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'PSurf_qc': xr.DataArray(
                    data   = data.PSurf_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Rainf_qc': xr.DataArray(
                    data   = data.Rainf_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Snowf_qc': xr.DataArray(
                    data   = data.Snowf_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Wind_N_qc': xr.DataArray(
                    data   = data.Wind_N_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Wind_E_qc': xr.DataArray(
                    data   = data.Wind_E_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
                },
            )
    
    return ds

#########################################################################

def build_new_dataset_analysis(data,sitedata):
    '''
    Builds xarray dataset in form complying witih Urban-PLUMBER protocol.
    Update sections with "<-enter data here" 

    Input:
    ------
    data (dataset):         data to use as xarray dataset

    Output
    ------
    ds (dataset):           new dataset construct
    '''

    ntimes = len(data.index)
    ndims = 1

    ###########################################################################

    dims1D = ['time']
    coords1D = {'time': data.index}

    dims2D = ['y','x']
    dims2D = {'y':[1], 'x':[1]}

    reshape3D = lambda x: x.reshape(ntimes, ndims, ndims)
    dims3D   = ['time','y','x']
    coords3D = {'time': data.index,'y':[1], 'x':[1]}

    # 4D variables (CABLE )
    reshape4D = lambda x: x.reshape(ntimes, ndims, ndims, ndims)
    dims4D   = ['time','y','x','z']
    coords4D = {'time': data.index, 'y':[1], 'x':[1], 'z':[1]}

    # set new net radiation data
    data['SWnet'] = data['SWdown'] - data['SWup']
    data['LWnet'] = data['LWdown'] - data['LWup']
    data['SWnet'] = np.where(data['SWnet']<0,0.,data['SWnet'])

    # set net radiation flags
    data['SWnet_qc'] = np.where(data['SWnet'].isna(),3,0)
    data['LWnet_qc'] = np.where(data['LWnet'].isna(),3,0)
    data['SWnet_qc'] = np.where((data['SWdown_qc']==1) | (data['SWup_qc']==1), 1, data['SWnet_qc'].values)
    data['LWnet_qc'] = np.where((data['LWdown_qc']==1) | (data['LWup_qc']==1), 1, data['LWnet_qc'].values)

    # create dataset
    ds = xr.Dataset({
        ###########################################################################
        ########################### dimension variables ###########################
        'latitude': xr.DataArray(
            data   = [[sitedata['latitude']]],
            dims   = dims2D,
            coords = dims2D,
            ),
        'longitude': xr.DataArray(
            data   = [[sitedata['longitude']]],
            dims   = dims2D,
            coords = dims2D,
            ),
        'measurement_height_above_ground': xr.DataArray(
            data   = [[sitedata['measurement_height_above_ground']]],
            dims   = dims2D,
            coords = dims2D,
            ),
        ###########################################################################
        ########################## forcing data variables #########################
        'SWnet': xr.DataArray(
                    data   = reshape3D(data.SWnet.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'LWnet': xr.DataArray(
                    data   = reshape3D(data.LWnet.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'SWup': xr.DataArray(
                    data   = reshape3D(data.SWup.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'LWup': xr.DataArray(
                    data   = reshape3D(data.LWup.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Qle': xr.DataArray(
                    data   = reshape3D(data.Qle.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        'Qh': xr.DataArray(
                    data   = reshape3D(data.Qh.values),
                    dims   = dims3D,
                    coords = coords3D,
                    ),
        #######################################################################
        'SWnet_qc': xr.DataArray(
                    data   = data.SWnet_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'LWnet_qc': xr.DataArray(
                    data   = data.LWnet_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'SWup_qc': xr.DataArray(
                    data   = data.SWup_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'LWup_qc': xr.DataArray(
                    data   = data.LWup_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Qle_qc': xr.DataArray(
                    data   = data.Qle_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
        'Qh_qc': xr.DataArray(
                    data   = data.Qh_qc.values.astype('int8'),
                    dims   = dims1D,
                    coords = coords1D,
                    ),
                },
            )

    ds = set_variable_attributes(ds)
    
    return ds

def convert_local_to_utc(df,offset_from_utc):

    print('converting to utc time')
    tzhrs = int(offset_from_utc)
    tzmin = int((offset_from_utc - int(offset_from_utc))*60)
    df.index = df.index - np.timedelta64(tzhrs,'h') - np.timedelta64(tzmin,'m')

    return df

def calc_midday_albedo(clean_ds,offset_from_utc):

    obs = clean_ds.squeeze().to_dataframe()

    ###############################################################################
    # calculate midday albedo
    obs['Albedo'] = obs['SWup']/obs['SWdown']
    obs['Albedo'] = obs['Albedo'].replace([np.inf,-np.inf],np.nan)

    albedo_stime = (pd.Timestamp('11:45') - pd.Timedelta('%s hours' %offset_from_utc)).strftime('%H:%M')
    albedo_etime = (pd.Timestamp('12:15') - pd.Timedelta('%s hours' %offset_from_utc)).strftime('%H:%M')

    print('')
    alb_midday = obs['Albedo'].between_time(albedo_stime,albedo_etime).mean()
    print('observed mean midday albedo: %0.3f' %(alb_midday))
    alb_midday = obs['Albedo'].between_time(albedo_stime,albedo_etime).median()
    print('observed median midday albedo: %0.3f' %(alb_midday))
    print('')

    return

def set_variable_attributes(ds):

    ##########################################################################
    ############################## coordinates ###############################

    key = 'time'
    if key in ds.coords.keys():
        seconds_since_str = pd.to_datetime(str(ds.time[0].values)).strftime('%Y-%m-%d %H:%M:%S')
        ds[key].attrs['long_name']     = 'Time'
        ds[key].attrs['standard_name'] = 'time'
        ds[key].encoding['units']      = 'seconds since %s' %seconds_since_str
        ds[key].encoding['calendar']   = 'standard'
        ds[key].encoding['dtype']      = 'i8'
        ds[key].encoding['_FillValue'] =  missing_int8

    ##########################################################################
    ################### critical energy balance components ###################
    key = 'longitude'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Longitude'
        ds[key].attrs['standard_name'] = 'longitude'
        ds[key].attrs['units']         = 'degrees_east'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'latitude'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Latitude'
        ds[key].attrs['standard_name'] = 'latitude'
        ds[key].attrs['units']         = 'degrees_north'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SWnet'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Net shortwave radiation (positive downward)'
        ds[key].attrs['standard_name'] = 'surface_net_downward_shortwave_flux'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'LWnet'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Net longwave radiation (positive downward)'
        ds[key].attrs['standard_name'] = 'surface_net_downward_longwave_flux'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qle'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Latent heat flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upward_latent_heat_flux'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qh'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Sensible heat flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upward_sensible_heat_flux'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qanth'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Anthropogenic heat flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upward_heat_flux_due_to_anthropogenic_energy_consumption'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qstor'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Net storage heat flux in all materials (increase)'
        ds[key].attrs['standard_name'] = 'surface_thermal_storage_heat_flux'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SWup'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Upwelling shortwave radiation flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upwelling_shortwave_flux_in_air'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'LWup'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Upwelling longwave radiation flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upwelling_longwave_flux_in_air'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    ###########################################################################
    ################### additional energy balance compoenents #################

    key = 'Qg'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Ground heat flux (positive downward)'
        ds[key].attrs['standard_name'] = 'downward_heat_flux_at_ground_level_in_soil'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qanth_Qh'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Anthropogenic sensible heat flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upward_sensible_heat_flux_due_to_anthropogenic_energy_consumption'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qanth_Qle'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Anthropogenic latent heat flux (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_upward_latent_heat_flux_due_to_anthropogenic_energy_consumption'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qtau'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Momentum flux (positive downward)'
        ds[key].attrs['standard_name'] = 'magnitude_of_surface_downward_stress'
        ds[key].attrs['units']         = 'N/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    ############################################################################
    ##################### general water balance components #####################

    key = 'Snowf'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Snowfall rate (positive downward)'
        ds[key].attrs['standard_name'] = 'snowfall_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Rainf'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Rainfall rate (positive downward)'
        ds[key].attrs['standard_name'] = 'rainfall_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Evap'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Total evapotranspiration (positive upward)'
        ds[key].attrs['standard_name'] = 'surface_evapotranspiration'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qs'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Surface runoff (positive out of gridcell)'
        ds[key].attrs['standard_name'] = 'surface_runoff_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qsb'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Subsurface runoff (positive out of gridcell)'
        ds[key].attrs['standard_name'] = 'subsurface_runoff_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qsm'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Subsurface runoff (positive out of gridcell)'
        ds[key].attrs['standard_name'] = 'subsurface_runoff_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qfz'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Re-freezing of water in the snow (liquid to solid)'
        ds[key].attrs['standard_name'] = 'surface_snow_and_ice_refreezing_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'DelSoilMoist'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Change in soil moisture (increase)'
        ds[key].attrs['standard_name'] = 'change_over_time_in_mass_content_of_water_in_soil'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'DelSWE'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Change in snow water equivalent (increase)'
        ds[key].attrs['standard_name'] = 'change_over_time_in_surface_snow_and_ice_amount'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'DelIntercept'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Change in interception storage (increase)'
        ds[key].attrs['standard_name'] = 'change_over_time_in_canopy_water_amount'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qirrig'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Anthropogenic water flux from irrigation (increase)'
        ds[key].attrs['standard_name'] = 'surface_downward_mass_flux_of_water_due_to_irrigation'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].encoding['_FillValue'] =  missing_float

    ##########################################################################
    ########################## surface state variables ########################

    key = 'SnowT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Snow surface temperature'
        ds[key].attrs['standard_name'] = 'surface_snow_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'snow'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'VegT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Vegetation canopy temperature'
        ds[key].attrs['standard_name'] = 'surface_canopy_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'vegetation'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'BaresoilT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Temperature of bare soil'
        ds[key].attrs['standard_name'] = 'surface_ground_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'baresoil'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'AvgSurfT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Average surface temperature (skin)'
        ds[key].attrs['standard_name'] = 'surface_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'RadT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Surface radiative temperature'
        ds[key].attrs['standard_name'] = 'surface_radiative_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Albedo'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Surface albedo'
        ds[key].attrs['standard_name'] = 'surface_albedo'
        ds[key].attrs['units']         = '1'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SWE'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Snow water equivalent'
        ds[key].attrs['standard_name'] = 'surface_albedo'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SurfStor'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Surface water storage'
        ds[key].attrs['standard_name'] = 'surface_water_amount_assuming_no_snow'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SnowFrac'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Snow covered fraction'
        ds[key].attrs['standard_name'] = 'surface_snow_area_fraction'
        ds[key].attrs['units']         = '1'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SAlbedo'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Snow albedo'
        ds[key].attrs['standard_name'] = 'snow_and_ice_albedo'
        ds[key].attrs['units']         = '1'
        ds[key].attrs['subgrid']       = 'snow'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'CAlbedo'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Vegetation canopy albedo'
        ds[key].attrs['standard_name'] = 'canopy_albedo'
        ds[key].attrs['units']         = '1'
        ds[key].attrs['subgrid']       = 'vegetation'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'UAlbedo'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Urban canopy albedo'
        ds[key].attrs['standard_name'] = 'urban_albedo'
        ds[key].attrs['units']         = '1'
        ds[key].attrs['subgrid']       = 'urban'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'LAI'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Leaf area index'
        ds[key].attrs['standard_name'] = 'leaf_area_index'
        ds[key].attrs['units']         = '1'
        ds[key].attrs['subgrid']       = 'vegetation'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'RoofSurfT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Roof surface temperature (skin)'
        ds[key].attrs['standard_name'] = 'surface_roof_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'roof'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'WallSurfT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Wall surface temperature (skin)'
        ds[key].attrs['standard_name'] = 'surface_wall_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'roof'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'RoadSurfT'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Road surface temperature (skin)'
        ds[key].attrs['standard_name'] = 'surface_wall_skin_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'road'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'TairSurf'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Near surface air temperature (2m)'
        ds[key].attrs['standard_name'] = 'air_temperature_near_surface'
        ds[key].attrs['units']         = 'K'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Tair2m'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Near surface air temperature (2m)'
        ds[key].attrs['standard_name'] = 'air_temperature_near_surface'
        ds[key].attrs['units']         = 'K'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'TairCanyon'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Air temperature in street canyon (bulk)'
        ds[key].attrs['standard_name'] = 'air_temperature_in_street_canyon'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'canyon'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'TairBuilding'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Air temperature in buildings (bulk)'
        ds[key].attrs['standard_name'] = 'air_temperature_in_buildings'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'building'
        ds[key].encoding['_FillValue'] =  missing_float

    ###########################################################################
    ######################## Sub-surface state variables ######################

    key = 'SoilMoist'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Average layer soil moisture'
        ds[key].attrs['standard_name'] = 'moisture_content_of_soil_layer'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].attrs['subgrid']       = 'soil'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SoilTemp'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Average layer soil temperature'
        ds[key].attrs['standard_name'] = 'soil_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].attrs['subgrid']       = 'soil'
        ds[key].encoding['_FillValue'] =  missing_float

        try:
            if ds.sitename in ['SG-TelokKurau','US-WestPhoenix']:
                ds[key].attrs['depth'] = '0.02 m' 
            if ds.sitename in ['US-Minneapolis1','US-Minneapolis2','CA-Sunset']:
                ds[key].attrs['depth'] = '0.05 m'
            if ds.sitename in ['US-Baltimore']:
                ds[key].attrs['depth'] = '0.10 m'
        except Exception:
            pass

    ###########################################################################
    ########################## Evaporation components #########################

    key = 'TVeg'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Vegetation transpiration'
        ds[key].attrs['standard_name'] = 'transpiration_flux'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].attrs['subgrid']       = 'vegetation'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'ESoil'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Bare soil evaporation'
        ds[key].attrs['standard_name'] = 'liquid_water_evaporation_flux_from_soil'
        ds[key].attrs['units']         = 'kg/m2/s'
        ds[key].attrs['subgrid']       = 'baresoil'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'RootMoist'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Root zone soil moisture'
        ds[key].attrs['standard_name'] = 'mass_content_of_water_in_soil_defined_by_root_depth'
        ds[key].attrs['units']         = 'kg/m2'
        ds[key].attrs['subgrid']       = 'soil'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'SoilWet'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Total soil wetness'
        ds[key].attrs['standard_name'] = 'relative_soil_moisture_content_above_wilting_point'
        ds[key].attrs['units']         = '1'
        ds[key].attrs['subgrid']       = 'soil'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'ACond'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Aerodynamic conductance'
        ds[key].attrs['standard_name'] = 'inverse_aerodynamic_resistance'
        ds[key].attrs['units']         = 'm/s'
        ds[key].attrs['subgrid']       = 'vegetation'
        ds[key].encoding['_FillValue'] =  missing_float

    ###########################################################################
    ########################## forcing data variables #########################

    key = 'SWdown'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Downward shortwave radiation at measurement height'
        ds[key].attrs['standard_name'] = 'surface_downwelling_shortwave_flux_in_air'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'LWdown'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Downward longwave radiation at measurement height'
        ds[key].attrs['standard_name'] = 'surface_downwelling_longwave_flux_in_air'
        ds[key].attrs['units']         = 'W/m2'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Tair'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Air temperature at measurement height'
        ds[key].attrs['standard_name'] = 'air_temperature'
        ds[key].attrs['units']         = 'K'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Qair'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Specific humidity at measurement height'
        ds[key].attrs['standard_name'] = 'surface_specific_humidity'
        ds[key].attrs['units']         = 'kg/kg'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'PSurf'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Air pressure at measurement height'
        ds[key].attrs['standard_name'] = 'surface_air_pressure'
        ds[key].attrs['units']         = 'Pa'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Wind'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Wind speed at measurement height'
        ds[key].attrs['standard_name'] = 'wind_speed'
        ds[key].attrs['units']         = 'm/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Wind_N'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Northward wind component at measurement height'
        ds[key].attrs['standard_name'] = 'northward_wind'
        ds[key].attrs['units']         = 'm/s'
        ds[key].encoding['_FillValue'] =  missing_float

    key = 'Wind_E'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Eastward wind component at measurement height'
        ds[key].attrs['standard_name'] = 'eastward_wind'
        ds[key].attrs['units']         = 'm/s'
        ds[key].encoding['_FillValue'] =  missing_float

    ######################################################################
    ########################## qc flag variables #########################

    for key in ds.keys():
        if key[-2:] == 'qc':
            ds[key].attrs['long_name']     = 'Quality control (qc) flag for %s' %key[:-3]
            ds[key].attrs['flag_values']   = '0, 1, 2, 3'
            ds[key].attrs['flag_meanings'] = '0: observed, 1: gapfilled_from_obs, 2: gapfilled_derived_from_era5, 3:missing'
            ds[key].encoding['_FillValue'] =  missing_int8
            ds[key].encoding['dtype']      = 'int8'
            ds[key].values                 = ds[key].values.astype('int8')
            ds['%s' %key[:-3]].attrs['ancillary_variables'] = key

    ######################################################################
    ################################ other ###############################

    key = 'measurement_height_above_ground'
    if key in ds.keys():
        ds[key].attrs['long_name']     = 'Measurement height above ground'
        ds[key].attrs['standard_name'] = 'height'
        ds[key].attrs['units']         = 'm'

    return ds

##########################################################################

def get_global_soil(globalpath,sitedata):

    fname = f'{globalpath}/OpenLandMap_global_soil_properties_v0.1.nc'
    ds = xr.open_dataset(fname)

    clay = ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['topsoil_clay_fraction'].values/100
    sand = ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['topsoil_sand_fraction'].values/100
    density = ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['topsoil_bulk_density'].values*10

    print(f'OpenLandMap global database:')
    print(f'    clay: {clay} [1] - https://doi.org/10.5281/zenodo.2525663')
    print(f'    sand: {sand} [1] - https://doi.org/10.5281/zenodo.2525662')
    print(f'    density {density} - [kg/m3] (https://doi.org/10.5281/zenodo.2525665')

    return clay,sand,density

def get_global_qanth(globalpath,sitedata):

    fname = f'{globalpath}/Dong2017_global_anthro_heat_v0.1.nc'
    ds = xr.open_dataset(fname)
    qanth1 = np.round(ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['Qanth'].values,1)

    print(f'Dong et al (2017) global Qf database:')
    print(f'    Qanth: {qanth1} [W/m2] - https://doi.org/10.1016/j.atmosenv.2016.11.040')

    fname = f'{globalpath}/Dong2021_global_anthro_heat_v0.1.nc'
    ds = xr.open_dataset(fname)

    qanth2 = np.round(ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['Qanth'].values,1)

    print(f'Varquez et al (2021) global Qf database:')
    print(f'    Qanth: {qanth2} [W/m2] - https://doi.org/10.1038/s41597-021-00850-w')

    return qanth1,qanth2

def get_climate(globalpath,sitedata):

    '''Legend linking the numeric values in the maps to the Köppen-Geiger classes.'''

    fname = f'{globalpath}/Beck_Koppen_climate_v1.nc'
    ds = xr.open_dataset(fname)
    climate_val = float(ds.sel(y=sitedata['latitude'],x=sitedata['longitude'], method='nearest')['climate'].values)

    if 0.5 < climate_val < 1.5:
        clim,desc  = 'Af','Tropical, rainforest'
    elif 1.99 < climate_val < 2.01:
        clim,desc = 'Am','Tropical, monsoon'
    elif 2.99 < climate_val < 3.01:
        clim,desc = 'Aw','Tropical, savannah'
    elif 3.99 < climate_val < 4.01:
        clim,desc = 'BWh','Arid, desert, hot'
    elif 4.99 < climate_val < 5.01:
        clim,desc = 'BWk','Arid, desert, cold'
    elif 5.99 < climate_val < 6.01:
        clim,desc = 'BSh','Arid, steppe, hot'
    elif 6.99 < climate_val < 7.01:
        clim,desc = 'BSk','Arid, steppe, cold'
    elif 7.99 < climate_val < 8.01:
        clim,desc = 'Csa','Temperate, dry summer, hot summer'
    elif 8.99 < climate_val < 9.01:
        clim,desc = 'Csb','Temperate, dry summer, warm summer'
    elif 9.99 < climate_val < 10.01:
        clim,desc = 'Csb','Temperate, dry winter, hot summer'
    elif 10.99 < climate_val < 11.01:
        clim,desc = 'Cwa','Temperate, dry winter, hot summer'
    elif 11.99 < climate_val < 12.01:
        clim,desc = 'Cwb','Temperate, dry winter, warm summer'
    elif 12.99 < climate_val < 13.01:
        clim,desc = 'Cwc','Temperate, dry winter, cold summer'
    elif 13.99 < climate_val < 14.01:
        clim,desc = 'Cfa','Temperate, no dry season, hot summer'
    elif 14.99 < climate_val < 15.01:
        clim,desc = 'Cfb','Temperate, no dry season, warm summer'
    elif 15.99 < climate_val < 16.01:
        clim,desc = 'Cfc' ,'Temperate, no dry season, cold summer'
    elif 16.99 < climate_val < 17.01:
        clim,desc = 'Dsa','Cold, dry summer, hot summer'
    elif 17.99 < climate_val < 18.01:
        clim,desc = 'Dsb','Cold, dry summer, warm summer'
    elif 18.99 < climate_val < 19.01:
        clim,desc = 'Dsc','Cold, dry summer, cold summer'
    elif 19.99 < climate_val < 20.01:
        clim,desc = 'Dsd','Cold, dry summer, very cold winter'
    elif 20.99 < climate_val < 21.01:
        clim,desc = 'Dwa','Cold, dry winter, hot summer'
    elif 21.99 < climate_val < 22.01:
        clim,desc = 'Dwb','Cold, dry winter, warm summer'
    elif 22.99 < climate_val < 23.01:
        clim,desc = 'Dwc','Cold, dry winter, cold summer'
    elif 23.99 < climate_val < 24.01:
        clim,desc = 'Dwd','Cold, dry winter, very cold winter'
    elif 24.99 < climate_val < 25.01:
        clim,desc = 'Dfa','Cold, no dry season, hot summer'
    elif 25.99 < climate_val < 26.01:
        clim,desc = 'Dfb','Cold, no dry season, warm summer'
    elif 26.99 < climate_val < 27.01:
        clim,desc = 'Dfc','Cold, no dry season, cold summer'
    elif 27.99 < climate_val < 28.01:
        clim,desc = 'Dfd','Cold, no dry season, very cold winter'
    elif 28.99 < climate_val < 29.01:
        clim,desc = 'ET','Polar, tundra'
    elif 29.99 < climate_val < 30.01:
        clim,desc = 'ET','Polar, frost'
    else:
        clim,desc = None,None

    print(f'Köppen climate; {clim}, {desc}')

    return clim,desc

def check_area_fractions(sitedata,bldhgt_max=None):
    '''sanity check on values provided in sitedata csv

    Note: relation for lambda_f required for roughness is based on 
     - Masson et al. 2020: htpps://doi.org/10.1016/j.uclim.2019.100536   (eq. 1)
     - Porson et al. 2010: https://doi.org/10.1002/qj.668                (eq. 7 & 8)'''

    print(sitedata,'\n')

    # check site data add to 1:
    area_fractions = sitedata['impervious_area_fraction'] + sitedata['tree_area_fraction'] + sitedata['grass_area_fraction'] + sitedata['bare_soil_area_fraction'] + sitedata['water_area_fraction']
    if round(area_fractions,3) != 1.:
        print(f'WARNING: all area fractions sum to {area_fractions}, should be 1.')

    # check impervious subfractions add to total impervious fraction
    impervious_fractions = sitedata['roof_area_fraction'] + sitedata['road_area_fraction'] + sitedata['other_paved_area_fraction']
    if round(impervious_fractions,3) != sitedata['impervious_area_fraction']:
        print(f'WARNING: impervious area fractions sum to {impervious_fractions}, should be {sitedata["impervious_area_fraction"]}')

    implied_lambda_w = 2*(1.-sitedata['roof_area_fraction'])*sitedata['canyon_height_width_ratio']
    print(f'implied lambda_w {implied_lambda_w:.2f} vs given {sitedata["wall_to_plan_area_ratio"]}')

    bldhgt_ave = sitedata['building_mean_height']
    sigma_h    = sitedata['building_height_standard_deviation']
    lambda_p   = sitedata['roof_area_fraction']
    lambda_f   = sitedata['wall_to_plan_area_ratio']/np.pi # eq 1 from Masson et al 2020, and eq 7 & 8 from Porson et al 2010.
    lambda_pv  = sitedata['tree_area_fraction']

    # roughness from morphology
    print('')
    d0,z0 = calc_site_roughness(bldhgt_ave,sigma_h,lambda_p,lambda_pv,lambda_f,bldhgt_max=bldhgt_max,mode=0)
    print(f'Mac: roughness length from morphology: {z0:.2f} vs given {sitedata["roughness_length_momentum"]}')
    print(f'Mac: zero-plane displacement from morphology: {d0:.2f} vs given {sitedata["displacement_height"]}')
    print('Mac: https://doi.org/10.1016/S1352-2310(97)00403-2')
    print('')
    d0,z0 = calc_site_roughness(bldhgt_ave,sigma_h,lambda_p,lambda_pv,lambda_f,bldhgt_max=bldhgt_max,mode=1)
    print(f'Kent: roughness length from morphology: {z0:.2f} vs given {sitedata["roughness_length_momentum"]}')
    print(f'Kent: zero-plane displacement from morphology: {d0:.2f} vs given {sitedata["displacement_height"]}')
    print('')
    d0,z0 = calc_site_roughness(bldhgt_ave,sigma_h,lambda_p,lambda_pv,lambda_f,bldhgt_max=bldhgt_max,mode=2)
    print(f'Mho: roughness length from morphology: {z0:.2f} vs given {sitedata["roughness_length_momentum"]}')
    print(f'Mho: zero-plane displacement from morphology: {d0:.2f} vs given {sitedata["displacement_height"]}')
    print('')
    d0,z0 = calc_site_roughness(bldhgt_ave,sigma_h,lambda_p,lambda_pv,lambda_f,bldhgt_max=bldhgt_max,mode=3)
    print(f'Kanda: roughness length from morphology: {z0:.2f} vs given {sitedata["roughness_length_momentum"]}')
    print(f'Kanda: zero-plane displacement from morphology: {d0:.2f} vs given {sitedata["displacement_height"]}')
    print('Kanda: https://doi.org/10.1007/s10546-013-9818-x')
    print('')

    return

############################################################################

def add_era_vegtype(ds,fpath_out):
    ''' add info to dataset of tile vegetation types and roughness
    from Table 8.3 (p121) in Part IV of ECMWF https://www.ecmwf.int/en/elibrary/16648-ifs-documentation-cy41r2-part-iv-physical-processes 
    '''
    fsr = f"era5 roughness: {ds['fsr'].values:.3f}"

    frac = round(ds['cvl'].values * 100,2)
    if 0.5 < ds['tvl'].values < 1.5:
        low = f'low veg: {frac}% crops, mixed farming with roughness 0.25m'
    elif 1.5 < ds['tvl'].values < 2.5:
        low = f'low veg: {frac}% short grass with roughness 0.2m'
    elif 6.5 < ds['tvl'].values < 7.5:
        low = f'low veg: {frac}% tall grass with roughness 0.47m'
    elif 8.5 < ds['tvl'].values < 9.5:
        low = f'low veg: {frac}% tundra with roughness 0.034m'
    elif 9.5 < ds['tvl'].values < 10.5:
        low = f'low veg: {frac}% irrigated crops roughness 0.5m'
    elif 10.5 < ds['tvl'].values < 11.5:
        low = f'low veg: {frac}% semidesert with roughness 0.17m'
    elif 12.5 < ds['tvl'].values < 13.5:
        low = f'low veg: {frac}% bogs and marshes with roughness 0.83m'
    elif 15.5 < ds['tvl'].values < 16.5:
        low = f'low veg: {frac}% evergreen shrubs with roughness 0.10m'
    elif 16.5 < ds['tvl'].values < 17.5:
        low = f'low veg: {frac}% deciduous shrubs with roughness 0.25m'
    else:
        low = 'no low veg recorded'
    lowfrac = frac

    frac = round(ds['cvh'].values * 100,2)
    if 2.5 < ds['tvh'].values < 3.5:
        high = f'high veg: {frac}% evergreen needleleaf trees with roughness 2.0m'
    elif 3.5 < ds['tvh'].values < 4.5:
        high = f'high veg: {frac}% deciduous needleleaf trees with roughness 2.0m'
    elif 4.5 < ds['tvh'].values < 5.5:
        high = f'high veg: {frac}% deciduous broadleaf trees with roughness 2.0m'
    elif 5.5 < ds['tvh'].values < 6.5:
        high = f'high veg: {frac}% evergreen broadleaf trees with roughness 2.0m'
    elif 17.5 < ds['tvh'].values < 18.5:
        high = f'high veg: {frac}% mixed forest/woodland roughness 2.0m'
    elif 18.5 < ds['tvh'].values < 19.5:
        high = f'high veg: {frac}% interrupted forest with roughness 1.1m'
    else:
        high = 'no high vegetation recorded'
    highfrac = frac

    # print('era5 vegetation types on tile')
    # print(high)
    # print(low)
    # print(fsr)
    # ds.close()

    # f = netCDF4.Dataset(fpath_out, 'r+')
    # f.variables['tvh'].desc = high
    # f.variables['tvl'].desc = low
    # f.variables['fsr'].desc = fsr
    # f.close()

    return low,high,lowfrac,highfrac

def create_markdown_observations(ds,siteattrs):
    '''Creates markdown websites from netcdf attributes

    inputs
    ------
    ds       : [xr.dataset] forcing dataset
    sitattrs : [dictionary] site attributes

    '''

    sitename = siteattrs['sitename']
    sitepath = siteattrs['sitepath']

    attrs = pd.Series(ds.attrs)
    attrs.name = 'observation_attributes'

    fpath = f'{sitepath}/{sitename}_sitedata_{siteattrs["sitedata_suffix"]}.csv'
    sitedata_full = pd.read_csv(fpath, index_col=1, delimiter=',')

    sdata = sitedata_full.copy()
    sdata['parameter'] = sdata.index
    sdata = sdata.set_index('id')[['parameter','value','units','source','doi']]

    for idx in sdata.index:
        try:
            if sdata.loc[idx,'doi'][:4] == 'http':
                sdata.loc[idx,'doi'] = '[%s](%s)' %(sdata.loc[idx,'doi'],sdata.loc[idx,'doi'])
        except Exception:
            print(f'WARNING: error processing sitedata links for {sdata.loc[idx,"parameter"]}.')
            continue

    site_photo = f'./images/{ds.sitename}_site_photo.jpg'
    site_region= f'./images/{ds.sitename}_region_map.jpg'
    site_sat   = f'./images/{ds.sitename}_site_sat.jpg'
    site_map   = f'./images/{ds.sitename}_site_map.jpg'
    all_obs_qc = f'./obs_plots/all_obs_qc.{img_fmt}'
    photo_source = siteattrs['photo_source']

    print('writing site markdown file to index.md')

    with open(f'{sitepath}/index.md', 'w') as f:
        f.write(f'''
# {ds.long_sitename} ({ds.sitename})

### Jump to section:

 - [Site forcing metadata](#site-forcing-metadata)
 - [Site images](#site-images)
 - [Site characteristics](#site-characteristics)
 - [Site forcing](#site-forcing)
 - [Quality control and gap filling procedure](#quality-control-qc-and-gap-filling-procedure)
 - [Bias correction diurnal comparison](#bias-correction-diurnal-comparison)

## Observations (before additional gap filling)

[![{all_obs_qc}]({all_obs_qc})]({all_obs_qc})

## Site forcing metadata

{attrs.to_markdown()}

## Site images

|                                             |                                             |    
|:-------------------------------------------:|:-------------------------------------------:|
| [![Region]({site_region})]({site_region})  <sub>Regional map. © OpenStreetMap</sub>    | [![site_map]({site_map})]({site_map}) <sub>Site map with 500 m radius. © OpenStreetMap</sub>    |
| [![site_photo]({site_photo})]({site_photo}) <sub>Site photo. © {photo_source}</sub>    | [![site_sat]({site_sat})]({site_sat}) <sub>Site aerial photo with 500 m radius. © OpenStreetMap, Microsoft</sub>    |

<sub>Maps developed from:
    Hrisko, J. (2020). [Geographic Visualizations in Python with Cartopy](https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy). Maker Portal.</sub> 

## Site characteristics

{sdata.to_markdown()}

## Site forcing

''')
        for flux in ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Snowf','Wind_N','Wind_E']:
            fname = f'./obs_plots/{flux}_gapfilled_forcing.{img_fmt}'
            f.write(f'### {flux} forcing\n')
            f.write('\n')
            f.write(f'[![{flux}]({fname})]({fname})\n')
            f.write('\n')
        f.write('''
## Quality control (qc) and gap filling procedure

**QC process on observations**

 1. **Out-of-range**: removal of unphysical values (e.g.  negative shortwave radiation) using the [ALMA expected range](https://www.lmd.jussieu.fr/~polcher/ALMA/qc_values_3.html) protocol.
 2. **Night**: nocturnal shortwave radiation set to zero, excluding civil twilight (when the sun is 6° below the horizon).
 3. **Constant**: four or more timesteps with the same value (excluding zero values for shortwave radiation, rainfall and snowfall) are removed as suspicious.
 4. **Outlier**: remove values outside ±4 standard deviations for each hour in a rolling 30-day window (to account for diurnal and seasonal variations). Repeat with a larger tolerance (± 5 standard deviations) until no outliers remain. The outlier test is not applied to precipitation.
 5. **Visual**: remaining suspect readings are removed manually via visual inspection.

**Gap-filling process**

 - contemporaneous and nearby flux tower or weather observing sites (where available and provided by observing groups)
 - small gaps (≤ 2 hours) are linearly interpolated from the adjoining observations
 - larger gaps (and a 10-year spin-up period) are filled with bias corrected ERA5 data (see below)
 - snowfall from ERA5, with water equivalent removed from rainfall to retain mass balance
 
''')

        filepaths = sorted(glob.glob(f'{sitepath}/obs_plots/*_qc_diurnal.{img_fmt}'))
        filenames = [os.path.basename(path) for path in filepaths]

        f.write('\n')
        for filename in filenames:

            if filename[:6]=='Wind_E':
                flux = 'Wind_E'
            elif filename[:6]=='Wind_N':
                flux = 'Wind_N'
            else:
                flux = filename.split('_')[0]
            fname = f'./obs_plots/{filename}'

            f.write(f'### {flux} diurnal qc\n')
            f.write('\n')
            f.write(f'[![{fname}]({fname})]({fname})\n')
            f.write('\n')


        f.write('\n')
        f.write(f'''## Bias correction diurnal comparison\n

Four methods drawing on ERA5 reanalysis are compared relative to the quality-controlled flux tower data. The methods are:

 1.  **ERA5**: the nearest land based 0.25° resolution ERA5 grid (i.e. without bias correction)
 2.  **WFDE5**: the nearest WFDE5 grid (which use 0.5° gridded monthly observations for bias correction)
 3.  **UP**: the Urban-PLUMBER methods used in this collection (using site observations for bias correction)
 4.  **LN**: linear methods based on FLUXNET2015 (using site observations for bias correction)

 **ERA5 bias correction**

The UP methods are as follows:
 
 - for downwelling longwave, temperature, humidity and pressure: calculate the mean bias between ERA5 and flux tower data in a 30-day rolling window for every hour and each day of the year, apply that bias correction to all ERA5 data. For periods not covered by observations, linearly interpolate between known biases for each hour separately.
 - for precipitation: calculate total precipitation in a 10-year period and calculate the ratio between ERA5 data and the nearest GHCN-D station and apply that correction factor to ERA5 data.
 - for wind: apply wind log profile correction from ERA5 10m wind to tower measurement height using site roughness and displacement, with original grid roughness iteratively revised so that mean corrected wind speed matches observation.
 - for downwelling shortwave: use ERA5 data without correction

Mean absolute error (MAE) is shown in the legend.
''')
        for flux in ['Tair','Qair','PSurf','LWdown','SWdown','Wind','Rainf']:
            fname = f'./era_correction/{sitename}_{flux}_all_diurnal.{img_fmt}'

            f.write(f'### {flux} diurnal bias correction\n')
            f.write('\n')
            f.write(f'[![{fname}]({fname})]({fname})\n')
            
        f.write('''\n
### Jump to section:

 - [Site forcing metadata](#site-forcing-metadata)
 - [Site images](#site-images)
 - [Site characteristics](#site-characteristics)
 - [Site forcing](#site-forcing)
 - [Quality control and gap filling procedure](#quality-control-qc-and-gap-filling-procedure)
 - [Bias correction diurnal comparison](#bias-correction-diurnal-comparison)

''')

    return

##########################################################################
# UNUSED PLOTS
##########################################################################

def plot_filled():

    # orig_ds = xr.open_dataset('AU-Preston_metforcing_v1.nc')

    fluxes = ['SWdown', 'LWdown', 'Tair', 'Qair', 'PSurf', 'Rainf', 'Snowf', 'Wind_N','Wind_E']

    fluxes = ['SWdown']

    sdate = forcing_ds.time_analysis_start
    edate = forcing_ds.time_coverage_end

    for flux in fluxes:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10,5))
        # era_ds[flux].sel(time=slice(sdate,edate)).plot(ax=ax,color='gold',marker='x',ms=2,label='orig era')
        orig_ds[flux].sel(time=slice(sdate,edate)).plot(ax=ax,color='r',marker='x',ms=2,label='original')
        corr_ds[flux].sel(time=slice(sdate,edate)).plot(ax=ax,color='b',marker='x',ms=2,label='new gap-filled era')
        forcing_ds[flux].where(forcing_ds['%s_qc' %flux] == 0 ).plot(ax=ax,color='k',marker='x',ms=2,label='observed')
        # forcing_ds[flux].where(forcing_ds['%s_qc' %flux] == 1 ).plot(ax=ax,color='g',marker='x',ms=2,label='gap-filled obs')
        ax.legend(loc = 'upper left')
        ax.set_title(forcing_ds[flux].long_name)

        plt.show() 

    return

def plot_lw():

    forcing = forcing_ds.squeeze().to_dataframe()
    e_vap = forcing['Qair']*forcing['PSurf']/0.6219665
    # Ukkola 2017
    LW = 2.648*forcing['Tair'] + 0.0346*e_vap - 474

    key = 'LWdown'
    plt.close('all')
    fig,ax = plt.subplots(figsize=(12,4.5))
    pdate,edate = forcing_ds.time_analysis_start, forcing_ds.time_coverage_end
    corr_ds.squeeze().to_dataframe().loc[pdate:edate,key].plot(ax=ax,color='gold',label='corrected era')
    LW.loc[pdate:edate].plot(
        ax=ax,lw=1.5,marker='x',ms=2,color='red',label='emperical function on Tair, Qair, PSurf')
    forcing.loc[pdate:edate,key].plot(ax=ax, color='g')
    forcing.loc[pdate:edate,key].where(forcing['%s_qc' %key]==0).plot(ax=ax, color='k')
    ax.legend(fontsize=8)

    plt.show()

    return

