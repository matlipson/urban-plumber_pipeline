'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Quality control observations"
__version__ = "2021-09-08"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

import numpy as np
import xarray as xr
import pandas as pd
import ephem
import matplotlib.pyplot as plt
import os
import requests
import pipeline_functions

pd.plotting.register_matplotlib_converters()

img_fmt = 'png'

################################################################################

def main(ds, sitedata, siteattrs, sitepath, plotdetail=False):

    sitename = siteattrs['sitename']

    window = 30
    sigma = 4 # initial sigma at 4 standard deviations
    offset_from_utc = ds.local_utc_offset_hours

    # get all variables in dataset
    all_variables = list(ds.keys())
    # get qc fluxes
    qc_list = [x for x in all_variables if '_qc' in x]
    # remove qc variables from list
    all_fluxes = [x for x in all_variables if '_qc' not in x]
    # remove Rainf and Snowf from sigma filtering set
    sigma_fluxes = [n for n in all_fluxes if n not in ['Rainf','Snowf']]

    # clean raw obs (including filled by nearby obs)
    to_clean = ds.to_dataframe()[all_fluxes]

    print('removing out-of-range values')
    cleaned1 = clean_out_of_range(to_clean,siteattrs)

    print('removing night periods for SW')
    # segmentation fault from ephem for these sites, use different method
    if sitename in ['FI-Kumpula','NL-Amsterdam','AU-SurreyHills','CA-Sunset']:
        cleaned2 = clean_night2(cleaned1,sitedata,siteattrs,offset_from_utc)
    else:
        cleaned2 = clean_night(cleaned1,sitedata,siteattrs,offset_from_utc)

    print('removing values constant for 4 or more timesteps')
    cleaned3 = clean_constant(cleaned2,siteattrs)

    print(f'removing outliers > x sigma for each hour in {window} day window')

    # pass cleaned fluxes through outlier check based on standard deviation x sigma for each hour, until clean
    loop = 0
    sigma_clean = cleaned3.copy()[sigma_fluxes]
    while True:

        # create sigma threshold for each flux being cleaned
        sigma = 4 if loop == 0 else 5
        sigma_vals = pd.Series(data=sigma,index=sigma_clean.columns.tolist())
        if sitename in ['NL-Amsterdam']:
            print('lowering sigma threshold for amsterdam Qh, Qle')
            sigma_vals[['Qh','Qle']] = sigma_vals[['Qh','Qle']] - 1

        print(f'pass: {loop}')
        sigma_flagged, sigma_clean = calc_sigma_data(sigma_clean,sigma_vals=sigma_vals,sitepath=sitepath,window=window,plotdetail=False)
        print('flagged results')
        print(sigma_flagged.describe())

        loop +=1

        # finish when no additional outliers are found
        if sigma_flagged.count().sum() == 0:
            # calculate dirty from those removed in sigma cleaning
            dirty = cleaned3[sigma_fluxes].where(sigma_clean.isna())
            print('\nsigma outliers: \n%s\n' %dirty.count())
            print(f'saving sigma stats to {sitename}_outliers_sigma.csv\n')
            dirty.to_csv(f'{sitepath}/processing/{sitename}_outliers_sigma.csv')
            break
    # # remove outlier data for select fluxes
    clean = cleaned3.copy()
    clean[sigma_fluxes] = sigma_clean[sigma_fluxes]

    ##########################################
    # manual corrections from visual inspection (not picked up automatically)

    if sitename == 'CA-Sunset':
        # remove unrealistically low LW measurements
        clean.loc['2014-04-30 18:00':'2014-04-30 22:00',['LWdown','LWup']] = np.nan
        clean.loc['2014-05-03 17:00':'2014-05-03 19:00',['LWdown','LWup']] = np.nan
        clean.loc['2014-05-05 18:00':'2014-05-05 22:00',['LWdown','LWup']] = np.nan
        # remove unrealistically low SW measurements
        clean.loc['2014-11-17 19:30':'2014-11-17 20:00',['SWdown','SWup']] = np.nan
        # remove negative Qair values
        clean['Qair'] = clean['Qair'].where(clean['Qair']>0)
        # remove zero valued wind before 2012-10-28
        clean.loc[:'2012-10-28 19:30:00',['Wind_N','Wind_E']] = np.nan

    if sitename == 'US-Baltimore':
        # remove spurious Qair outlier
        clean.loc['2003-09-29 06:00:00','Qair'] = np.nan
        # remove spurious nightime SWdown in September 2003
        clean.loc['2003-09-06':'2003-09-08 08:00','SWdown'] = np.nan
        clean.loc['2003-09-05 22:00:00','SWdown'] = np.nan
        # remove spike in Tair
        clean.loc['2002-07-10 10:00:00','Tair'] = np.nan
        clean.loc['2002-07-10 20:00:00','Tair'] = np.nan
        clean.loc['2003-10-08 17:00:00','Tair'] = np.nan

    if sitename == 'PL-Lipowa':
        # remove spurious wind
        clean.loc['2010-01-09 18:00':'2010-01-21 06:00',['Wind_N','Wind_E']] = np.nan

    if sitename == 'PL-Narutowicza':
        # remove spurious wind (other periods match very well with ERA5, these diverge considerably)
        clean.loc['2009-01-11',['Wind_N','Wind_E']] = np.nan
        clean.loc['2009-06-29':'2009-07-04',['Wind_N','Wind_E']] = np.nan
        clean.loc['2010-01-09 18:00':'2010-01-22',['Wind_N','Wind_E']] = np.nan
        clean.loc['2010-12-12':'2010-12-17',['Wind_N','Wind_E']] = np.nan

    if sitename == 'MX-Escandon':
        # remove spurious pressure:
        clean.loc['2012-06-05 02:00:00',['PSurf','Qair','Qtau']] = np.nan
        clean.loc['2012-06-24 02:30:00',['PSurf','Qair','Qtau']] = np.nan
        clean.loc['2012-06-24 02:00:00',['PSurf','Qair','Qtau']] = np.nan

    if sitename == 'JP-Yoyogi':
        # remove unrealistically low Qair values
        clean['Qair'] = np.where(clean['Qair']<0.0002, np.nan, clean['Qair'])

    ##########################################

    # calculate dirty from all missing clean
    dirty = to_clean.where(clean.isna())
    dirty.to_csv(f'{sitepath}/processing/{sitename}_dirty.csv')

    # collect observed only to ascertain "missing"
    obs_only,fill_only = pd.DataFrame(),pd.DataFrame()
    for flux in all_fluxes:
        obs_only[flux] = to_clean[flux].where(ds[f'{flux}_qc'].to_series() == 0)
        fill_only[flux] = to_clean[flux].where(ds[f'{flux}_qc'].to_series() == 1)

    # gather quality stats
    stats = pd.DataFrame({  
                    'missing'    : 1. - to_clean.count()/len(to_clean),
                    'qc_flagged' : dirty.count()/len(to_clean),
                    'filled' : fill_only.count()/len(to_clean),
                    'available'  : clean.count()/len(to_clean),
                    })
    print(stats)
    print(f'saving stats to {sitename}_cleanstats.csv\n')
    stats.to_csv(f'{sitepath}/processing/{sitename}_cleanstats.csv',float_format='%.4f')

    # place qc flags back in to cleaned df if still present, or fill with qc=3 if missing
    for key in qc_list:
        orig_qc = ds.to_dataframe()[qc_list][key]
        clean[key] = np.where(clean[key[:-3]].isna(), 3, orig_qc)

    if plotdetail:

        # convert to dataframe in local time
        local_clean = convert_utc_to_local(clean.copy(), ds.local_utc_offset_hours)
        local_dirty = convert_utc_to_local(dirty.copy(), ds.local_utc_offset_hours)

        # plot strings dictionary
        plt_str = {}
        plt_str['time_start'] = 'start date: %s'  %(convert_utc_to_local_str(ds.time_coverage_start, ds.local_utc_offset_hours))
        plt_str['time_end']   = 'end date: %s'    %(convert_utc_to_local_str(ds.time_coverage_end, ds.local_utc_offset_hours))
        plt_str['days']       = 'period: %s days' %((pd.to_datetime(ds.time_coverage_end) - pd.to_datetime(ds.time_coverage_start)).days + 1)
        plt_str['interval']   = 'interval: %s s'  %(ds.timestep_interval_seconds)
        plt_str['timesteps']  = 'timesteps: %s'   %(len(ds.time))

        for flux in all_fluxes:

            # plot strings dictionary for flux
            plt_str['missing']   = 'missing: %.2f %%' %(100*stats.loc[flux,'missing'])
            plt_str['qcflags']   = 'QC flag: %.2f %%' %(100*stats.loc[flux,'qc_flagged'])
            plt_str['filled']   = 'filled: %.2f %%' %(100*stats.loc[flux,'filled'])
            plt_str['available'] = 'available: %.2f %%' %(100*stats.loc[flux,'available'])

            # plot_qc_timeseries(ds,local_clean,local_dirty,plt_str,flux,sitename,sitepath,saveplot=True)

            plot_qc_diurnal(ds,local_clean,local_dirty,plt_str,flux,sitename,sitepath,saveplot=True)

        plot_all_obs(clean,dirty,stats,sitename,sitepath,all_fluxes,qc_list,saveplot=True)
        plt.close('all')

    clean_ds = clean.to_xarray()

    return clean_ds

################################################################################

def clean_out_of_range(df,siteattrs):

    # alma expected range of values, per:
    #   https://www.lmd.jussieu.fr/~polcher/ALMA/qc_values_3.html#A1
    alma_ranges = pd.DataFrame({
    'SWnet'        : (0,1200),
    'LWnet'        : (-500,510),
    'Qle'          : (-700,700),
    'Qh'           : (-600,600),
    'SWup'         : (0,1360),   # new
    'LWup'         : (0,1000),   # new
    'Qg'           : (-500,500),
    'Qtau'         : (-100,100),
    'Snowf'        : (0,0.0085),
    'Rainf'        : (0,0.02),
    'Evap'         : (-0.0003,0.0003),
    'Qs'           : (0,5),
    'Qsb'          : (0,5),
    'Qsm'          : (0,0.005),
    'Qfz'          : (0,0.005),
    'DelSoilMoist' : (-2000,2000),
    'DelSWE'       : (-2000,2000),
    'DelIntercept' : (-100,100),
    'SnowT'        : (213,280),
    'VegT'         : (213,333),
    'BaresoilT'    : (213,343),
    'AvgSurfT'     : (213,333),
    'RadT'         : (213,353),
    'Albedo'       : (0,1),
    'SWE'          : (0,2000),
    'SurfStor'     : (0,2000),
    'SnowFrac'     : (0,1),
    'SAlbedo'      : (0,1),
    'CAlbedo'      : (0,1),
    'UAlbedo'      : (0,1),
    'SoilMoist'    : (0,2000),
    'SoilTemp'     : (213,343), # increase max + 10 for Phoenix
    'TVeg'         : (-0.0003,0.0003),
    'ESoil'        : (-0.0003,0.0003),
    'RootMoist'    : (0,2000),
    'SoilWet'      : (-0.2,1.2),
    'ACond'        : (0,1),
    'SWdown'       : (0,1360),
    'LWdown'       : (0,750),
    'Tair'         : (213,333),
    'Tair2m'       : (213,333), # new
    'Qair'         : (0,0.03),
    'PSurf'        : (5000,110000),
    'Wind'         : (-75,75),
    'Wind_N'       : (-75,75), # new
    'Wind_E'       : (-75,75), # new
    },index=('min','max')) 

    # remove ranges
    clean = df.where ( (df >= alma_ranges.loc['min',:]) & (df <= alma_ranges.loc['max',:]) )

    # remove RH above 101 (requires Qair, Tair and PSurf to be valid)
    RH = pipeline_functions.convert_qair_to_rh(clean.Qair, clean.Tair, clean.PSurf)
    clean['Qair'] = np.where(RH>101,np.nan,clean['Qair'])
 
    dirty = df.where(clean.isna())
    print('out of range: \n%s\n' %dirty.count())

    print(f'saving out-of-range stats to {siteattrs["sitename"]}_outliers_range.csv\n')
    dirty.to_csv(f'{siteattrs["sitepath"]}/processing/{siteattrs["sitename"]}_outliers_range.csv')

    return clean

def clean_night(df,sitedata,siteattrs,offset_from_utc):
    '''
    flags as dirty night periods for shortwave
    calculate night using PyEphem sunset/sunrise + civil twilight (-6°)
    '''

    local_df = convert_utc_to_local(df.copy(), offset_from_utc)

    # Make ephem observer
    site = ephem.Observer()
    site.lon = str(sitedata['longitude'])
    site.lat = str(sitedata['latitude'])
    site.elev = sitedata['ground_height']

    # relocate the horizon to get twilight times
    site.horizon = '-6' #-6=civil twilight, -12=nautical, -18=astronomical

    # timestep adjustment
    ts = df.index[1] - df.index[0]

    new_group_list = []

    local_sw = local_df[['SWdown','SWup']]

    if local_sw.count().sum()==0:
        print('no valid SWdown or SWup values')
        # return

    for date, group in local_df.loc[:,['SWdown','SWup']].groupby(lambda x: x.date):

        # utc midday
        utc_midday = pd.Timestamp(date) + pd.Timedelta(hours=12) - pd.Timedelta(hours=offset_from_utc)
        site.date = str(pd.Timestamp(utc_midday))

        try:
            utc_sunrise = site.previous_rising(ephem.Sun())
            utc_solarnoon = site.next_transit(ephem.Sun(), start=utc_sunrise)
            utc_sunset = site.next_setting(ephem.Sun())

            local_sunrise = ephem.Date(utc_sunrise + offset_from_utc/24.)
            local_solarnoon = ephem.Date(utc_solarnoon + offset_from_utc/24.)
            local_sunset = ephem.Date(utc_sunset + offset_from_utc/24.)

            # include timestep adjustment for observation period
            local_sunrise_ts = pd.Timestamp( local_sunrise.datetime() + ts ).strftime('%H:%M')
            local_solarnoon_ts =  pd.Timestamp( local_solarnoon.datetime() + ts ).strftime('%H:%M')
            local_sunset_ts = pd.Timestamp( local_sunset.datetime() + ts ).strftime('%H:%M')

            # print('local morning twilight:', local_sunrise_ts )
            # print('local evening twilight:', local_sunset_ts )

            # print('local solar noon:', local_solarnoon_ts )

            day = group.between_time(start_time=local_sunrise_ts,end_time=local_sunset_ts)

        except Exception as e:
            print('WARNING: Ephem calculation failed')
            print(e)
            pass

        # day = group

        night = group[~group.index.isin(day.index)]

        # remove any values which are not zero during night
        night = night.where(night==0)

        # remove all night SWup values (not analysed)
        night['SWup'] = np.nan

        new_group = pd.concat([day,night])
        new_group_list.append(new_group)

    clean_local_sw = pd.concat(new_group_list).sort_index()

    local_df[['SWdown','SWup']] = clean_local_sw

    clean = convert_local_to_utc(local_df, offset_from_utc)

    dirty = df.where(clean.isna())
    print('night: \n%s\n' %dirty.count())

    print(f'saving night stats to {siteattrs["sitename"]}_outliers_night.csv\n')
    dirty.to_csv(f'{siteattrs["sitepath"]}/processing/{siteattrs["sitename"]}_outliers_night.csv')

    return clean


def clean_night2(df,sitedata,siteattrs,offset_from_utc):
    '''
    At some locations EPHEM module throws segmentation fault (!)
    therefore use alternative web-based solution (is very slow)
    calculate night using https://sunrise-sunset.org/api
    '''


    # timestep adjustment
    ts = df.index[1] - df.index[0]

    new_group_list = []

    sw = df[['SWdown','SWup']]

    if sw.count().sum()==0:
        print('no valid SWdown or SWup values')
        # return

    for date, group in sw.groupby(lambda x: x.date):

        print(date)

        date_str = date.strftime('%Y-%m-%d')

        r = requests.get(url=f'http://api.sunrise-sunset.org/json?lat={sitedata["latitude"]}&lng={sitedata["longitude"]}&date={date_str}&formatted=0')
        data = r.json()['results']

        # include timestep adjustment for observation period
        start_ts = (pd.Timestamp( data['civil_twilight_begin'] ) + ts).strftime('%H:%M')
        end_ts = (pd.Timestamp( data['civil_twilight_end'] ) + ts).strftime('%H:%M')

        day = group.between_time(start_time=start_ts,end_time=end_ts)
        night = group[~group.index.isin(day.index)]

        # remove any values which are not zero during night
        night = night.where(night==0)

        # remove all night SWup values (not analysed)
        night['SWup'] = np.nan

        new_group = pd.concat([day,night])
        new_group_list.append(new_group)

    clean_sw = pd.concat(new_group_list).sort_index()

    clean = df.copy()
    clean[['SWdown','SWup']] = clean_sw

    dirty = df.where(clean.isna())
    print('night: \n%s\n' %dirty.count())

    print(f'saving night stats to {siteattrs["sitename"]}_outliers_night.csv\n')
    dirty.to_csv(f'{siteattrs["sitepath"]}/processing/{siteattrs["sitename"]}_outliers_night.csv')

    return clean

def clean_constant(df, siteattrs):
    '''
    flag as dirty variables that are constant for some time (suspect instrument failure)
    '''

    # some variables have expected constant zero fluxes (e.g. rain), so allow this at all sites
    zero_fluxes_ok = ['SWdown','Rainf','Snowf']

    # Some variables like SoilTemp change very slowly, and at some sites measured with fewer significant figures
    # so allow longer period of constant fluxes in some cases
    if siteattrs['sitename'] in ['UK-Swindon','PL-Lipowa','PL-Narutowicza','US-Minneapolis','US-Minneapolis1','US-Minneapolis2']:
        constant_fluxes_ok = ['SoilTemp','PSurf']
    else:
        constant_fluxes_ok = ['SoilTemp']

    # get list of fluxes except constant_fluxes_ok
    fluxes0 = [n for n in df.columns.to_list() if n not in constant_fluxes_ok]
    # get list of fluxes from fluxes0, except zero_fluxes_ok (i.e. standard set of variables)
    fluxes1 = [n for n in fluxes0 if n not in zero_fluxes_ok]

    # QC: where values repeat for 4 steps in row (standard qc)
    df1 = df[fluxes1]
    constant1 = df1.where( ( df1.eq(df1.shift(1))) & (df1.eq(df1.shift(2))) & (df1.eq(df1.shift(3))) )

    # QC: where values repeat for 4 steps in row, and excluding zero
    df2 = df[zero_fluxes_ok]
    constant2 = df2.where( (df2.eq(df2.shift(1))) & (df2.eq(df2.shift(2))) & (df2.eq(df2.shift(3))) & (df2.ne(0)) )

    # QC: where values repeat for 12 steps in a row (special cases)
    df3 = df[df.columns.intersection(constant_fluxes_ok)]
    constant3 = df3.where( ( df3.eq(df3.shift(1))) & (df3.eq(df3.shift(2))) & (df3.eq(df3.shift(3))) 
        & (df3.eq(df3.shift(4))) & (df3.eq(df3.shift(5))) & (df3.eq(df3.shift(6))) & (df3.eq(df3.shift(7))) & (df3.eq(df3.shift(8)))
        & (df3.eq(df3.shift(9))) & (df3.eq(df3.shift(10))) & (df3.eq(df3.shift(11))) )

    # bring all flagged dirty together
    dirty = pd.concat([constant1,constant2,constant3], axis=1)
    
    clean = df.where(dirty.isna())

    print('constant: \n%s\n' %dirty.count())

    print(f'saving constant stats to {siteattrs["sitename"]}_outliers_constant.csv\n')
    dirty.to_csv(f'{siteattrs["sitepath"]}/processing/{siteattrs["sitename"]}_outliers_constant.csv')

    return clean

################################################################################

def plot_all_obs(clean,dirty,stats,sitename,sitepath,all_fluxes,qc_list,saveplot=True):

    print('plotting all_obs')

    df = clean.combine_first(dirty)

    fig_hgt = len(all_fluxes)*.4

    plt.close('all')
    fig, axes = plt.subplots(
                    nrows=len(all_fluxes),
                    ncols=1,
                    sharex=True,
                    figsize=(10,fig_hgt))

    for i,ax in enumerate(axes.flatten()):

        flux = all_fluxes[i]

        # #### exclude gap-filled from obs ####
        # # plot clean
        clean[flux].where(clean[f'{flux}_qc']==0).plot(ax=ax, color='k', lw=0.5)

        # plot obs filled
        clean[flux].where(clean[f'{flux}_qc']==1).plot(ax=ax, color='tab:blue', lw=0.5)

        # plot missing
        missing_idx = df[flux][(df[flux].isna())].index
        missing = pd.Series( np.full( len(missing_idx),df[flux].min() ),index=missing_idx )
        if len(missing) > 0:
            missing.plot(ax=ax,color='darkorange',lw=0,marker='.',ms=1.5)

        # plot dirty
        dirty[flux].plot(ax=ax, color='red', lw=0.0, marker='.', ms=1.5)

        # annotations
        ax.text(-0.01,0.5,flux, 
            fontsize=10, ha='right',va='center',transform=ax.transAxes)

        if i==0:
            ax.set_title('Observations at %s' %sitename)
            ax.text(1.03,1.01,'missing' , 
                fontsize=7, color='darkorange', ha='center',va='bottom',transform=ax.transAxes)
            ax.text(1.08,1.01,'flagged' , 
                fontsize=7, color='red', ha='center',va='bottom',transform=ax.transAxes)
            ax.text(1.13,1.01,'filled' , 
                fontsize=7, color='tab:blue', ha='center',va='bottom',transform=ax.transAxes)
            ax.text(1.18,1.01,'avail.' , 
                fontsize=7, color='k', ha='center',va='bottom',transform=ax.transAxes)

        ax.text(1.03,0.5,'%.1f%%' %(100*stats.loc[flux,'missing']), 
            fontsize=7, color='darkorange', ha='center',va='center',transform=ax.transAxes)
        ax.text(1.08,0.5,'%.1f%%' %(100*stats.loc[flux,'qc_flagged']), 
            fontsize=7, color='red', ha='center',va='center',transform=ax.transAxes)
        ax.text(1.13,0.5,'%.1f%%' %(100*stats.loc[flux,'filled']), 
            fontsize=7, color='tab:blue', ha='center',va='center',transform=ax.transAxes)
        ax.text(1.18,0.5,'%.1f%%' %(100*stats.loc[flux,'available']), 
            fontsize=7, color='k', ha='center',va='center',transform=ax.transAxes)

        ax.axes.get_yaxis().set_ticks([])
        ax.tick_params(axis='x',which='minor',bottom=False)
        ax.set_xlabel(None)

    if saveplot:
        fig.savefig(f'{sitepath}/obs_plots/all_obs_qc.{img_fmt}', dpi=200,bbox_inches='tight')
    else:
        plt.show()

def plot_qc_timeseries(ds,clean,dirty,plt_str,flux,sitename,sitepath,saveplot=True):

    print('plotting %s timeseries' %flux)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    clean[flux].plot(ax=ax, color='0.5', lw=0.5, label='clean obs')
    dirty[flux].plot(ax=ax, color='r',marker='x',ms=3,lw=0,label='qc flagged')

    # annotations
    ax = plt_annotate(ax,plt_str,fs=7)

    ax.legend(loc='upper center', fontsize=7)

    ax.set_title(f"{sitename}: {ds[flux].attrs['long_name']}" )
    ax.set_ylabel(f"{ds[flux].attrs['long_name']} [{ds[flux].attrs['units']}]")

    ax.set_xlim((ds.time_coverage_start,ds.time_coverage_end))

    if saveplot==True:
        fig.savefig(f'{sitepath}/obs_plots/{flux}_obs_qc_ts.{img_fmt}', dpi=150,bbox_inches='tight')
    else:
        plt.show()

def plot_qc_diurnal(ds,local_clean,local_dirty,plt_str,flux,sitename,sitepath,saveplot=True):

    print('plotting %s qc diurnal' %flux)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5))

    clean_date = local_clean[flux].groupby(local_clean.index.date)
    clean_time = local_clean[flux].groupby(local_clean.index.time)

    for i,(key,item) in enumerate(clean_date):
        # ax.plot(item.index.time, item, color='0.75',lw=0.3,label='all clean data' if i == 0 else None)
        item.index = item.index.time
        item.plot(color='0.75',lw=0.3,label='all clean data' if i == 0 else '_nolegend_')

    if local_dirty[flux].count()>0:
        dirty_date = local_dirty[flux].groupby(local_dirty.index.date)

        for i,(key,item) in enumerate(dirty_date):
            # ax.plot(item.index.time, item, color='r',marker='x',ms=3,lw=0.3,label='qc flagged' if i == 0 else None)
            item.index = item.index.time
            item.plot(color='r',marker='x',ms=3,lw=0.3,label='qc flagged' if i == 0 else '_nolegend_')

    clean_time.mean().plot(ax=ax, color='k',lw=1.5,label='mean of clean data')
    clean_time.quantile(0.10).plot(ax=ax, color='k',lw=1,ls='dashed',label='10th & 90th percentiles')
    clean_time.quantile(0.90).plot(ax=ax, color='k',lw=1,ls='dashed',label='_nolegend_')

    # annotations
    ax = plt_annotate(ax,plt_str,fs=7)

    ax.legend(loc='upper center', fontsize=7,ncol=2)

    ax.set_title(f"{sitename}: {ds[flux].attrs['long_name']} diurnal values" )
    ax.set_ylabel(f"{ds[flux].attrs['long_name']} [{ds[flux].attrs['units']}]")

    ax.set_xlim(('00:00','23:30:00'))
    ax.set_xticks([str(x).zfill(2)+':00' for x in range(0,24,3)] )

    if saveplot==True:
        fig.savefig(f'{sitepath}/obs_plots/{flux}_obs_qc_diurnal.{img_fmt}', dpi=150,bbox_inches='tight')
    else:
        plt.show()

def plt_annotate(ax,plt_str,fs=7):

    ax.text(0.02,0.96, plt_str['time_start'], fontsize=fs, va='center',ha='left', transform=ax.transAxes)
    ax.text(0.02,0.92, plt_str['time_end'],   fontsize=fs, va='center',ha='left', transform=ax.transAxes)
    ax.text(0.02,0.88, plt_str['days'],       fontsize=fs, va='center',ha='left', transform=ax.transAxes)
    ax.text(0.02,0.84, plt_str['interval'],   fontsize=fs, va='center',ha='left', transform=ax.transAxes)

    ax.text(0.98, 0.96, plt_str['timesteps'], fontsize=fs, va='center',ha='right', transform=ax.transAxes)
    ax.text(0.98, 0.92, plt_str['missing'],   fontsize=fs, va='center', ha='right', transform=ax.transAxes)
    ax.text(0.98, 0.88, plt_str['qcflags'],   fontsize=fs, va='center', ha='right', transform=ax.transAxes)
    ax.text(0.98, 0.84, plt_str['available'], fontsize=fs, va='center', ha='right', transform=ax.transAxes)

    return ax

def convert_utc_to_local_str(utc_str,offset_from_utc):

    local = pd.to_datetime(utc_str) + pd.Timedelta('%s hours' %offset_from_utc)
    local_str = local.strftime('%Y-%m-%d %H:%M:%S')

    return local_str

def convert_utc_to_local(df,offset_from_utc):

    print('converting to local time')
    tzhrs = int(offset_from_utc)
    tzmin = int((offset_from_utc - int(offset_from_utc))*60)
    df.index = df.index + np.timedelta64(tzhrs,'h') + np.timedelta64(tzmin,'m')

    return df

def convert_local_to_utc(df,offset_from_utc):

    print('converting to utc')
    tzhrs = int(offset_from_utc)
    tzmin = int((offset_from_utc - int(offset_from_utc))*60)
    df.index = df.index - np.timedelta64(tzhrs,'h') - np.timedelta64(tzmin,'m')

    return df

# std and mean function for period
def get_sigma(start,end,data,sigma):
    ''' '''
    subset = data.loc[start:end]

    std = subset.groupby(subset.index.hour).std()
    mean = subset.groupby(subset.index.hour).mean()

    high_sigma = mean + sigma*std
    low_sigma  = mean - sigma*std

    return subset,high_sigma,low_sigma,mean

def calc_sigma_data(alldata, sigma_vals, sitepath, window=30, plotdetail=False):
    '''looks for outliers by caclulating which values in each hour within the window are outside the standard deviation x sigma'''

    alldirtydata = pd.DataFrame()
    allcleandata = pd.DataFrame()

    for flux in alldata.columns:

        sigma = sigma_vals[flux]

        print(f"analysing {flux} for {sigma} sigma")

        to_clean = alldata[[flux]]

        out_period_frames = []
        outside_sum = 0

        # select first handling period
        start = pd.Timestamp(to_clean.index[0])
        end = start + pd.DateOffset(days=window) - pd.DateOffset(minutes=1)

        totdays = (to_clean.index[-1] - to_clean.index[0]).components.days + 1

        # now loop over periods in year with steps of ndays
        for nloop in range(1,totdays,window):
            # print(f"analysing {flux}: day {nloop}-{nloop+window} for {sigma} sigma", end='\r')

            # # get subset dataframe info for period
            subset, high, low, mean = get_sigma(start,end,to_clean,sigma)

            # check if data is outside bounds in each hour
            hour_frames = []

            #########################
            for i in range(24):

                # select hour in subset and create dataframe if outside range
                df = subset[subset.index.hour==i]
                hour_outside = df[(df < low.loc[i]) | (df > high.loc[i])]
                # append each hour for later concat
                hour_frames.append(hour_outside)

                # hourly detailed plot
                if plotdetail:

                    if i == 0:
                        plt.close('all')
                        fig = plt.figure(figsize=(14,8))

                    ax = fig.add_subplot(4, 6, i+1)

                    # plot all points in black
                    ax.scatter(x=df.index,y=df,alpha=0.5,c='k',s=6,marker='o',edgecolors='none')

                    for flux in df.columns:
                        # df.plot.scatter(k,flux, ax=ax)
                        ax.plot([start,end],[high.loc[i,flux],high.loc[i,flux]], lw=0.4, color='r')
                        ax.plot([start,end],[low.loc[i,flux], low.loc[i,flux]],  lw=0.4, color='r')
                        ax.plot([start,end],[mean.loc[i,flux],mean.loc[i,flux]], lw=1.5, color='k')

                    ax.scatter(x=df.index,y=hour_outside,s=16,marker='o',edgecolors='r',color='none')

                    ax.text(0.01,1.08,'hour %s' %i,transform=ax.transAxes,va='top',ha='left',fontsize=8)
                    ax.text(0.99,1.08,'n > sigma = %s' %(hour_outside.count()[0]),transform=ax.transAxes,va='top',ha='right',fontsize=8)
                    # ax.set_xticks([start,end])
                    ax.set_xticks([])
                    ax.tick_params(labelsize=7)

                    outside_sum = outside_sum + hour_outside.count()[0]

                    if i == 23:
                        title_text = f'{flux} for days {nloop} - {nloop+window} with n > sigma({sigma}) = {outside_sum}'

                        fig.suptitle(title_text ,x=0.5,y=0.92, fontsize=16)
                        # plt.show()
                        plt.savefig(f'{sitepath}/obs_plots/{flux}_qc_sigma_{nloop}_{nloop+window}.{img_fmt}' , dpi=300,bbox_inches='tight')
                        plt.close('all')

            #########################

            subset_out = pd.concat(hour_frames)

            # append each period for later collection
            out_period_frames.append(subset_out)

            start = end + pd.DateOffset(minutes=1)
            end = start + pd.DateOffset(days=window) - pd.DateOffset(minutes=1)

        # collect outlier dataframes into single sorted set
        dirtydata = pd.concat(out_period_frames)
        dirtydata = dirtydata.sort_index()

        # get clean data by masking outliers over period
        cleandata = to_clean.copy()
        cleandata = cleandata.mask(dirtydata.notna())

        alldirtydata[flux] = dirtydata[flux]
        allcleandata[flux] = cleandata[flux]

    return alldirtydata,allcleandata

if __name__ == "__main__":

    print('run from pipeline_functions.py')


