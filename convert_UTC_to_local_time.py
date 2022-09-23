'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2022 Mathew Lipson

This software is licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Converts a netcdf file from UTC to local time"
__version__ = "2022-09-15"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

import os
import xarray as xr
import pandas as pd

###################


sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

# example filepath, site and file to convert
projpath = '.'
sitename = sitelist[0]

fpath = f'{projpath}/sites/{sitename}/timeseries/{sitename}_clean_observations_v1.nc'
converted_fpath = f'{projpath}/converted/{sitename}_clean_observations_v1_localstandardtime.nc'

# ensure output directory exists
outpath = os.path.dirname(converted_fpath)
if not os.path.exists(outpath):
    print(f'creating {outpath}')
    os.makedirs(outpath)

###################

def main():
    
    print(f'converting {fpath}')

    ds = xr.open_dataset(fpath)

    ds_converted = convert_utc_to_local(ds)

    ds_converted.to_netcdf(converted_fpath)
    
    print(f'done! see {converted_fpath}')

    return

def convert_utc_to_local(ds):
    '''converts an xarray dataset to local time'''

    if ds.time_shown_in == 'UTC':

        print('converting to local time')

        utc_times = ds.time.values
        offset = ds.local_utc_offset_hours

        tzhrs = int(offset)
        tzmin = int((offset - int(offset))*60)
        local_times = ds.time.values + pd.Timedelta(hours=tzhrs) + pd.Timedelta(minutes=tzmin)

        ds = ds.assign_coords(time=local_times)

        ds.attrs['time_coverage_start'] = str(pd.to_datetime(ds.time_coverage_start) + pd.Timedelta(hours=ds.local_utc_offset_hours))
        ds.attrs['time_coverage_end'] = str(pd.to_datetime(ds.time_coverage_end) + pd.Timedelta(hours=ds.local_utc_offset_hours))
        ds.attrs['time_shown_in'] = 'local standard time'

    else:
        print('Time does not appear to be in UTC (see ds.time_shown_in')

    return ds

###################

if __name__ == "__main__":
    main()
