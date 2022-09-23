'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright 2022 Mathew Lipson

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

__title__ = "Collect all obs into one netCDF"
__version__ = "2022-09-22"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

# import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

projpath = '.'
version = 'v1'

sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

keys = ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Snowf','Wind_N',
        'Wind_E','SWup','LWup','Qle','Qh','Qtau','SoilTemp','Qg',
        'SWdown_qc','LWdown_qc','Tair_qc','Qair_qc','PSurf_qc','Rainf_qc','Snowf_qc',
        'Wind_N_qc','Wind_E_qc','SWup_qc','LWup_qc','Qle_qc','Qh_qc','Qtau_qc','SoilTemp_qc','Qg_qc',
        'station_name','station_long_name','station_contact','station_reference','station_utc_offset']

################################################################################

def main(convert_to_local=False):
    '''create single file from site timeseries in UTC or local standard time'''

    ds_list = []

    for i,sitename in enumerate(sitelist):

        # open site, convert time and append to list
        tmp = get_station_data(sitename)
        if convert_to_local:
            fname_suffix = 'localstandardtime'
            tmp = convert_utc_to_local(tmp)
        else:
            fname_suffix = 'UTC'
            # save metadata to file
            with open ('./obs_in_one/site_metadata.md', 'a') as f:
                f.write(f'## {sitename}: {tmp.attrs["long_sitename"]}\n\n')
                for key, value in tmp.attrs.items():
                    f.write(f' - {key}: {value}\n')
                f.write('\n\n')

        tmp = tmp.expand_dims({'station':[i]})

        ds_list.append(tmp)

    ds = xr.merge(ds_list)[keys]

    # set attributes
    ds.attrs = {'title': 'Flux tower observations (after QC) from all sites in the Urban-PLUMBER collection'}
    for key in ['version','featureType','license','time_shown_in','project','project_contact','source']:
        ds.attrs[key] = tmp.attrs[key]
    ds.attrs['comment'] = 'for station metadata refer to site_metadata.md or the full archive site folder'
    ds['station'].attrs['station_ids'] = [f'{i}: {sitename}' for i,sitename in enumerate(sitelist)]

    data_vars = ['SWdown','LWdown','Tair','Qair','PSurf','Rainf','Snowf','Wind_N','Wind_E','SWup','LWup','Qle','Qh','Qtau','SoilTemp','Qg']
    for key in data_vars:
        ds[key].encoding = {}
        ds[key].encoding.update({'zlib':True, '_FillValue':-999,'dtype':'float32'})
        key = key+'_qc'
        ds[key].encoding.update({'zlib':True, 'dtype':'int8'})

    ds.to_netcdf(f'{projpath}/obs_in_one/UP_all_clean_observations_{fname_suffix}_{version}.nc')

    return

def get_station_data(sitename):

    fpath = f'{projpath}/sites/{sitename}/timeseries/{sitename}_clean_observations_{version}.nc'
    tmp = xr.open_dataset(fpath)

    tmp['station_name'] = sitename
    tmp['station_name'].attrs = {'long_name': 'station name'}

    tmp['station_utc_offset'] = tmp.attrs['local_utc_offset_hours']
    tmp['station_utc_offset'].attrs['long_name'] = 'local time to UTC offset in hours'

    tmp['station_long_name'] = tmp.attrs['long_sitename']
    tmp['station_long_name'].attrs['long_name'] = 'long name of station, including country'

    tmp['station_contact'] = tmp.attrs['observations_contact']
    tmp['station_contact'].attrs['long_name'] = 'contacts for station data originators'

    tmp['station_reference'] = tmp.attrs['observations_reference']
    tmp['station_reference'].attrs['long_name'] = 'references for the station site observations'

    return tmp

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

def make_readme():

    readme_str = '''

## All observations in one file

This folder contains all site clean observations (after qc) in a single netcdf file (for convenience).
It is a subset of data contained in the full collection available from: https://doi.org/10.5281/zenodo.5517550

Gap-filled timeseries, ERA5 derived timeseries, observations prior to quality control, text formats, plots and photos are not included in this archive. 
Only tower observations after quality control are included, in two files:

- `UP_all_clean_observations_UTC_v1.nc`: in coordinated universal time (UTC)
- `UP_all_clean_observations_localstandardtime_v1.nc`: in local standard time

This data is associated with the manuscript "Harmonized gap-filled datasets from 20 urban flux tower sites", 
published in Earth System Science Data, 2022.

Corresponding author: Mathew Lipson (m.lipson@unsw.edu.au)

Authors: Mathew Lipson, Sue Grimmond, Martin Best, Andreas Christen, Andrew Coutts, Ben Crawford, 
Bert Heusinkveld, Erik Velasco, Helen Claire Ward, Hirofumi Sugawara, Je-Woo Hong, Jinkyu Hong, Jonathan Evans, 
Joseph McFadden, Keunmin Lee, Krzysztof Fortuniak, Leena Järvi, Matthias Roth, Nektarios Chrysoulakis, Nigel Tapper, 
Oliver Michels, Simone Kotthaus, Stevan Earl, Sungsoo Jo, Valéry Masson, Winston Chow, Wlodzimierz Pawlak, Yeon-Hee Kim.


## Usage

Using python with xarray, then some example usage code is below:

```
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset('UP_all_clean_observations_localstandardtime_v1.nc')

# select data from station 5: 'FR-Capitole', and plot SWdown
key, sid = 'SWdown',5
fig,ax = plt.subplots()
ds.sel(station=sid)[key].plot(ax=ax)
ax.set_title(f"{key} at {ds.station.attrs['station_ids'][sid]}")
plt.savefig('example_plot1.jpg')

# plot all site mean SWdown by hour of day
key = 'SWdown'
grp = ds[key].groupby('time.hour').mean()
fig,ax = plt.subplots()
grp.plot.line(ax=ax,x='hour',add_legend=False)
ax.set_title(f'{key} mean at hour of day')
plt.savefig('example_plot2.jpg')

# plot the mean Tair for each month and each station
key = 'Tair'
grp = ds[key].groupby('time.month').mean()
fig,ax = plt.subplots()
grp.plot(ax=ax)
ax.set_yticks(range(0,21))
ax.set_xticks(range(1,13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title(f'site {key} mean for month of year')
fig.savefig('example_plot3.jpg')
```

SWdown (station 5)         |  SWdown all sites by hour | Tair all sites by month
:-------------------------:|:-------------------------:|:-----------------------:
[![](./example_plot1.jpg)](./example_plot1.jpg)   |  [![](./example_plot2.jpg)](./example_plot2.jpg) | [![](./example_plot3.jpg)](./example_plot3.jpg)


## Station IDs

Each flux tower is contained within a station dimension numbered 0 to 20, with the following IDs:

 - 0: AU-Preston
 - 1: AU-SurreyHills
 - 2: CA-Sunset
 - 3: FI-Kumpula
 - 4: FI-Torni
 - 5: FR-Capitole
 - 6: GR-HECKOR
 - 7: JP-Yoyogi
 - 8: KR-Jungnang
 - 9: KR-Ochang
 - 10: MX-Escandon
 - 11: NL-Amsterdam
 - 12: PL-Lipowa
 - 13: PL-Narutowicza
 - 14: SG-TelokKurau06
 - 15: UK-KingsCollege
 - 16: UK-Swindon
 - 17: US-Baltimore
 - 18: US-Minneapolis1
 - 19: US-Minneapolis2
 - 20: US-WestPhoenix

## Data variables

The data files contain the following variables (noting variables will be empty for some sites):

 - SWdown: Downward shortwave radiation at measurement height
 - LWdown: Downward longwave radiation at measurement height
 - Tair: Air temperature at measurement height
 - Qair: Specific humidity at measurement height
 - PSurf: Air pressure at measurement height
 - Rainf: Rainfall rate (positive downward)
 - Snowf: Snowfall rate (positive downward)
 - Wind_N: Northward wind component at measurement height
 - Wind_E: Eastward wind component at measurement height
 - SWup: Upwelling shortwave radiation flux (positive upward)
 - LWup: Upwelling longwave radiation flux (positive upward)
 - Qle: Latent heat flux (positive upward)
 - Qh: Sensible heat flux (positive upward)
 - Qtau: Momentum flux (positive downward)
 - SoilTemp: Average layer soil temperature
 - Qg: Ground heat flux (positive downward)
 - SWdown_qc: Quality control (qc) flag for SWdown
 - LWdown_qc: Quality control (qc) flag for LWdown
 - Tair_qc: Quality control (qc) flag for Tair
 - Qair_qc: Quality control (qc) flag for Qair
 - PSurf_qc: Quality control (qc) flag for PSurf
 - Rainf_qc: Quality control (qc) flag for Rainf
 - Snowf_qc: Quality control (qc) flag for Snowf
 - Wind_N_qc: Quality control (qc) flag for Wind_N
 - Wind_E_qc: Quality control (qc) flag for Wind_E
 - SWup_qc: Quality control (qc) flag for SWup
 - LWup_qc: Quality control (qc) flag for LWup
 - Qle_qc: Quality control (qc) flag for Qle
 - Qh_qc: Quality control (qc) flag for Qh
 - Qtau_qc: Quality control (qc) flag for Qtau
 - SoilTemp_qc: Quality control (qc) flag for SoilTemp
 - Qg_qc: Quality control (qc) flag for Qg
 - station_name: station name
 - station_long_name: long name of station, including country
 - station_contact: contacts for station data originators
 - station_reference: references for the station site observations
 - station_utc_offset: local time to UTC offset in hours

## Acknowledgements

We would like to thank the vast number of people involved in day-to-day running of these sites that have been involved in instrument and tower installations (permitting, purchasing and site installation), routine (and unexpected event) maintenance, data collection, routine data processing and final data processing. We thank all those that have provided sites for the towers to be located and sometimes power and internet access. We acknowledge the essential funding for the instrumentation and other infrastructure, for staff (administrative, technical and scientific) and students for these activities. We also thank those who offered data for use in this project which are not included at this time.

The project coordinating team are supported by UNSW Sydney and the Australian Research Council (ARC) Centre of Excellence for Climate System Science (grant CE110001028), University of Reading, the Met Office and ERC urbisphere 855005. Computation support from the ARC Centre of Excellence for Climate Extremes (grant CE170100023) and National Computational Infrastructure (NCI) Australia. Contains modified Copernicus Climate Change Service Information. Site-affiliated acknowledgments are listed in Table 10 of the associated manuscript.

'''
    with open(f'./obs_in_one/README.md', 'w') as f:
        f.write(readme_str)

    metadata_str = '''
# Sites metadata

As the metadata from each site file cannot be stored in the single file netcdf, it is reproduced here:

'''


    with open(f'./obs_in_one/site_metadata.md', 'w') as f:
        f.write(metadata_str)

    return

if __name__ == "__main__":
    make_readme()
    main(convert_to_local=False)
    main(convert_to_local=True)
