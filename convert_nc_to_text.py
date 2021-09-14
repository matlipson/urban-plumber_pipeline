'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Converts netcdf to text"
__version__ = "2021-09-08"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__description__ = 'Converts "clean_observations" and "metforcing" netcdf (nc) files in the Urban-PLUMBER project to text'


import os
import sys
import importlib
import xarray as xr

# data path (local or server)
oshome=os.getenv('HOME')
projpath = f'{oshome}/git/urban-plumber_pipeline'                  # root of repository

sys.path.append(projpath)
import pipeline_functions
importlib.reload(pipeline_functions)

###################

sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

version = 'v0.9'

###################

def main():

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        try:

            forcing_ds = xr.open_dataset(f'{projpath}/sites/{sitename}/timeseries/{sitename}_metforcing_{version}.nc')
            clean_ds = xr.open_dataset(f'{projpath}/sites/{sitename}/timeseries/{sitename}_clean_observations_{version}.nc')
            
            print(f'writing {sitename} forcing to text file')
            fpath = f'{projpath}/sites/{sitename}/timeseries/{sitename}_metforcing_{version}.txt'
            pipeline_functions.write_netcdf_to_text_file(ds=forcing_ds,fpath_out=fpath,ds_type='forcing')

            print(f'writing {sitename} clean observations to text file')
            fpath = f'{projpath}/sites/{sitename}/timeseries/{sitename}_clean_observations_{version}.txt'
            pipeline_functions.write_netcdf_to_text_file(ds=clean_ds,fpath_out=fpath,ds_type='clean')

        except Exception as e:
            print(f'WARNING: could not convert {sitename} to text')
            print(e)

    return


if __name__ == "__main__":
    main()
