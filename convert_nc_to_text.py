'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

This software is licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Converts netcdf to text"
__version__ = "2021-09-20"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__description__ = 'converts any netcdf (nc) file found in the sitename/timeseries folder to text'


import os
import sys
import xarray as xr
import glob

# data path (local or server)
oshome=os.getenv('HOME')

projpath = '.'

###################

sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

###################

def main():

    for sitename in sitelist:

        fpaths = glob.glob(f'{projpath}/sites/{sitename}/timeseries/*.nc')

        for fpath in fpaths:
            print(f'converting {fpath}')

            ds = xr.open_dataset(fpath)
            txt_fpath = fpath[:-2]+'txt'

            write_netcdf_to_text_file(ds=ds,fpath_out=txt_fpath)

            print(f'done! see {txt_fpath}')

    return

###################

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

if __name__ == "__main__":
    main()
