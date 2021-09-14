'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Primary script which runs all site subscipts"
__version__ = "2021-09-08"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

import glob
import os
import sys
import importlib
import pandas as pd

# data path (local or server)
oshome=os.getenv('HOME')
projpath = f'{oshome}/git/urban-plumber_pipeline'            # root of repository
datapath = f'{oshome}/git/urban-plumber_pipeline/input_data' # raw data path (site data, global data)
webpath  = f'{oshome}/git/urban-plumber.github.io'

sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

# sitelist = ['AU-Preston'] # for testing

full_processing = True   # undertake full processing. Requires local input_data. 
create_sitemaps = True   # create regional and site maps (not required)

'''
arguments for running the create_dataset* python files are:
 -- log:        logs print statements in processing to file
 -- existing:   loads previously processed nc files
 -- global:     provides site characteristics from global datasets (large files)
 -- projpath:   allows changing the root project folder
 -- datapath:   allows updating the folder holding raw site data
'''
###################

def main():

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'
        sys.path.append(sitepath)

        fname = glob.glob(f'{sitepath}/create_dataset*')[0]
        print('')
        print(f'processing {fname}')

        if full_processing:
            os.system(f'python {fname} --log --global --projpath {projpath} --datapath {datapath}')
        else:
            os.system(f'python {fname} --existing --global --projpath {projpath} --datapath {datapath}')

    # additional plots (not required for timeseries output)
    if create_sitemaps:

        print('')
        print('plotting site maps into /sites/SITENAME/images/')
        import plot_sitemaps
        plot_sitemaps.main(sitelist)

    print('done all!')

    return

if __name__ == "__main__":
    main()
