'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Copyright (c) 2021 Mathew Lipson

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
'''

__title__ = "Ancillary functions for archiving and copying processed files"
__version__ = "2022-05-29"
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

sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
            'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
            'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon',
            'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

restricted = ['SG-TelokKurau'] # as used in model evaluation
open_sites = sorted(list(set(sitelist) - set(restricted)))

input_version = 'v0.9'
archive_version = 'v0.92'
sitedata_version = 'v1'

###################
# v0.9 original submission
# v0.92 update with SG-TelokKurau06
###################

def main():

    archive()
    # web_archive()
    # archive_light()

    # copy_to_website()

    return

def archive():

    archpath = f'{oshome}/up_archive_{archive_version}'

    if not os.path.exists(archpath):
        print(f'making {archpath} dir')
        os.mkdir(archpath)

    os.chdir(archpath)

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        # make directories if necessary
        for dirname in ['timeseries','index_files']:
            if not os.path.exists(f'{archpath}/{sitename}/{dirname}'):
                print(f'making {sitename}/{dirname} dir')
                os.makedirs(f'{archpath}/{sitename}/{dirname}')

        print(f'copying {sitename} files archive {input_version}')

        os.system(f'cp {sitepath}/log_processing_{sitename}_{input_version}.txt {archpath}/{sitename}/log_processing_{sitename}_{input_version}.txt')
        os.system(f'cp {sitepath}/{sitename}_sitedata_{sitedata_version}.csv {archpath}/{sitename}/{sitename}_sitedata_{sitedata_version}.csv')

        os.system(f'cp {sitepath}/index.html {archpath}/{sitename}/index.html')
        os.system(f'cp {sitepath}/index_files/* {archpath}/{sitename}/index_files/')

        for timeseries in ['clean_observations','era5_corrected','metforcing','raw_observations']:
            os.system(f'cp {sitepath}/timeseries/{sitename}_{timeseries}_{input_version}.nc {archpath}/{sitename}/timeseries/')
            os.system(f'cp {sitepath}/timeseries/{sitename}_{timeseries}_{input_version}.txt {archpath}/{sitename}/timeseries/')

    ########################################################################

    # RESTRICTED NO LONGER REQUIRED, ALL ARE OPEN

    # licence = 'These files are restricted. You must have written authorisation from data providers to access. Do not distribute.'
    # create_readme(archpath,licence)

    # print('creating zip archives')
    # for sitename in restricted:
    #     os.system(f'zip -r -X Urban-PLUMBER_Sites_{sitename}.zip {sitename} README.md')

    # os.system(f'zip -r -X Urban-PLUMBER_Sites_FullCollection.zip { f" ".join(sitelist)} README.md')

    ########################################################################

    licence = 'Data are licenced under CC-BY-4.0. https://creativecommons.org/licenses/by/4.0/'
    create_readme(archpath,licence)

    os.system(f'zip -r -X Urban-PLUMBER_Sitedata_OpenCollection_{archive_version}.zip { f" ".join(open_sites)} README.md')

    print('deleting unzipped folders')
    for sitename in sitelist:
        os.system(f'rm -r {archpath}/{sitename}')
    os.system(f'rm {archpath}/README.md')

    return


def archive_light():

    archpath = f'{oshome}/UP_KUOM_{input_version}'

    os.chdir(archpath)

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        # make directories if necessary
        for dirname in ['timeseries','index_files']:
            if not os.path.exists(f'{archpath}/{sitename}/{dirname}'):
                print(f'making {sitename}/{dirname} dir')
                os.makedirs(f'{archpath}/{sitename}/{dirname}')

        print(f'copying {sitename} files archive {input_version}')

        os.system(f'cp {sitepath}/log_processing_{sitename}_{input_version}.txt {archpath}/{sitename}/log_processing_{sitename}_{input_version}.txt')
        os.system(f'cp {sitepath}/{sitename}_sitedata_{sitedata_version}.csv {archpath}/{sitename}/{sitename}_sitedata_{sitedata_version}.csv')

        os.system(f'cp {sitepath}/index.html {archpath}/{sitename}/index.html')
        os.system(f'cp {sitepath}/index_files/* {archpath}/{sitename}/index_files/')

        for timeseries in ['clean_observations','era5_corrected','metforcing','raw_observations']:
            os.system(f'cp {sitepath}/timeseries/{sitename}_{timeseries}_{input_version}.nc {archpath}/{sitename}/timeseries/')
            os.system(f'cp {sitepath}/timeseries/{sitename}_{timeseries}_{input_version}.txt {archpath}/{sitename}/timeseries/')

    ########################################################################

    licence = 'These files licenced under CC-BY-4.0. https://creativecommons.org/licenses/by/4.0/'
    create_readme(archpath,licence)

    os.system(f'zip -r -X Urban-PLUMBER_Sites_KUOM.zip { f" ".join(open_sites)} README.md')

    print('deleting unzipped folders')
    for sitename in sitelist:
        os.system(f'rm -r {archpath}/{sitename}')
    os.system(f'rm {archpath}/README.md')

    return

def create_readme(archpath,licence):

    with open(f'{archpath}/README.md', 'w') as f:
        f.write(f'''
{licence}

---

# Urban-PLUMBER site data collection

Files in this folder are associated with the manuscript:

>  "Harmonized, gap-filled dataset from 20 urban flux tower sites"

Use of any data must give credit through citation of the above manuscript and other sources as appropriate.
We recommend data users consult with site contributing authors and/or the coordination team in the project planning stage. 
Relevant contacts are included in timeseries metadata.

For site information and timeseries plots see https://urban-plumber.github.io/sites.

For processing code see https://github.com/matlipson/urban-plumber_pipeline.

### Included files

Within each site folder:

- `index.html`: A summary page with site characteristics and timeseries plots.
- `SITENAME_sitedata_vX.csv`: comma seperated file for numerical site characteristics e.g. location, surface cover fraction etc.
- `timeseries/` (following files available as netCDF and txt)
    - `SITENAME_raw_observations_vX`: site observed timeseries before project-wide quality control.
    - `SITENAME_clean_observations_vX`: site observed timeseries after project-wide quality control. 
    - `SITENAME_metforcing_vX`: site observed timeseries after project-wide quality control and gap filling.
    - `SITENAME_era5_corrected_vX`: site ERA5 surface data (1990-2020) with bias corrections as applied in the final dataset.
- `log_processing_SITENAME_vX.txt`: a log of the print statements through running the create_dataset_SITENAME scripts.

### Authors

Mathew Lipson, Sue Grimmond, Martin Best, Andreas Christen, Andrew Coutts, Ben Crawford, Bert Heusinkveld, 
Erik Velasco, Helen Claire Ward, Hirofumi Sugawara, Je-Woo Hong, Jinkyu Hong, Jonathan Evans, Joseph McFadden, 
Keunmin Lee, Krzysztof Fortuniak, Leena Järvi, Matthias Roth, Nektarios Chrysoulakis, Nigel Tapper, Oliver Michels, 
Simone Kotthaus, Stevan Earl, Sungsoo Jo, Valéry Masson, Winston Chow, Wlodzimierz Pawlak, Yeon-Hee Kim.

Corresponding author: Mathew Lipson <m.lipson@unsw.edu.au>

''')

    return

def copy_from_website_light():
    ''' copies files not automatically produced by the pipeline from the website folder the pipeline folder'''

    for sitename in sitelist:
        sitepath = f'{projpath}/sites/{sitename}'
        print(f'copying image files from website')
        for image in ['site_photo.jpg']:
            os.system(f'cp {webpath}/{sitename}/images/{sitename}_{image} {sitepath}/images/{sitename}_{image}')

    subsitelist = ['FI-Kumpula','FI-Torni','JP-Yoyogi','US-Minneapolis1','US-Minneapolis2']

    for sitename in subsitelist:
        sitepath = f'{projpath}/sites/{sitename}'
        print(f'copying image files from website')
        for image in ['site_sat.jpg']:
            os.system(f'cp {webpath}/{sitename}/images/{sitename}_{image} {sitepath}/images/{sitename}_{image}')

    return

def web_archive():
    '''for archiving index.html site pages (created from index.md)'''

    archpath = f'{oshome}/up_web_archive_{input_version}'

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        # make directories if necessary
        for dirname in ['index_files']:
            if not os.path.exists(f'{archpath}/{sitename}/{dirname}'):
                print(f'making {sitename}/{dirname} dir')
                os.makedirs(f'{archpath}/{sitename}/{dirname}')

        print(f'copying {sitename} files archive {input_version}')

        os.system(f'cp {sitepath}/index.html {archpath}/{sitename}/index.html')
        os.system(f'cp {sitepath}/index_files/* {archpath}/{sitename}/index_files/')

    return

def copy_to_website():

    webpath  = f'{oshome}/git/urban-plumber.github.io'

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        for dirname in ['obs_plots','era_correction','images']:
            # make directories if necessary
            if not os.path.exists(f'{webpath}/{sitename}/{dirname}'):
                print(f'making {dirname} dir')
                os.makedirs(f'{webpath}/{sitename}/{dirname}')

        print(f'copying {sitename} files to git for website')
        os.system(f'cp {sitepath}/index.md {webpath}/{sitename}/index.md')
        os.system(f'cp {sitepath}/obs_plots/* {webpath}/{sitename}/obs_plots/')
        for flux in ['Tair','Qair','PSurf','LWdown','SWdown','Wind','Rainf']:
            os.system(f'cp {sitepath}/era_correction/{sitename}_{flux}_all_diurnal.png {webpath}/{sitename}/era_correction/')

        print(f'copying image files to website')
        for image in ['region_map.jpg','site_map.jpg','site_photo.jpg','site_sat.jpg']:
            os.system(f'cp {sitepath}/images/{sitename}_{image} {webpath}/{sitename}/images/{sitename}_{image} ')

    return

def clean_out():

    for sitename in sitelist:
        sitepath = f'{projpath}/sites/{sitename}'

        os.system(f'rm -r {sitepath}/{sitename}_siteattrs*')
        os.system(f'rm -r {sitepath}/log_processing*')
        os.system(f'rm -r {sitepath}/index.md')
        os.system(f'rm -r {sitepath}/index.html')

        print(f'removing {sitename} files')
        if os.path.exists(f'{sitepath}/obs_plots'):
            os.system(f'rm -r {sitepath}/obs_plots')
        if os.path.exists(f'{sitepath}/era_correction'):
            os.system(f'rm -r {sitepath}/era_correction')
        if os.path.exists(f'{sitepath}/precip_plots'):
            os.system(f'rm -r {sitepath}/precip_plots')
        if os.path.exists(f'{sitepath}/processing'):
            os.system(f'rm -r {sitepath}/processing')
        if os.path.exists(f'{sitepath}/timeseries'):
            os.system(f'rm -r {sitepath}/timeseries')
        if os.path.exists(f'{sitepath}/index_files'):
            os.system(f'rm -r {sitepath}/index_files')
        if os.path.exists(f'{sitepath}/images'):
            os.system(f'rm -r {sitepath}/images')

if __name__ == "__main__":
    main()
