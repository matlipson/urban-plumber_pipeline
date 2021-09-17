# Urban-PLUMBER site data processing pipeline

This code is used to produce data associated with the manuscript:

>  "Harmonized, gap-filled dataset from 20 urban flux tower sites"

The code takes data provided by observing groups in various formats and creates a standardised, harmonised, 
gap filled dataset for each of the 20 sites at 30 and 60 minute resolutions (depending on data provided).

This is the code base. Input data (i.e. from observing groups, ERA5, WFDE5 and other global datasets) is held seperately.

Timeseries plots are available at https://urban-plumber.github.io/sites ([archive](https://doi.org/10.5281/zenodo.5507036)).

Code for associated manuscript figures are available at https://doi.org/10.5281/zenodo.5508694

Timeseries output data are not yet publicly available.

## Included files

 - `run_all.py`: wrapper to run pipeline at all sites
 - `pipeline_functions.py`: main pipeline and shared functions
 - `qc_observations`: quality control processing
 - `convert_nc_to_text.py`: for converting previously created netcdf output to text format (for convenience)
 - `plot_sitemaps.py`: for plotting regional and local map images
 - in `sites/SITENAME/`:
    - `create_dataset_SITENAME`: site specific wrapper for setting site information, importing site observations and calling pipeline functions to process and create output.
    - `SITENAME_sitedata_vX.csv`: site specific numerical metadata, e.g. latitude, longitude, surface cover fractions, morphology, population density etc. Includes sources.

## Usage

Code is written in Python 3.8 with dependencies including numpy, pandas, xarray, statsmodels, matplotlib, ephem and cartopy.

The folder input_data is required for processing (not included in this repository).

#### For processing all sites

 - use `/run_all.py`
 - edit variables `projpath` to point to this repository, and `datapath` to point to input data
 - edit the sitelist as required
 - type `python run_all.py`

#### For all processing one site

 - use `/sites/SITENAME/create_dataset_SITENAME.py`
 - edit variables `projpath` to point to this repository, and `datapath` to point to input data (or use arguments at the command line, see below)
 - update the output version with `out_suffix`
 - type `python create_dataset_SITENAME.py`

Optional arguments for running `create_dataset_SITENAME.py` are:

 - `-- log`:        logs print statements in processing to file
 - `-- projpath XXX`:   changes the root project folder
 - `-- datapath XXX`:   changes the folder containing raw site data
 - `-- global`:     provides site characteristics from global datasets (large files)
 - `-- existing`:   loads previously processed nc files (if running after processing)

## Outputs

When run the following outputs are produced for each of the 20 sites within the `site/SITENAME` folder:

- `index.md`: markdown file for displaying key site metadata and output plots in a single webpage

 - `timeseries/`:
    - `SITENAME_raw_observations_vX.nc`: site observed timeseries before project-wide quality control.
    - `SITENAME_clean_observations_vX.nc`: site observed timeseries after project-wide quality control. Includes site characteristic metadata.
    - `SITENAME_metforcing_vX.nc`: site observed timeseries after project-wide quality control and gap filling.
    - `SITENAME_era5_corrected_vX.nc`: site ERA5 surface data (1990-2020) with bias corrections as applied in the final dataset.
    - `SITENAME_era5_linear_vX.nc`: site ERA5 surface data (1990-2020) with bias corrections using a linear regression on observations.
    - `SITENAME_ghcnd_precip.csv`: continuous daily precipitation timeseries using the nearest available GHCND station observations.

 - `obs_plots/`:
    - `all_obs_qc.png`: a summary plot of all included observations as provided (i.e. with nearby tower gap filling but before project gap filling, with project quality controlled periods flagged)
    - `VARIBALE_gapfilled_forcing.png`: timeseries after gap filling and prepending with corrected ERA5 data

 - `era_correction/`:
    - mean hourly diurnal and timeseries images for each corrected variable comparing bias correction methods. 'all' includes all observed data (after qc) for for training and testing. 'out-of-sample' uses an 80/20 approach to train/test data.
    - `SITENAME_erabias_VARIABLE`: the corrections applied to ERA5 data
 - `processing/`:
    - timeseries (csv) for qc flagged observations, with seperate timeseries for each qc step ('range','constant','night','sigma') and all 'dirty'.
    - error metric results when comparing ERA5 and bias corrected timeseries with available observations at 60 minute resolution. 'all' includes all observed data (after qc) for for training and testing. 'out-of-sample' uses an 80/20 approach to train/test data.
 - `precip_plots/`:
    - a series of plots comparing cumulative precipitation timeseries. The 'snow_correction' plot shows precipitation after including snowfall from ERA5 and after partitioning to maintain mass balance (see `partition_precip_to_snowf_rainf` subroutine in `pipeline_functions.py`).
 - `images/`:
    - various maps at regional and local scales. `SITENAME_site_photo.jpg` provided seperately by observing groups.
 - `log_processing_AU-Preston_v0.9.txt`: a log of the print statements through running the create_dataset_SITENAME scripts.


## Authors

This code base was developed by Mathew Lipson: https://orcid.org/0000-0001-5322-1796

Co-authors for the associated manuscript are: Sue Grimmond, Martin Best, Andreas Christen, Andrew Coutts, Ben Crawford, 
Bert Heusinkveld, Erik Velasco, Helen Claire Ward, Hirofumi Sugawara, Je-Woo Hong, Jinkyu Hong, Jonathan Evans, 
Joseph McFadden, Keunmin Lee, Krzysztof Fortuniak, Leena Järvi, Matthias Roth, Nektarios Chrysoulakis, Nigel Tapper, 
Oliver Michels, Simone Kotthaus, Stevan Earl, Sungsoo Jo, Valéry Masson, Winston Chow, Wlodzimierz Pawlak, Yeon-Hee Kim.

## Acknowledgements

Mathew Lipson acknowledges the input from all co-authors, plus the guidance and suggestions from Andy Pitman, Gab Abramowitz, Anna Ukkola, Arden Burrell and Paola Petrelli.

Mathew is supported by supported by UNSW Sydney, the Australian Research Council (ARC) Centre of Excellence for Climate System Science (grant CE110001028) and ARC Centre of Excellence for Climate Extremes (grant CE170100023).

