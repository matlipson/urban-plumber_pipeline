'''
Urban-PLUMBER processing code
Associated with the manuscript: Harmonized, gap-filled dataset from 20 urban flux tower sites

Maps developed based on: 
Hrisko, J. (2020). Geographic Visualizations in Python with Cartopy. Maker Portal. 
https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
'''

__title__   = "Plot OpenStreetMap site and regional maps"
__version__ = "2021-09-08"
__author__  = "Mathew Lipson"
__email__   = "m.lipson@unsw.edu.au"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import cartopy
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs

import cartopy.io.img_tiles as cimgt
import io
from urllib.request import urlopen, Request
from PIL import Image
import shapely

oshome=os.getenv('HOME')
projpath = f'{oshome}/git/urban-plumber_pipeline'    # root of repository

##########################################################################

def main(sitelist):

    for sitename in sitelist:

        sitepath = f'{projpath}/sites/{sitename}'

        # make plot directory if necessary
        if not os.path.exists(f'{sitepath}/images'):
            print('making dir')
            os.makedirs(f'{sitepath}/images')

        vi='v1'

        print(f'loading site parameters for {sitename}')
        fpath = f'{sitepath}/{sitename}_sitedata_{vi}.csv'
        sitedata_full = pd.read_csv(fpath, index_col=1, delimiter=',')
        sitedata = pd.to_numeric(sitedata_full['value'])

        # style can be 'map' or 'satellite'
        for style in ['map','satellite']:
            osm_image(sitename, sitepath, sitedata,style=style,radius=500,npoints=0)

        plot_grid_site(sitename, sitepath, sitedata,style='map')

    return

##########################################################################

def osm_image(sitename,sitepath,sitedata,style='satellite',radius=500,npoints=500):
    ''' make OpenStreetMap satellite or map image with circle and random points
    Heavily based on code by Joshua Hrisko at:
        Hrisko, J. (2020). Geographic Visualizations in Python with Cartopy. Maker Portal.
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy

    change np.random.seed() number to produce different (reproducable) random patterns of points'''

    lat, lon = sitedata['latitude'], sitedata['longitude']

    if style=='map':
        cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.OSM() # spoofed, downloaded street map
        copyright = '© OpenStreetMap'
    elif style =='satellite':
        cimgt.QuadtreeTiles.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.QuadtreeTiles() # spoofed, downloaded street map
        copyright = '© OpenStreetMap, © Microsoft'
    else:
        print('no valid style')

    ############################################################################

    plt.close('all')
    fig = plt.figure(figsize=(10,10)) # open matplotlib figure
    ax = plt.axes(projection=img.crs) # project using coordinate reference system (CRS) of street map
    data_crs = ccrs.PlateCarree()

    ax.set_title(f'{sitename} ({lat},{lon})',fontsize=15)

    # NOTE: scale specifications should be selected based on radius:
    scale = int(120/np.log(radius))
    scale = (scale<20) and scale or 19
    # or change extent manually
    # -- 2     = coarse image, select for worldwide or continental scales
    # -- 4-6   = medium coarseness, select for countries and larger states
    # -- 6-10  = medium fineness, select for smaller states, regions, and cities
    # -- 10-12 = fine image, select for city boundaries and zip codes
    # -- 14+   = extremely fine image, select for roads, blocks, buildings

    extent = calc_extent(lon,lat,radius*1.1)
    ax.set_extent(extent) # set extents

    ax.add_image(img, int(scale)) # add OSM with zoom specification

    # add site
    ax.plot(lon,lat, color='black', marker='x', ms=7, mew=3, transform=data_crs)
    ax.plot(lon, lat, color='red', marker='x', lw=0, ms=6, mew=2, transform=data_crs)

    if npoints>0:
        # set random azimuth angles (seed for reproducablity)
        np.random.seed(16)
        rand_azimuths_deg = np.random.random(npoints)*360

        # set random distances (seed for reproducablity)
        np.random.seed(24)
        rand_distances = radius*np.sqrt((np.random.random(npoints)))
        # rand_distances = radius*(np.random.random(npoints))

        rand_lon = cgeo.Geodesic().direct((lon,lat),rand_azimuths_deg,rand_distances).base[:,0]
        rand_lat = cgeo.Geodesic().direct((lon,lat),rand_azimuths_deg,rand_distances).base[:,1]

        ax.plot(rand_lon,rand_lat,color='black',lw=0,marker='x',ms=4.5,mew=1.0,transform=data_crs)
        ax.plot(rand_lon,rand_lat,color='yellow',lw=0,marker='x',ms=4,mew=0.5, transform=data_crs)

    # add cartopy geodesic circle
    circle_points = cgeo.Geodesic().circle(lon=lon, lat=lat, radius=radius)
    geom = shapely.geometry.Polygon(circle_points)
    ax.add_geometries((geom,), crs=ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=2.5)

    radius_text = cgeo.Geodesic().direct(points=(lon,lat),azimuths=35,distances=radius).base[:,0:2][0]
    stroke = [pe.Stroke(linewidth=2.5, foreground='w'), pe.Normal()]
    ax.text(radius_text[0]+0.0002,radius_text[1],f'r={radius} m', color='red', 
        fontsize=8, ha='left',va='bottom', path_effects=stroke, transform=data_crs)

    # nc.plot(ax=ax,alpha=0.75,transform=data_crs)

    gl = ax.gridlines(draw_labels=True, crs=data_crs,
                        color='k',lw=0.5)

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    # copyright notice
    ax.text(0.99,0.01,copyright, color='k', fontsize=6, ha='right', path_effects=stroke, transform=ax.transAxes)

    # plt.show()

    fig.savefig(f'{sitepath}/images/{sitename}_site_{style[:3]}.jpg', dpi=150, bbox_inches='tight')

    return

def plot_grid_site(sitename,sitepath,sitedata, style='map'):

    lat, lon = sitedata['latitude'], sitedata['longitude']

    site_lat_box,site_lon_box = calc_box(loc=(lat,lon),dist=1000)

    era_lats = np.arange(-90,90.25,0.25)
    era_lons = np.arange(-180,180,0.25)

    wfd_lats = np.arange(-89.75,90,0.5)
    wfd_lons = np.arange(-179.75,180,0.5)

    era_lat_box, era_lon_box = define_grid_box(lat,lon,era_lats,era_lons,0.25)
    wfd_lat_box, wfd_lon_box = define_grid_box(lat,lon,wfd_lats,wfd_lons,0.5)

    # MANUAL EXTENTS AND SCALE
    extent = calc_extent(lon,lat,70000)
    scale = 10
    if sitename in ['FI-Torni','FI-Kumpula']:
        scale = 9
    data_crs = ccrs.PlateCarree()

    img = cimgt.OSM()
    proj = img.crs

    plt.close('all')
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection=proj)

    ax.set_title(f'{sitename} ({lat},{lon})',fontsize=15, pad=4)

    if style == 'map':
        # alt map
        cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.OSM() # spoofed, downloaded street map
        ax.add_image(img, int(scale)) # add OSM with zoom specification
        # add transparent white rectangle
        xy = (extent[0],extent[2])
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        rec = patches.Rectangle(xy=xy,width=width,height=height,
            facecolor='white',alpha=0.25,transform=data_crs)
        ax.add_artist(rec)
        copyright = '© OpenStreetMap'

    elif style =='satellite':
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.QuadtreeTiles() # spoofed, downloaded street map
        copyright = '© OpenStreetMap, © Microsoft'

    if style == 'cartopy':
        # add_map_image(ax, alpha=0.45)
        ax.coastlines(lw=0.5,resolution='10m', color='k')
        ax.add_feature(cartopy.feature.NaturalEarthFeature(category='physical', name='land', scale='10m',
                                                        facecolor='green',alpha=0.25))

    ax.set_extent(extent,crs=data_crs)

    pad = 0.01
    if abs(lon - era_lon_box[0]) < 0.1 and abs(lat - era_lat_box[1]) < 0.1: 
        print('lower left text')
        era_lon_text = era_lon_box[0]+pad
        era_lat_text = era_lat_box[0]
        va = 'bottom'
    else:
        print('upper left text')
        era_lon_text = era_lon_box[0]+pad
        era_lat_text = era_lat_box[1]-pad
        va = 'top'

    # plot boxes
    stroke = [pe.Stroke(linewidth=2.5, foreground='w'), pe.Normal()]
    
    # plot WFDE5
    # ax.plot(wfd_lon_box,wfd_lat_box,color='blue',transform=data_crs)
    # ax.text(wfd_lon_box[0]+pad,wfd_lat_box[1]-pad, 'WFDE5', color='blue', 
    #     fontsize=14, ha='left',va='top', path_effects=stroke, transform=data_crs)

    ax.plot(era_lon_box,era_lat_box,color='red',transform=data_crs, linewidth=1)
    ax.text(era_lon_text,era_lat_text, 'ERA5', color='red', 
        fontsize=14, ha='left',va=va,path_effects=stroke, transform=data_crs)

    # plot site
    # add cartopy geodesic circle
    circle_points = cgeo.Geodesic().circle(lon=lon, lat=lat, radius=500)
    geom = shapely.geometry.Polygon(circle_points)
    ax.add_geometries((geom,), crs=ccrs.PlateCarree(), edgecolor='black', facecolor='White', linewidth=1.5,
        path_effects = [pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
    ax.text(lon,lat-pad, sitename, color='black', 
        fontsize=14, ha='center',va='top', path_effects=stroke, transform=data_crs)

    if sitename == 'SG-TelokKurau':
        era_lat_box2, era_lon_box2 = define_grid_box(lat+0.25,lon-0.25,era_lats,era_lons,0.25)
        ax.plot(era_lon_box2,era_lat_box2,color='red', ls='dashed',transform=data_crs)
        ax.text(era_lon_box2[0]+pad,era_lat_box2[1]-pad, 'ERA5 alternate', color='red', 
            fontsize=14, ha='left',va='top', path_effects=stroke, transform=data_crs)

    # plot distance bar
    start = (extent[0]+0.05,extent[2]+0.05)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=10000).base[:,0:2][0]
    ax.plot([start[0],end[0]],[start[1],end[1]], color='k', linewidth=1, mew=1, transform=data_crs)
    ax.text(start[0]+0.005,start[1]+0.005, '10 km', color='black', 
        fontsize=10, ha='left',va='bottom', path_effects=stroke, transform=data_crs)

    # gridlines
    xticks = np.arange(round(lon)-8,round(lon)+8,0.25)
    yticks = np.arange(round(lat)-8,round(lat)+8,0.25)

    gl = ax.gridlines( draw_labels=True, xlocs = xticks, ylocs = yticks, crs=data_crs,
                        color='0.75',lw=0.5, ls='dashed')

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # copyright notice
    ax.text(0.99,0.01,copyright, color='k', fontsize=6, ha='right', path_effects=stroke, transform=ax.transAxes)

    # plt.show()

    fig.savefig(f'{sitepath}/images/{sitename}_region_map.jpg', dpi=150, bbox_inches='tight')


    return

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def calc_extent(lon,lat,dist):
    '''This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    '''

    dist_cnr = np.sqrt(2*dist**2)
    top_left = cgeo.Geodesic().direct(points=(lon,lat),azimuths=-45,distances=dist_cnr).base[:,0:2][0]
    bot_right = cgeo.Geodesic().direct(points=(lon,lat),azimuths=135,distances=dist_cnr).base[:,0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent

def calc_box(loc,dist):
    '''calculates latitude and longitude of square
    Inputs:
        loc: (lat/lon) of location in degrees
        dist: size of box in metres
    Outputs:
        grid_lat[] latitude values for box (including return to start)
        grid_lon[] longitude values for box (including return to start)
    '''
    earth_radius = 6.3781E6
    lat,lon = loc[0],loc[1]
    lat_rad = lat*np.pi/180.

    ratio_of_circumf = dist/(2*np.pi*earth_radius)
    lat_offset = 0.5*(ratio_of_circumf*360.)
    lon_offset = 0.5*(ratio_of_circumf*360./np.cos(lat_rad))

    grid_lat = [ lat - lat_offset,
                 lat - lat_offset,
                 lat + lat_offset,
                 lat + lat_offset,
                 lat - lat_offset,
                 lat - lat_offset]

    grid_lon = [ lon - lon_offset,
                 lon + lon_offset,
                 lon + lon_offset,
                 lon - lon_offset,
                 lon - lon_offset,
                 lon + lon_offset]

    return grid_lat, grid_lon

def define_grid_box(site_lat,site_lon,grid_lats,grid_lons,grid_res):

    grid_lat = (find_nearest(grid_lats,site_lat))
    grid_lon = (find_nearest(grid_lons,site_lon))

    grid_lat_box = [grid_lat - grid_res/2,
                    grid_lat + grid_res/2,
                    grid_lat + grid_res/2,
                    grid_lat - grid_res/2,
                    grid_lat - grid_res/2 ]

    grid_lon_box = [grid_lon - grid_res/2,
                    grid_lon - grid_res/2,
                    grid_lon + grid_res/2,
                    grid_lon + grid_res/2,
                    grid_lon - grid_res/2 ]

    return grid_lat_box, grid_lon_box

def image_spoof(self, tile):
    '''this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy'''

    url = self._image_url(tile)                # get the url of the street map API
    req = Request(url)                         # start request
    req.add_header('User-agent','Anaconda 3')  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())            # get image
    fh.close()                                 # close url
    img = Image.open(im_data)                  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

##########################################################################

if __name__ == "__main__":

    sitelist = ['AU-Preston','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole',
                'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam',
                'PL-Lipowa','PL-Narutowicza','SG-TelokKurau','UK-KingsCollege','UK-Swindon',
                'US-Baltimore','US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

    main(sitelist)
