
#python3 order_api_landsat.py LandSat8_Andromache.json 17-Oct-19 19-Oct-19 andromache           

import os
import sys
import json
import datetime
import calendar
import re
import time
import datetime
from datetime import datetime as dt
from datetime import timezone as tzo
import requests
from requests.auth import HTTPBasicAuth

#from IPython.display import Image 
#from IPython.display import display

#import rasterio
import numpy
from xml.dom import minidom

from osgeo import gdal


import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely import wkt

#import shapely.wkt
import shapely.geometry as geometry

import os
os.environ['USE_PYGEOS'] = '0'

import geopandas
from geopandas import GeoSeries, GeoDataFrame
import pandas as pd
import numpy as np

# for calculation of meters squared:
import pyproj    
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial

import scipy
from affine import Affine
import fiona
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, transform


# Better to have API Key stored as an env variable
PLANET_API_KEY =  ""

# handle globs of *.csv 
# handle globs of *.geojson

def get_remaining_quota():
    leftquota = 0.0 
    main = requests.get(
        'https://api.planet.com/auth/v1/' +
        'experimental/public/my/subscriptions',
        auth=HTTPBasicAuth(
            PLANET_API_KEY, ''))
    if main.status_code == 200:
        content = main.json()
        for item_id in content:
           if (item_id['quota_sqkm'])is not None:
               leftquota = (float(item_id['quota_sqkm'] - float(item_id['quota_used'])))
               print('Remaining Quota in SqKm: %s' % leftquota)
               return(leftquota)
           else:
               print('No Quota Allocated')
           
    else:
        print('Failed with exception code: ' + str(main.status_code))

def calculate_sqkm(df,kbuf):
    pll = df[['lat','long']]
    geometryX = [Point(xy) for xy in zip(pll['long'], pll['lat'])]
    
    lbuf = 0.0025#0.01
    xbuf = lbuf#/2.0
    mn = 0
    totalm2 = 0.0
    for pp in geometryX:
        listarray = []
        #listarray.append([pp.x, pp.y])
        listarray.append([pp.x+xbuf, pp.y-lbuf])
        listarray.append([pp.x-xbuf, pp.y-lbuf])
        listarray.append([pp.x-xbuf, pp.y+lbuf])
        listarray.append([pp.x+xbuf, pp.y+lbuf]) # self intersection?
        
        nparray = np.array(listarray)
        poly = geometry.Polygon(nparray)
        
        geom = poly
        geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)
    
        # Print the area in m^2
        #print(geom_area.area)
        totalm2 += geom_area.area
    print("Generated geojson estimated to consume",totalm2/1000000.0,"square kilometers per instance")
    return(totalm2/1000000.0)

def generate_geojson_from_coords_df(df,kbuf):
    pll = df[['lat','long']]
    geometryX = [Point(xy) for xy in zip(pll['long'], pll['lat'])]
    
    gjlist= []
    lbuf = 0.01
    xbuf = lbuf#/2.0
    mn = 0
    totalm2 = 0.0
    for pp in geometryX:
        listarray = []
        #listarray.append([pp.x, pp.y])
        listarray.append([pp.x+xbuf, pp.y-lbuf])
        listarray.append([pp.x-xbuf, pp.y-lbuf])
        listarray.append([pp.x-xbuf, pp.y+lbuf])
        listarray.append([pp.x+xbuf, pp.y+lbuf]) # self intersection?
        
        nparray = np.array(listarray)
        poly = geometry.Polygon(nparray)
        
        geom = poly
        geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)
    
        # Print the area in m^2
        #print(geom_area.area)
        totalm2 += geom_area.area
        convex_hull_x, convex_hull_y = [z.tolist() for z in poly.convex_hull.exterior.coords.xy]
        #print(poly.wkt) 
        g1 = shapely.geometry.mapping(poly)
        gjlist.append(json.dumps(g1))
        #print(json.dumps(g1,indent=1))
        #print(mn)
        mn += 1
    #print("Generated geojson estimated to consume ",totalm2/1000000.0," square kilometers per instance")
    return(gjlist)



def get_month_day_range(date):
    first_day = date.replace(day = 1)
    last_day = date.replace(day = calendar.monthrange(date.year, date.month)[1])
    return first_day, last_day


def construct_search_request_for_month(geojson_geometry,dtmonth):
    xdates = get_month_day_range(dtmonth)
    start_date = xdates[0] 
    end_date = xdates[1]
    # get images that overlap with our AOI
    geometry_filter = {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": json.loads(geojson_geometry)
            }
    
    gte = start_date.isoformat() + '.000Z' 
    lte = end_date.isoformat()  + '.000Z' 
    # get images acquired within a date range
    date_range_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {
                "gte": gte, 
                "lte": lte 
                }
            }
    # only get images which have <50% cloud coverage
    cloud_cover_filter = {
            "type": "RangeFilter",
            "field_name": "cloud_cover",
            "config": {
                "lte": 0.02
                }
            }
    
    # combine our geo, date, cloud filters
    combined_filter = {
            "type": "AndFilter",
            "config": [geometry_filter, date_range_filter, cloud_cover_filter]
            }
    
    #item_type = "PSScene3Band"
    item_type = "PSScene4Band"
    
    # API request object
    search_request = {
            "interval": "day",
            "item_types": [item_type],
            "filter": combined_filter
            }
    #print(search_request)
    return(search_request)

def construct_search_request(geojson_geometry,date_string,date_string_end):
    #print(geojson_geometry)
    start_date = datetime.datetime.strptime(date_string, '%d-%b-%y')
    end_date = datetime.datetime.strptime(date_string_end, '%d-%b-%y')
    #start_date = date - datetime.timedelta(days=days_before)
    #end_date = date + datetime.timedelta(days=days_after)
    # get images that overlap with our AOI
    geometry_filter = {
            "type": "GeometryFilter",
            "field_name": "geometry",
            #"config": json.loads(geojson_geometry)
            "config": geojson_geometry
            }
    
    gte = start_date.isoformat() + '.000Z' 
    lte = end_date.isoformat()  + '.000Z' 
    # get images acquired within a date range
    date_range_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {
                "gte": gte, 
                "lte": lte 
                }
            }
    # only get images which have <50% cloud coverage
    cloud_cover_filter = {
            "type": "RangeFilter",
            "field_name": "cloud_cover",
            "config": {
                "lte": 0.02
                }
            }
    
    # combine our geo, date, cloud filters
    combined_filter = {
            "type": "AndFilter",
            "config": [geometry_filter, date_range_filter, cloud_cover_filter]
            }
    
    #item_type = "PSScene3Band"
    #item_type = "PSScene4Band"
    item_type = "Landsat8L1G" 
    # API request object
    search_request = {
            #"interval": "day",
            "item_types": [item_type],
            "filter": combined_filter
            }
    #print(search_request)
    return(search_request)


def execute_search_get_json(search_request):
   search_result = \
           requests.post(
                   'https://api.planet.com/data/v1/quick-search',
                   auth=HTTPBasicAuth(PLANET_API_KEY, ''),
                   json=search_request)
   print(json.dumps(search_result.json(), indent=1))
   return(search_result.json())
   

def execute_search_get_item_ids(search_request):
   search_result = \
           requests.post(
                   'https://api.planet.com/data/v1/quick-search',
                   auth=HTTPBasicAuth(PLANET_API_KEY, ''),
                   json=search_request)
   
   #print(json.dumps(search_result.json(), indent=1))
   
   # extract image IDs only
   image_ids = [feature['id'] for feature in search_result.json()['features']]
   print(image_ids)
   print(len(image_ids))
   
   #item_type = "PSScene3Band"
   #item_type = "PSScene4Band"
   item_type = "Landsat8L1G" 
   item_ids = []
   
   for i in range(len(image_ids)):
      id0 = image_ids[i]
      id0_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, id0)
      item_ids.append(id0)
   
   return(item_ids)
 
def get_thumbs_from_search_json(search_json):
    filenames = []
    tb = [feature['_links'] for feature in search_json['features']]
    image_ids = [feature['id'] for feature in search_json['features']]
    for i in range(len(tb)):
        tb_url = search_json['features'][i]["_links"]['thumbnail']
        filestring = str(image_ids[i])+".jpg"
        if os.path.exists(os.path.dirname(filestring)) != True:
            r = requests.get(tb_url, auth=HTTPBasicAuth(PLANET_API_KEY, ''),allow_redirects=True)
            filenames.append(filestring)
            open(filestring, 'wb').write(r.content)
            time.sleep(5)
    return(filenames)

def get_item_ids_from_search_json(search_json):
   print(search_json)
   image_ids = [feature['id'] for feature in search_json['features']]
   #item_type = "PSScene3Band"
   #item_type = "PSScene4Band"
   item_type = "Landsat8L1G" 
   item_ids = []
   for i in range(len(image_ids)):
      id0 = image_ids[i]
      id0_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, id0)
      item_ids.append(id0)
   return(item_ids) 
   
#The Data API does not pre-generate assets, so they are not always immediately availiable to download.
#In order to download an asset, we first have to activate it.
   
def construct_requestv2(name,aoi,item_ids):
    print("construct_requestv2")
    req = {
    "name": name,
      "products": [
        {
          "item_ids": item_ids,
          "item_type": "Landsat8L1G",
          "product_bundle": "analytic"
        }
      ],
      "tools": [
        {
          "clip": {
            "aoi": {
                "type": "Polygon",
                "coordinates": aoi 
            }
          }
        }
      ]
    }
    print(req)
    return(req)

def requestv2(req):
    #This is where we pass the original request:
    search_result = \
                 requests.post(
                     'https://api.planet.com/compute/ops/orders/v2',
                     auth=HTTPBasicAuth(PLANET_API_KEY, ''),
                     json=req)
    print(search_result)
    print(json.dumps(search_result.json(), indent=1))
    return(search_result)


def get_requestv2_results():
#### This is where we get the results from the original request:
### this will always try to pull the most recent order
    hp = 'https://api.planet.com/compute/ops/orders/v2'
    result = \
             requests.get(
                 hp,
                 auth=HTTPBasicAuth(PLANET_API_KEY, '')
             )
    return(result)


# Will return a list of loc_ids, iterate over each using get_download_url below:
def extract_location_ids_v1(result):
    loc_ids = [order['_links'] for order in result.json()['orders']]
    loc_ids_self = loc_ids[0]["_self"]
    return(loc_ids)

def extract_location_ids(r2js,name):
    loc_ids = []
    for order in r2js['orders']:
        if(order['name']==name):
            loc_ids.append(order["_links"]["_self"])
    return(loc_ids)

# We now perform this tasdk in perform_download    
def get_download_url(loc_ids_self):
#### This is where we get the URLS for downloading files from the original request:
    result = \
             requests.get(
                 loc_ids_self,
                 auth=HTTPBasicAuth(PLANET_API_KEY, '')
             )
    #print(result)
    #print(json.dumps(result.json(), indent=1))
    dres = result.json()['_links']
    #dres = spec_result['_links']
    act_ids = [order['location'] for order in dres['results']]
    return(act_ids)
    #print(act_ids)
    #names = [order['name'] for order in dres['results']]
    #print(names)
    

def perform_download(order_name,loc_ids_self):
#### This is where we get the URLS for downloading files from the original request:
    result = \
             requests.get(
                 loc_ids_self,
                 auth=HTTPBasicAuth(PLANET_API_KEY, '')
             )
    if(result.status_code == 429):
        time.sleep(5)
    
        result = \
                 requests.get(
                     loc_ids_self,
                     auth=HTTPBasicAuth(PLANET_API_KEY, '')
                 )
        #print(json.dumps(result.json(), indent=1))
    
    #print(result.status.code)
    #print(json.dumps(result.json(), indent=1))
    dres = result.json()['_links']
    #dres = spec_result['_links']
    act_ids = [order['location'] for order in dres['results']]
    #print(act_ids)
    
    names = [order['name'] for order in dres['results']]
    #print(names)
    
    ####  FOR DOWNLOAD
    ix = 0
    for i in range(len(names)):
        order_dirname = f'{order_name}/{names[i]}'
#        print("Name",i,names[i])
        if os.path.exists(order_dirname) != True:  # Let's not download if we already have it
           if os.path.exists(os.path.dirname(order_dirname)) != True: 
               os.makedirs(os.path.dirname(order_dirname))
           print("Downloading",order_dirname)
           r = requests.get(act_ids[i], auth=HTTPBasicAuth(PLANET_API_KEY, ''),allow_redirects=True)
           if(r.status_code == 200):
               #open(names[i], 'wb').write(r.content)
               open(order_dirname, 'wb').write(r.content)
           elif(r.status_code == 429):
               time.sleep(5)
               r = requests.get(act_ids[i], auth=HTTPBasicAuth(PLANET_API_KEY, ''),allow_redirects=True)
               #open(names[i], 'wb').write(r.content)
               open(order_dirname, 'wb').write(r.content)
           else:
               print("Encountered unexpected error code:",r.status_code)
    return(names)


def lldist(lat1,lon1,lat2,lon2):
    geom = [Point(lon1,lat1),Point(lon2,lat2)]
    gdf=geopandas.GeoDataFrame(geometry=geom,crs={'init':'epsg:4326'})
    epsg_for_utm = 3112 # Using: https://epsg.io/ & http://www.dmap.co.uk/utmworld.htm & https://spatialreference.org/ref/epsg/wgs-84-utm-zone-56s/
    #gdf.to_crs(epsg=epsg_for_utm,inplace=True)
    gdf.to_crs(epsg=3112,inplace=True)
    l=gdf.distance(gdf.shift())
    return(l)

def estimate_quota_usage(search_json):
    image_ids = [feature['id'] for feature in search_json['features']]

def display_thumbs(filenames,search_json):
    ac = [feature['properties'] for feature in search_json['features']]
    ims = []
    for i in range(len(filenames)):
        print(ac[i]['acquired'])
        ims.append(Image(filename=filenames[i]))
    display(*ims)

def resume_and_sync():
    resob = get_requestv2_results()
    rjs = resob.json()
    now = datetime.datetime.now()
    complete = 1
    orders = rjs['orders']
    completed_list = []
    for i in range(len(orders)):
        tt = rjs["orders"][i]["created_on"]
        tid = rjs["orders"][i]["id"]
        ttx = str.split(tt,'T')[0]
        xd = datetime.datetime.strptime(ttx, '%Y-%m-%d')
        xdd = now - xd
        state = rjs["orders"][i]["state"] 
        if(state == 'success'):
            cname = rjs["orders"][i]["name"]
            print(rjs["orders"][i]["name"])
            if(os.path.exists(cname)):
                print(" check subdirs")
            else:
                print("who503")
                #requestv2_json = rjs#get_requestv2_results()
                loc_ids = extract_location_ids(rjs,cname)
                r = re.compile('MS_clip.tif')
                for i in range(len(loc_ids)):
                   dfnames = perform_download(cname,loc_ids[i])
                   tflist = []
                   get_remaining_quota()
                   for fn in dfnames:
                      if r.search(fn):
                          print("Acquired image: ",fn)
        




def get_list_of_completed_orders(rjs):
    now = datetime.datetime.now()
    complete = 1
    orders = rjs['orders']
    completed_list = []
    for i in range(len(orders)):
        tt = rjs["orders"][i]["created_on"]
        tid = rjs["orders"][i]["id"]
        ttx = str.split(tt,'T')[0]
        xd = datetime.datetime.strptime(ttx, '%Y-%m-%d')
        xdd = now - xd
        state = rjs["orders"][i]["state"] 
        if(state == 'running'):
            complete = 0
            lm = rjs["orders"][i]["last_message"]
            print(lm)
        if(xdd.days < 2 and state != 'running'):
            lm = rjs["orders"][i]["last_message"]
            #print(tid,tt,lm)
            completed_list.append(tid)
            if(lm != 'Delivery completed'):
                complete = 0
    return(completed_list)

def check_results_for_completion(rjs):
    now = datetime.datetime.now()
    complete = 1
    orders = rjs['orders']
    for i in range(len(orders)):
        tt = rjs["orders"][i]["created_on"]
        tid = rjs["orders"][i]["id"]
        ttx = str.split(tt,'T')[0]
        xd = datetime.datetime.strptime(ttx, '%Y-%m-%d')
        xdd = now - xd
        state = rjs["orders"][i]["state"] 
        if(state == 'running'):
            complete = 0
            if "last_message" in rjs["orders"][i]:
                lm = rjs["orders"][i]["last_message"]
                print("562:",lm)
                if(lm == 'Delivery completed'):
                    complete = 1
#        if(xdd.days < 2 and state != 'running'):
#            lm = rjs["orders"][i]["last_message"]
#            print(tid,tt,lm)
#            if(lm != 'Delivery completed'):
#                complete = 0
    return(complete)
        

if __name__ == "__main__":
    print("Launching Order_API")
    
    print("Loading Planet API key from key.txt")
    with open('key.txt', 'r') as file:
        PLANET_API_KEY = file.read().replace('\n', '')

    locfile = sys.argv[1]
    if locfile == "resume":
        resume_and_sync()
        exit()
    if locfile == "quota":
        get_remaining_quota()
        exit()

    observation_date = sys.argv[2] # 'DD-Mon-YY'
    observation_date_end = sys.argv[3] # 'DD-Mon-YY'
    name = sys.argv[4]
    print("Checking quota:")
    #get_remaining_quota()
    print("Search request is named:",name)
    print("Searching for images from ",observation_date," to ",observation_date_end)
    print("from locations in",locfile)

    print(locfile)
    regcsv = re.compile(r'.csv')
    regjson = re.compile(r'.json')

    regcsvm = regcsv.search(locfile)
    regjsonm = regjson.search(locfile)

    df = pd.DataFrame()
    geojson_list = []

# if locfile is csv:
    if(regcsvm):
        df = pd.read_csv(locfile)
        geojson_list = generate_geojson_from_coords_df(df,0.05) 
        exsqkm = calculate_sqkm(df,0.05)
    # locfile is json:
    if(regjson):
        myj = json.loads(open(locfile).read())
        print(myj)
        geojson_list.append(myj)
        exsqkm = 0.0 # unknown 
    
    dfwdd_rows = []
    gtodcounter = 0
    for g in geojson_list:
        odfr = {} # df.iloc[gtodcounter].to_dict()
        gtodcounter += 1
        gj = myj#json.loads(g)
        print(gj)
        gjc = gj['features'][0]['geometry']#['coordinates']
        #print(gjc)
        #search_request = construct_search_request(gj,observation_date,observation_date_end)
        search_request = construct_search_request(gjc,observation_date,observation_date_end)
        print("search_request")
        print(search_request)
        search_json = execute_search_get_json(search_request)
        print(search_json)
        print("search_json")
        item_ids = get_item_ids_from_search_json(search_json)
        #item_ids = [item_ids[0]]
        if( len(item_ids) < 1 ):
            print("No items found for this location")
            exit()
        print(item_ids) 
        #exit()
        #gquota = get_remaining_quota()
        #print("Expected to consume ",exsqkm*len(item_ids)," square kilometers of ",gquota," square kilometers remaining")
        
        #gj = json.loads(g)
        #gjc = gj['coordinates']
        gjcx = gj['features'][0]['geometry']['coordinates']
        order_request = construct_requestv2(name,gjcx,item_ids)
        print(order_request)

        print("ordering")
        request_result = requestv2(order_request) # Actually activate new data
        while request_result.status_code == 400 and len(item_ids) > 0:
            rrj = request_result.json()
            msg = rrj['field']['Details'][0]['message']
            print(msg)
            regx = re.compile(r'4Band/.+')
            itrem = regx.search(msg)
            trm = re.sub('4Band/','',itrem.group(0))
            if trm in item_ids: item_ids.remove(trm)
            order_request = construct_requestv2(name,gjc,item_ids)
            print(order_request)
            request_result = requestv2(order_request) # Actually activate new data



        requestv2_json = get_requestv2_results()
        complete_flag = check_results_for_completion(requestv2_json.json())
        waittimestart = time.time()
        while (complete_flag != 1):
            time.sleep(30)
            print(f"Seconds elapsed waiting for activation of requested imagery: {time.time()-waittimestart}")
            requestv2_json = get_requestv2_results()
            complete_flag = check_results_for_completion(requestv2_json.json())
        
        loc_ids = extract_location_ids(requestv2_json.json(),name)
        r = re.compile('MS_clip.tif')
        for i in range(len(loc_ids)):
           dfnames = perform_download(name,loc_ids[i])
           tflist = []
           get_remaining_quota()
           for fn in dfnames:
              if r.search(fn):
                  print("Acquired image: ",fn)
                  dfr = odfr.copy()
                  dfr["filename"] = fn
                  dfwdd_rows.append(dfr)
                  #fb = load_image4(fn)
                  #print(len(fb))
                  #plot_bands4(fb,title=fn)          
                  #t = calc_nvdi(fb)
                  #plt.set_cmap('jet')
                  #plt.imshow(t)
                  #plt.show()
    
    finaldf = pd.DataFrame(dfwdd_rows)
    finaldf.to_csv(f"{name}_storage_paths.csv")
