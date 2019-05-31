# python3.6 create_report.py orderName
# can you filter on how much image overlaps polygon?
# add line from csv to every line that is generated from create_report
# more data in ndvi/ecarr images

import os
import sys
import re
import pandas as pd
import json
import rasterio
import numpy as np

from xml.dom import minidom

import matplotlib.pyplot as plt
import matplotlib.colors as colors
#Import gdal
from osgeo import gdal





"""
The reflectance values will range from 0 to 1. You want to use a diverging color scheme to visualize the data,
and you want to center the colorbar at a defined midpoint. The class below allows you to normalize the colorbar.
"""

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# FRom: https://github.com/kscottz/PythonFromSpace/blob/master/TheBasics.ipynb
def load_image4(filename):
    """Return a 4D (r, g, b, nir) numpy array with the data in the specified TIFF filename."""
    path = os.path.abspath(os.path.join('./', filename))
    if os.path.exists(path):
        with rasterio.open(path) as src:
            b, g, r, nir = src.read()
            return np.dstack([r, g, b, nir])    

def load_image4_to_RGB(filename):
    path = os.path.abspath(os.path.join('./', filename))
    if os.path.exists(path):
        with rasterio.open(path) as src:
            b, g, r, nir = src.read()
            img_4band = np.dstack([r, g, b, nir])    
            img_3band = img_4band[:,:,:3]
            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = src.profile
        
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                dtype=rasterio.uint16,
                count=3,
                compress='lzw')
        
            with rasterio.open(f"{filename}_example.tif", 'w', **profile) as dst:
                dst.write(img_3band.astype(rasterio.uint16), 1)

def load_image3(filename,fndr):
    """Return a 3D (r, g, b) numpy array with the data in the specified TIFF filename."""
    path = os.path.abspath(os.path.join('./', filename))
    if os.path.exists(path):
        with rasterio.open(path) as src:
            b,g,r,mask = src.read()
            img_3band = np.dstack([b, g, r])

            profile = src.profile
        
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                dtype=rasterio.uint16,
                count=1,
                compress='lzw')
        
            with rasterio.open(f"{fndr}/example.tif", 'w', **profile) as dst:
                dst.write(img_3band[:,:,1].astype(rasterio.uint16), 1)


def rgbir_to_rgb(img_4band):
    """Convert an RGBIR image to RGB"""
    return img_4band[:,:,:3]

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        
 
 
def getListOfMetadataFiles(dirName):
    regex = re.compile(r'\(metadata.json\)$')
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if regex.match(fullPath):
                allFiles.append(fullPath)

    #filtered = [i for i in full if not regex.match(i)]
    #return filtered        
    return allFiles


def main():
    
    dirName = sys.argv[1]#os.getcwd()
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    
    regex_tif = re.compile(r'MS_clip.tif')
    filtered_tif = [i for i in listOfFiles if regex_tif.search(i)]
    #filename = "20170623_180038_0f34_3B_AnalyticMS.tif"

#gdalinfo -mm 1155205_2017-03-31_RE3_3A_rgb.tif
#gdal_translate 1155205_2017-03-31_RE3_3A_rgb.tif 1155205_2017–03–31_RE3_3A_rgb_scaled.tif -scale 1422 49572 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB

    dateImageTuples = []
    ndvistats = []
    for filename in filtered_tif:
        #print(filename)    
        fnbn = os.path.basename(filename)
        fndn = os.path.dirname(filename)

        #print(dirName)
        #print(fndn)
        rg = re.compile(r'' + dirName + '/(.+)/(.+)/files/')
        xd = rg.search(filename)
        #print(xd)
        #print(xd[0])
        #print(xd[1])

        jsfn = re.sub('_3B_AnalyticMS_clip.tif','_metadata.json',filename)
        js = json.load(open(jsfn))

        pmid = js['id']
        acqdate = js['properties']['acquired']
        satid = js['properties']['satellite_id']
        dateImageTuples.append((acqdate,fndn,xd[1]))

    #dateImageTuples.sort()
    #mnum = 1000
    #for m in dateImageTuples:
    #    print(f"{m[0]} {m[1]}")
    #    mnum += 1
    #    os.system(f"cp {m[1]}/ndvi.png dt_ndvi_{mnum}.png")
    #print(dateImageTuples)
    #exit()

    # first pass:
    for filename in filtered_tif:
        #print(filename)    
        # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
        with rasterio.open(filename) as src:
            band_blue_radiance = src.read(1)
            
        with rasterio.open(filename) as src:
            band_green_radiance = src.read(2)
        
        with rasterio.open(filename) as src:
            band_red_radiance = src.read(3)
       
        with rasterio.open(filename) as src:
            band_nir_radiance = src.read(4)

        fnbn = os.path.basename(filename)
        fndn = os.path.dirname(filename)
       
        #show(src.read())

        rg = re.compile(r'' + dirName + '/(.+)/(.+)/files/')
        xd = rg.search(filename)
        
        jsfn = re.sub('_3B_AnalyticMS_clip.tif','_metadata.json',filename)
        js = json.load(open(jsfn))
        acqdate = js['properties']['acquired']
        satid = js['properties']['satellite_id']

        #dateImageTuples.append((acqdate,fndn))

        xmlfn = re.sub('AnalyticMS_clip.tif','AnalyticMS_metadata_clip.xml',filename)
        xmldoc = minidom.parse(xmlfn)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        
        # XML parser refers to bands by numbers 1-4
        coeffs = {}
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                i = int(bn)
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i] = float(value)
        
        #print("Conversion coefficients: {}".format(coeffs))        
        # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
        band_blue_reflectance = band_blue_radiance * coeffs[1]
        band_green_reflectance = band_green_radiance * coeffs[2]
        band_red_reflectance = band_red_radiance * coeffs[3]
        band_nir_reflectance = band_nir_radiance * coeffs[4]
        
        print("Red band radiance is from {} to {}".format(np.amin(band_red_radiance), np.amax(band_red_radiance)))
        print("Red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
        #Set spatial characteristics of the output object to mirror the input
        kwargs = src.meta
        kwargs.update(
           dtype=rasterio.uint16,
           count = 4)
        
        print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance),
                                                                              np.amax(band_red_reflectance)))
        
        # Here we include a fixed scaling factor. This is common practice.
        scale = 10000
        blue_ref_scaled = scale * band_blue_reflectance
        green_ref_scaled = scale * band_green_reflectance
        red_ref_scaled = scale * band_red_reflectance
        nir_ref_scaled = scale * band_nir_reflectance
        
        print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled),
                                                                            np.amax(red_ref_scaled)))
        # NDVI 
        band_nir = band_nir_reflectance
        band_red = band_red_reflectance
        #Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)

        # Set min/max values from reflectance range for image (excluding NAN)
        min=np.nanmin(ndvi)
        max=np.nanmax(ndvi)
        mid=(max-min)/2.0# Could perhaps also be center of mass distribution?
        #mid=0.20
        ndvistats.append((min,max))
        #.set_size_inches(float(H)/float(DPI),float(W)/float(DPI)) 

        #fig = plt.figure(figsize=(20,10))
        fig = plt.figure(figsize=(1600/200,1000/200))
        ax = fig.add_subplot(111)
        #ax.figtext(-1,-2,f"{max} {mid} {min}") 

        # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        # note that appending '_r' to the color scheme name reverses it!
        #cmap = plt.cm.get_cmap('RdGy_r')
        cmap = plt.cm.get_cmap('RdYlGn')
        cax = ax.imshow(ndvi, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        ax.axis('off')
        ax.set_title(f"NDVI {acqdate} {satid}", fontsize=18, fontweight='bold')
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)#65
        #fig.savefig(f"{fndn}/ndvi.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
        fig.savefig(f"{fndn}/ndvi.png", dpi=200, bbox_inches='tight', pad_inches=0.0)
        os.system(f"cp {fndn}/ndvi.png dt_ndvi_{acqdate[0:10]}_{xd[1]}.png")

    xmin = 0.0
    xmax = 0.0
    for s in ndvistats:
        if xmax < s[1]:
            xmax = s[1]

    ndvi_stats_mid = (xmax-xmin)/2.0

    # second pass:
    nsc = 0
    for filename in filtered_tif:
        print(filename)    
        # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
        with rasterio.open(filename) as src:
            band_blue_radiance = src.read(1)
            
        with rasterio.open(filename) as src:
            band_green_radiance = src.read(2)
        
        with rasterio.open(filename) as src:
            band_red_radiance = src.read(3)
       
        with rasterio.open(filename) as src:
            band_nir_radiance = src.read(4)

        fnbn = os.path.basename(filename)
        fndn = os.path.dirname(filename)
        
        rg = re.compile(r'' + dirName + '/(.+)/(.+)/files/')
        xd = rg.search(filename)
        
        jsfn = re.sub('_3B_AnalyticMS_clip.tif','_metadata.json',filename)
        js = json.load(open(jsfn))
        acqdate = js['properties']['acquired']
        satid = js['properties']['satellite_id']

        #dateImageTuples.append((acqdate,fndn))

        xmlfn = re.sub('AnalyticMS_clip.tif','AnalyticMS_metadata_clip.xml',filename)
        xmldoc = minidom.parse(xmlfn)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        
        # XML parser refers to bands by numbers 1-4
        coeffs = {}
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                i = int(bn)
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i] = float(value)
        
        #print("Conversion coefficients: {}".format(coeffs))        
        # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
        band_blue_reflectance = band_blue_radiance * coeffs[1]
        band_green_reflectance = band_green_radiance * coeffs[2]
        band_red_reflectance = band_red_radiance * coeffs[3]
        band_nir_reflectance = band_nir_radiance * coeffs[4]
        
        print("Red band radiance is from {} to {}".format(np.amin(band_red_radiance), np.amax(band_red_radiance)))
        print("Red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
        #Set spatial characteristics of the output object to mirror the input
        kwargs = src.meta
        kwargs.update(
           dtype=rasterio.uint16,
           count = 4)
        
        print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance),
                                                                              np.amax(band_red_reflectance)))
        
        # Here we include a fixed scaling factor. This is common practice.
        scale = 10000
        blue_ref_scaled = scale * band_blue_reflectance
        green_ref_scaled = scale * band_green_reflectance
        red_ref_scaled = scale * band_red_reflectance
        nir_ref_scaled = scale * band_nir_reflectance
        
        print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled),
                                                                            np.amax(red_ref_scaled)))
        # NDVI 
        band_nir = band_nir_reflectance
        band_red = band_red_reflectance
        #Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)

        # Set min/max values from reflectance range for image (excluding NAN)
        min=np.nanmin(xmin)
        max=np.nanmax(xmax)
        mid=ndvi_stats_mid
        nsc += 1
        #mid=0.20
        print(nsc," ",xmin," ",xmax," ",mid)

        #.set_size_inches(float(H)/float(DPI),float(W)/float(DPI)) 

        #fig = plt.figure(figsize=(20,10))
        fig = plt.figure(figsize=(1600/200,1000/200))
        ax = fig.add_subplot(111)
        #ax.figtext(-1,-2,f"{max} {mid} {min}") 

        # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        # note that appending '_r' to the color scheme name reverses it!
        #cmap = plt.cm.get_cmap('RdGy_r')
        cmap = plt.cm.get_cmap('RdYlGn')
        cax = ax.imshow(ndvi, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        ax.axis('off')
        ax.set_title(f"NDVI {acqdate} {satid}", fontsize=18, fontweight='bold')
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)#65
        #fig.savefig(f"{fndn}/ndvi.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
        fig.savefig(f"{fndn}/ndvi_post.png", dpi=200, bbox_inches='tight', pad_inches=0.0)
        os.system(f"cp {fndn}/ndvi_post.png dt_ndvi_{acqdate[0:10]}_{xd[1]}.png")
        os.system(f"convert dt_ndvi_{acqdate[0:10]}_{xd[1]}.png -gravity center -background white -extent 1200x1000 dt_m_ndvi_{acqdate[0:10]}_{xd[1]}.png")
        
        ## ECARR RED 
        #ecarrRed = 0.0161 * ((band_red_reflectance.astype(float) / (band_green_reflectance.astype(float) * band_red_reflectance.astype(float) ) ** 0.7784))
        ## Set min/max values from reflectance range for image (excluding NAN)
        #min=np.nanmin(ecarrRed)
        #max=np.nanmax(ecarrRed)
        #mid=(max-min)/2.0# Could perhaps also be center of mass distribution?
        #
        #fig = plt.figure(figsize=(20,10))
        #ax = fig.add_subplot(111)
        #
        ## diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        ## note that appending '_r' to the color scheme name reverses it!
        ##cmap = plt.cm.get_cmap('RdGy_r')
        #cax = ax.imshow(ecarrRed, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        #ax.axis('off')
        #ax.set_title(f"ECARR_red {acqdate} {satid}", fontsize=18, fontweight='bold')
        #cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)
        #fig.savefig(f"{fndn}/ecarr_red.png", dpi=200, bbox_inches='tight', pad_inches=0.7)

       ## ECARR NIR 
        #ecarrnir = 0.0161 * ((band_nir_reflectance.astype(float) / (band_green_reflectance.astype(float) * band_nir_reflectance.astype(float) ) ** 0.7784))
        ## Set min/max values from reflectance range for image (excluding NAN)
        #min=np.nanmin(ecarrnir)
        #max=np.nanmax(ecarrnir)
        #mid=(max-min)/2.0# Could perhaps also be center of mass distribution?
        ##mid=0.20
        #
        #fig = plt.figure(figsize=(20,10))
        #ax = fig.add_subplot(111)
        #
        ## diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        ## note that appending '_r' to the color scheme name reverses it!
        ##cmap = plt.cm.get_cmap('RdGy_r')
        #cax = ax.imshow(ecarrnir, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        #ax.axis('off')
        #ax.set_title(f"ECARR_nir {acqdate} {satid}", fontsize=18, fontweight='bold')
        #cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)
        #fig.savefig(f"{fndn}/ecarr_nir.png", dpi=200, bbox_inches='tight', pad_inches=0.7)





        # Write band calculations to a new raster file
        #print(f"in {fndn}:")
        #print(f"about to write: {fndn}/rgb.png")
        #with rasterio.open(f"{fndn}/rgb.png", 'w', **kwargs) as dst:
        #        #dst.write_band(1, band_red_reflectance.astype(rasterio.uint16))
        #        #dst.write_band(2, band_green_reflectance.astype(rasterio.uint16))
        #        #dst.write_band(3, band_blue_reflectance.astype(rasterio.uint16))
        #        dst.write_band(4, band_nir_reflectance.astype(rasterio.uint16))
        #        dst.write_band(3, band_red_reflectance.astype(rasterio.uint16))
        #        dst.write_band(2, band_green_reflectance.astype(rasterio.uint16))
        #        dst.write_band(1, band_blue_reflectance.astype(rasterio.uint16))
        #        ##dst.write_band(4, band_nir_reflectance.astype(rasterio.uint16))
        ## Set min/max values from reflectance range for image (excluding NAN)
        min=np.nanmin(band_nir_reflectance)
        max=np.nanmax(band_nir_reflectance)
        mid=(max-min)/2.0# Could perhaps also be center of mass distribution?
        #mid=0.20
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        
        # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        # note that appending '_r' to the color scheme name reverses it!
        cmap = plt.cm.get_cmap('RdGy_r')
        cax = ax.imshow(band_nir_reflectance, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        ax.axis('off')
        ax.set_title('NIR Reflectance', fontsize=18, fontweight='bold')
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)
        fig.savefig(f"{fndn}/nir.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
        
        #from osgeo import gdal
        print(f"About to open {filename}")
        #load_image4_to_RGB(filename)
        load_image3(filename,fndn)
        #gtif = gdal.Open(filename)
        #bmin = 0
        #bmax = 1000000

        #for band in range(1,4):
        #    print(band)
        #    srcband = gtif.GetRasterBand(band)
        #    #srcband.ComputeStatistics(0)
        #    bst = srcband.ComputeRasterMinMax()
        #    tbmin = bst[0] 
        #    tbmax = bst[1] 
        #    if tbmin < bmin:
        #        bmin = tbmin
        #    if tbmax > bmax:
        #        bmax = tbmax
    
        #for band in range(1,4):
        #    print(band)
        #    srcband = gtif.GetRasterBand(band)
        #    sgtif = gdal.Translate(f"{fndn}/output{band}.tif", srcband, options = f"-scale {bmin} {bmax} 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB")
        #    sgtif = None

        #gtif = gdal.Translate(f"{fndn}/output.tif", gtif, scaleParams = [bmin,bmax,0,65535], exponents=[0.5])
        #gtif = gdal.Translate(f"{fndn}/output.tif", gtif, scaleParams = [bmin,bmax,0,65535])
        #gtif = gdal.Translate(f"{fndn}/output.tif", gtif, options = f"-scale {bmin} {bmax} 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB")
        #gtif = None
        
        
        ## Write band calculations to a new raster file
        #print(f"in {fndn}:")
        #print(f"about to write: {fndn}/rgb.png")
        #with rasterio.open(f"{fndn}/rgb.png", 'w', **kwargs) as dst:
        #        #dst.write_band(1, band_red_reflectance.astype(rasterio.uint16))
        #        #dst.write_band(2, band_green_reflectance.astype(rasterio.uint16))
        #        #dst.write_band(3, band_blue_reflectance.astype(rasterio.uint16))
        #        dst.write_band(4, band_nir_reflectance.astype(rasterio.uint16))
        #        dst.write_band(3, band_red_reflectance.astype(rasterio.uint16))
        #        dst.write_band(2, band_green_reflectance.astype(rasterio.uint16))
        #        dst.write_band(1, band_blue_reflectance.astype(rasterio.uint16))
        #        ##dst.write_band(4, band_nir_reflectance.astype(rasterio.uint16))
        
    regex = re.compile(r'_metadata.json')
    filtered = [i for i in listOfFiles if regex.search(i)]

    rowlist = []
    # #print the files    
    for elem in filtered:
        print(elem)    
        dict1 = json.load(open(elem))
        dict1['filepath'] = elem
        for p in dict1['properties']:
            dict1[p] = dict1['properties'][p]
        dict1.pop('properties',None)
        rowlist.append(dict1)
        
    df = pd.DataFrame(rowlist)
    df.to_csv(f"{sys.argv[1]}.csv")

    dateImageTuples.sort()
    mnum = 1000
    for m in dateImageTuples:
        print(f"{m[0]} {m[1]}")
        mnum += 1
        os.system(f"cp {m[1]}/ndvi.png dt_{m[2]}_ndvi_{mnum}.png")

        
if __name__ == '__main__':
    main()
