# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:22:55 2019

@author: Victor
"""
import os
import sys
import glob
import json
import cv2
import geopandas as gpd
from osgeo import gdal
import numpy as np

def coor_t(ary, ori):
    return abs(ary - ori)//0.5

def p_b(fn, en):
    raster = gdal.Open(fn)
    geotransform = raster.GetGeoTransform()
    x = geotransform[0] 
    y = geotransform[3]
    e, n = en.exterior.coords.xy
    e_ary = np.array(e)
    n_ary = np.array(n)
    e_ary = coor_t(e_ary, x)
    n_ary = coor_t(n_ary, y)
    seg = list(np.vstack((e_ary, n_ary)).T.flatten())
    
    bound = np.array(en.bounds)
    bound_x = coor_t(bound[::2], x)
    bound_y = coor_t(bound[1::2], y)
    box = [min(bound_x), max(bound_y), 
           max(seg[::2])-min(seg[::2]), max(seg[1::2])-min(seg[1::2])]
    raster = None
    return seg, box
    

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

IMAGE_DIR = os.path.join(ROOT_DIR, "data", "BKK", "BKK_train", "images")
#MASK_DIR = os.path.join(ROOT_DIR, "data", "BKK", "BKK_train", "mask")
SHP_DIR = os.path.join(ROOT_DIR, "data", "BKK", "BKK_train", "shp", "BKK_train_Intersect.shp")
json_fn = os.path.join(ROOT_DIR, "data", "BKK", "BKK_train", "annotation.json")

image_files = glob.glob(os.path.join(IMAGE_DIR, "*.TIF"))

#info
info = { "contributor" : "Victor",
         "about" : "NatCatDAX",
         "image_captured" : "27/11/2018",
         "description" : "MASK R-CNN of building footprint detection", 
         "url" : "https://www.natcatdax.org/", 
         "version" : "1.1",
         "year" : 2019}

annotation = {"info" : info}
annotation['categories'] = [{"id": 100, "name": "building", "supercategory": "building"}]

#images
images = []
for file in image_files:
    #bgr = cv2.imread(file)
    #image_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #height, width, band = image_array.shape

    raster = gdal.Open(file)
    band = raster.GetRasterBand(1)
    image = band.ReadAsArray()
    height, width = image.shape
    raster = None
    band = None
    
    fn = file.split('\\')[-1]
    _id = int(fn.replace(".TIF","").split('_')[-1])
    images.append({"id" : _id, "file_name": fn, "width": width, "height": height})
    
annotation['images'] = images

# annotations (shapefile)
shp = gpd.read_file(SHP_DIR)
annotation['annotations'] = []
for idx, row in shp.iterrows():
    polygon, bbox = p_b(IMAGE_DIR + "\\" + row['image_fn'], row['geometry'])
    area = row['geometry'].area // 0.25
    _fn = int(row['image_fn'].replace(".TIF","").split('_')[-1])
    anno_dic = {"id":idx, "image_id":_fn, "segmentation":[polygon], "area":area, "bbox":bbox, 
                "category_id":100, "iscrowd": 0}
    annotation['annotations'].append(anno_dic)
    

#Save to .json
with open(json_fn, 'w') as fp:
    json.dump(annotation, fp)