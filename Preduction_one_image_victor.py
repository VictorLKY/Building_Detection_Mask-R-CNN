# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:57:27 2019

@author: Victor
"""
import os
import sys
import time
import numpy as np
import skimage.io

from pycocotools.coco import COCO
from pycocotools.cocoeval   import COCOeval

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
#from mrcnn import visualize

import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random

from skimage import measure
from collections import defaultdict
from osgeo import gdal
from shapely.geometry import Polygon
import geopandas as gpd 

ROOT_DIR = os.getcwd()

#Import Mask RCNN
sys.path.append(ROOT_DIR) #To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "pretrained_weights.h5")
PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "mask_rcnn_crowdai-mapping-challenge_0160.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "data","BKK", "BKK_test", "images")

class InferenceConfig(coco.CocoConfig):
    #Set batch size to 1 since we'll be running inference on one image at a time.
    #Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # 1 Background + 1 Building
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320
#    NAME = "crowdai-mapping-challenge"
    NAME = "Victor"
config = InferenceConfig()
#config.diaplay()

model = modellib.MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = config)
model_path = PRETRAINED_MODEL_PATH

# or if you want to use the lastest trained model, you can use:
# model_path = model.find_last()[1]

model.load_weights(model_path, by_name = True)

class_names = ['BG', 'building'] # In our case, we have 1 class for the background, and 1 class for building

file_names = next(os.walk(IMAGE_DIR))[2]
fn = os.path.join(IMAGE_DIR, random.choice(file_names))
random_image = skimage.io.imread(fn)

predictions = model.detect([random_image] * config.BATCH_SIZE, verbose = 1) # We are replicating the same image to fill up the batch_size

p = predictions[0]
#visualize.display_instances(random_image, p['rois'], p['masks'], p['class_ids'], class_names, p['scores'])

#annotation = {"segmentation": []}
seg = []
for _idx in range(len(p['class_ids'])):
    contours = measure.find_contours(p['masks'].astype(np.uint8)[:, :, _idx], 0.5)
    
    for contour in contours:
        contour = np.flip(contour, axis = 1)
        segmentation = contour.tolist()
#        annotation["segmentation"].append(segmentation)
        seg.append(segmentation)

# mask to shapefile
raster = gdal.Open(fn) 
geotransform = raster.GetGeoTransform()
pixel_width = geotransform[1]
pixel_height = geotransform[5]
geo_vertices = defaultdict(list)
shp_li = []
for mask_num in range(len(seg)):
    mask_list = seg[mask_num]
    poly = []
    for idx, val in enumerate(mask_list):
        point = (geotransform[0] + val[0] * pixel_width,
                 geotransform[3] + val[1] * pixel_height)
        poly.append(point)
    shp_li.append(Polygon(poly))
shp = gpd.GeoDataFrame(shp_li, columns = ['geometry'])
shp.crs = raster.GetProjection()
shp_fn = fn.replace(".TIF", "")
shp_fn = shp_fn.replace("images", "shapefiles")
shp.to_file('%s.shp'%shp_fn)
