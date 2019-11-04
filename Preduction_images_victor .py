# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:00:00 2019

@author: Victor
"""
import os
import sys
import numpy as np
import skimage.io

from pycocotools.coco import COCO
from pycocotools.cocoeval   import COCOeval
from pycocotools import mask as maskUtils

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
#from mrcnn import visualize

import zipfile
import urllib.request
import shutil
import glob
import tqdm


from skimage import measure
from osgeo import gdal
from shapely.geometry import Polygon
import geopandas as gpd 

ROOT_DIR = os.getcwd()

#Import Mask RCNN
sys.path.append(ROOT_DIR) #To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

def mask2shp(p, fn):
    seg = []
    contours = measure.find_contours(p['masks'].astype(np.uint8)[:, :, _idx], 0.5)
    for contour in contours:
        contour = np.flip(contour, axis = 1)
        segmentation = contour.tolist()
        seg.append(segmentation)
        
    raster = gdal.Open(fn) 
    geotransform = raster.GetGeoTransform()
    raster = None
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    _polygon = []
    for mask_num in range(len(seg)):
        mask_list = seg[mask_num]
        points = []
        for idx, val in enumerate(mask_list):
            point = (geotransform[0] + val[0] * pixel_width,
                     geotransform[3] + val[1] * pixel_height)
            points.append(point)
        _polygon.append(Polygon(points))
    return _polygon

#PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "pretrained_weights.h5")
#PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "mask_rcnn_crowdai-mapping-challenge_0160.h5")
PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "mask_rcnn_crowdai-mapping-challenge_0160_crowdai.h5")
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


# Gather all TIF files in the test set as small batches
files = glob.glob(os.path.join(IMAGE_DIR, "*.TIF"))
ALL_FILES = []
_buffer = []
for _idx, _file in enumerate(files):
    print(_file, len(ALL_FILES))
    if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
        ALL_FILES.append(_buffer)
        _buffer = []
        _buffer.append(_file)
    else:
        _buffer.append(_file)
    
if len(_buffer) > 0:
    ALL_FILES.append(_buffer)

# Iterate over all the batches and predict
_final_object = []
shp_li = []

for _files in tqdm.tqdm(ALL_FILES):
    images = [skimage.io.imread(x) for x in _files]
    predictions = model.detect(images, verbose = 0)
    for _idx, r in enumerate(predictions):
        _file = _files[_idx]
        image_fn = _file.split('\\')[-1]
        image_id = image_fn.replace(".TIF", "")

        for _idx, class_id in enumerate(r["class_ids"]):
            if class_id == 1:
                mask = r["masks"].astype(np.uint8)[:, :, _idx]
                bbox = np.around(r["rois"][_idx], 1)
                bbox = [float(x) for x in bbox]
                _result = {}
                _result["image_id"] = image_id
                _result["category_id"] = 100
                _result["score"] = float(r["scores"][_idx])
                _mask = maskUtils.encode(np.asfortranarray(mask))
                _mask["counts"] = _mask["counts"].decode("UTF-8")
                _result["segmentation"] = _mask
                _result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                _final_object.append(_result)
                
                # mask to shapefile 
                poly_list = mask2shp(r, _file)
                shp_li = shp_li + poly_list
                
#Export shp                
shp = gpd.GeoDataFrame(shp_li, columns = ['geometry'])
#shp.crs = raster.GetProjection()
SHP_DIR = IMAGE_DIR.replace("images", "shapefiles")
SHP_DIR = os.path.join(SHP_DIR, "predict.shp")
shp.to_file(SHP_DIR)
            
#Export json
"""
fp = open("predictions.json", "w")
import json
fp.write(json.dumps(_final_object))
fp.close()
"""             


