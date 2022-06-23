# this script count the result based on detectron2 inference
#
import argparse
import os
import random
import time
import warnings
import pickle
import numpy as np
import math
import sys
import copy
import re
import pandas as pd
import json
import cv2
from shapely.geometry import Polygon

from detectron2.structures import BoxMode
from detectron2.structures.masks import polygons_to_bitmask
#
def get_amseg_dicts(img_dir,classes):
    anno_file=os.path.join(img_dir,"regiondata.csv")
    annotab=pd.read_csv(anno_file,delimiter="\t")
    files=annotab['filename'].unique()
    dataset_dicts=[]
    for idx,file in enumerate(files):
        record={}
        filename=os.path.join(img_dir,file)
        height,width=cv2.imread(filename).shape[:2]
        record["file_name"]=filename
        record["image_id"]=idx
        record["height"]=height
        record["width"]=width
        subtab=annotab[annotab['filename']==file]
        objs=[]
        for anno_i in range(subtab.shape[0]):#multiple masks/boxes
            tab_rec=subtab.iloc[anno_i]
            # assert not tab_rec["region_attributes"]#check it is []
            anno=json.loads(tab_rec["region_shape_attributes"])
            if len(anno)==0:
                continue
            
            px=anno["all_points_x"]
            py=anno["all_points_y"]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            category_id=np.where([ele==tab_rec['region_attributes'] for ele in classes])[0][0]
            obj={
                "bbox":[np.min(px),np.min(py),np.max(px),np.max(py)],
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id":category_id,
            }
            objs.append(obj)
        
        record["annotations"]=objs
        dataset_dicts.append(record)
    
    return dataset_dicts, files

datapath='./'#./
# inference data statistics
restab=pd.read_csv(datapath+"segmentation.txt",delimiter=",")
with open(datapath+'masks.pickle','rb') as handle:
    masklist=pickle.load(handle)

classlist=restab['annotations'].unique()
#
classlist_col=[]
#
areasum_list_col=[]
numseg_list_col=[]
numimgexi_list_col=[]
#
for classele in classlist:
    clasind=restab['annotations']==classele
    classele
    sumind=clasind
    subtab=restab[sumind]
    # areas sum
    rowind=np.where(sumind.values)[0]
    sumarea=0
    for fileele in subtab['filename'].unique():
        fileind=np.where((subtab['filename']==fileele).values)[0]
        ind_1file=fileind[0]
        fullmask=np.full((subtab['height'].iloc[ind_1file],subtab['width'].iloc[ind_1file]),False)
        for indhere in fileind:
            locmasknpup=np.unpackbits(masklist[rowind[indhere]]).reshape(fullmask.shape).view(np.bool)
            fullmask=fullmask | locmasknpup
        sumarea=sumarea+fullmask.sum()
    areasum_list_col.append(sumarea)
    # number of segmentation
    numseg_list_col.append(subtab.shape[0])
    # number of images
    numimgexi_list_col.append(len(subtab['filename'].unique()))
    #
    classlist_col.append(classele)

stat_count_tab=pd.DataFrame({
    'class': classlist_col,
    'sum_area': areasum_list_col,
    'number_segments': numseg_list_col,
    'number_images': numimgexi_list_col})
zerocond=(stat_count_tab['sum_area']==0) & (stat_count_tab['number_segments']==0) & (stat_count_tab['number_images']==0)
stat_count_tab=stat_count_tab.drop(stat_count_tab[zerocond].index)
stat_count_tab.to_csv("stat_summary_inference.txt",index=False)

# ground truth statistics
classes=['root','AMF internal hypha','AMF external hypha','AMF arbuscule','AMF vesicle','AMF spore','others']
dataset_dicts, imagfiles=get_amseg_dicts("./data/test",classes)#
# convert the dictionary into the segmentation result table
classlist=[]
polygonlist=[]
mergpolygonlist=[]
hlist=[]
wlist=[]
namelist=[]
arealist=[]
idlist=[]
for fileind,d in enumerate(dataset_dicts):
    print(d["file_name"])
    clasind=[ele['category_id'] for ele in d['annotations']]
    for segi in range(len(clasind)):
        namelist.append(d["file_name"])
        idlist.append(fileind)
        hlist.append(d['height'])
        wlist.append(d['width'])
        classlist.append(classes[clasind[segi]])
        #
        mergpolygon=d['annotations'][segi]['segmentation'][0]
        all_points_x=mergpolygon[::2]
        all_points_y=mergpolygon[1::2]
        polygonlist.append(json.dumps({"all_points_x": (all_points_x),"all_points_y": (all_points_y)}))
        mergpolygonlist.append(mergpolygon)
        #
        pgon=Polygon(zip(all_points_x,all_points_y))
        arealist.append(pgon.area)

rectab=pd.DataFrame({
    'filename': namelist,
    'id':idlist,
    'height': hlist,
    'width': wlist,
    'annotations': classlist,
    'segmentation': polygonlist,
    'segmentation_merg': mergpolygonlist,
    'area': arealist})
#
classlist_col=[]
#
areasum_list_col=[]
numseg_list_col=[]
numimgexi_list_col=[]
#
for classele in classes:
    clasind=rectab['annotations']==classele
    classele
    sumind=clasind
    subtab=rectab[sumind]
    # areas sum
    sumarea=0
    for fileele in subtab['filename'].unique():
        fileind=np.where((subtab['filename']==fileele).values)[0]
        ind_1file=fileind[0]
        heighthere=subtab['height'].iloc[ind_1file]
        widthhere=subtab['width'].iloc[ind_1file]
        fullmask=np.full((heighthere,widthhere),False)
        for indhere in fileind:
            locmasknpup=polygons_to_bitmask([subtab['segmentation_merg'].iloc[indhere]],heighthere,widthhere)
            fullmask=fullmask | locmasknpup
        sumarea=sumarea+fullmask.sum()
    areasum_list_col.append(sumarea)
    # number of segmentation
    numseg_list_col.append(subtab.shape[0])
    # number of images
    numimgexi_list_col.append(len(subtab['filename'].unique()))
    #
    classlist_col.append(classele)

stat_count_tab=pd.DataFrame({
    'class': classlist_col,
    'sum_area': areasum_list_col,
    'number_segments': numseg_list_col,
    'number_images': numimgexi_list_col})
zerocond=(stat_count_tab['sum_area']==0) & (stat_count_tab['number_segments']==0) & (stat_count_tab['number_images']==0)
stat_count_tab=stat_count_tab.drop(stat_count_tab[zerocond].index)
stat_count_tab.to_csv("stat_summary_groundtruth.txt",index=False)

# stat_count_tab[stat_count_tab['class']=='root']['number_images'].sum()
# len(restab['filename'].unique())
# stat_count_tab['number_segments'].sum()
# restab.shape
