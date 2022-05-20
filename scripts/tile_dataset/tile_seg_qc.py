# this script quality check the segmentation for tiles
# Mask-RCNN on detectron2:
# the folder structure should be:
#   data/train, val (images)
#   pretrained: the pretrained model model_best.pth
#   qc: quality check
#   ini_infer: inference with no training on the data
#   training: training on the new dataset
# Running:
#   module load Detectron2/0.3-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
#
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import math
import sys
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2
from itertools import compress
from shapely.geometry import Polygon
import matplotlib.colors as mplc

import torch
import torch.nn as nn
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer,HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,GenericMask
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader,DatasetMapper,build_detection_train_loader,MetadataCatalog,DatasetCatalog
import detectron2.data.transforms as T
import detectron2.utils.comm as comm

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
            # anno=json.loads(tab_rec["region_shape_attributes"])
            anno=json.loads(tab_rec["segmentation"])
            if len(anno)==0:
                continue
            
            px=anno[::2]
            py=anno[1::2]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            category_id=np.where([ele==tab_rec['category_id'] for ele in classes])[0][0]
            obj={
                "bbox":[np.min(px),np.min(py),np.max(px),np.max(py)],
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id":category_id,
            }
            objs.append(obj)
        
        record["annotations"]=objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def get_amseg_dicts_raw(img_dir,classes):
    anno_file=os.path.join(img_dir,"regiondata.csv")
    annotab=pd.read_csv(anno_file,delimiter=",")
    files=annotab['filename'].unique()
    files=files[files!='undefined']
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
            try:
                classatt=json.loads(tab_rec['region_attributes'])
            except :
                continue
            
            if len(anno)==0 or len(classatt)==0:
                continue
            
            classatt=classatt['object']
            px=anno["all_points_x"]
            py=anno["all_points_y"]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            matchmask=[ele==classatt for ele in classes]
            if not any(matchmask):
                continue
            category_id=np.where(matchmask)[0][0]
            obj={
                "bbox":[np.min(px),np.min(py),np.max(px),np.max(py)],
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id":category_id,
            }
            objs.append(obj)
        
        record["annotations"]=objs
        dataset_dicts.append(record)
    
    return dataset_dicts

class Visualizer_font(Visualizer):
    def draw_text(self,text,position,*,font_size=None,color="g",horizontal_alignment="center",rotation=0,):
        if not font_size:
            font_size=self._default_font_size
        font_size=10#set the font size
        # since the text background is dark, we don't want the text to be dark
        color=np.maximum(list(mplc.to_rgb(color)),0.2)
        color[np.argmax(color)]=max(0.8,np.max(color))
        
        x,y=position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size*self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

projdir='/scratch/yw44924/amf_tile/'
tilingdir=projdir+'tiling/'
outputdir=projdir+'qc/'
classes=['root','AMF internal hypha','AMF external hypha','AMF arbuscule','AMF vesicle','AMF spore','others']
for direc in [tilingdir]:
    DatasetCatalog.register("am_qc",lambda direc=direc: get_amseg_dicts(direc,classes))
    MetadataCatalog.get("am_qc").set(thing_classes=classes)#classes name list

# test data set
am_metadata_val=MetadataCatalog.get("am_qc")
dataset_dicts=get_amseg_dicts(tilingdir,classes)#
#
# random viszualize of 10 images
imageset=random.sample(dataset_dicts,20)
for d in imageset:
    im=cv2.imread(d["file_name"])
    idimagefile=re.search(r"\d+_\d+_\d+_\d+_\d+\.",d["file_name"]).group()
    # ground truth
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_val,
                   scale=2,
                   instance_mode=ColorMode.IMAGE_BW
    )
    out=v.draw_dataset_dict(d)
    cv2.imwrite(outputdir+'showimage.exp.groundtruth'+str(d['image_id'])+'_'+idimagefile+'jpg',out.get_image()[:, :, ::-1])

# raw images (large)
for direc in ['preprocess']:
    DatasetCatalog.register("am_"+direc,lambda direc=projdir+direc: get_amseg_dicts_raw(tilingdir,classes))
    MetadataCatalog.get("am_"+direc).set(thing_classes=classes)#classes name list

am_metadata_raw=MetadataCatalog.get("am_preprocess")
dataset_dicts=get_amseg_dicts_raw(projdir+"preprocess/",classes)#
#
imageset=random.sample(dataset_dicts,15)
for d in imageset:
    im=cv2.imread(d["file_name"])
    idimagefile=re.search(r"CNN.+\.",d["file_name"]).group()
    # ground truth
    v=Visualizer_font(im[:,:,::-1],
                   metadata=am_metadata_raw,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW
    )
    v._default_font_size=3
    out=v.draw_dataset_dict(d)
    cv2.imwrite(outputdir+'full'+idimagefile+'_show.jpg',out.get_image()[:, :, ::-1])
