# this script inference based on trained models
# Mask-RCNN on detectron2:
# the folder structure should be:
#   data/AM_classify2/test (images)
#   run: the running folder contains scripts am_seg_inference.py
#   pretrained: the pretrained model model_best.pth
# Running:
#   module load Detectron2/0.3-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
#
#   time python am_seg_inference.py --seed 1 --net-struct mask_rcnn_R_50_FPN_3x --gpu-use=0 

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

import torch
import torch.nn as nn
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer,HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader,DatasetMapper,build_detection_train_loader,MetadataCatalog,DatasetCatalog
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
# default Arguments
args_internal_dict={
    "batch_size": (2,int),
    "epochs": (300,int),
    "learning_rate": (0.00025,float),
    # "no_cuda": (False,bool),
    "seed": (1,int),
    "net_struct": ("mask_rcnn_R_50_FPN_3x",str),
    # "optimizer": ("adam",str),##adam
    "gpu_use": (1,int),# whehter use gpu 1 use 0 not use
    "freeze_at": (2,int), #till n block ResNet18 has 10 blocks
    "aug_flag": (1,int)#whether do the more comprehensive augmentation (1) or not (0)
}
def build_aug(cfg):
    return [
        T.ResizeShortestEdge(
            short_edge_length=(640, 672, 704, 736, 768, 800),
            max_size=1333,
            sample_style='choice',
        ),
        T.RandomBrightness(0.5, 2.0),
        T.RandomCrop("relative_range", [0.5, 0.5]),
        T.RandomFlip(),
        T.RandomRotation([0, 360]),
    ]

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def get_amseg_dicts_inference(img_dir):
    files=os.listdir(img_dir)
    dataset_dicts=[]
    for idx,file in enumerate(files):
        record={}
        filename=os.path.join(img_dir,file)
        height,width=cv2.imread(filename).shape[:2]
        record["file_name"]=filename
        record["image_id"]=idx
        record["height"]=height
        record["width"]=width
        dataset_dicts.append(record)
    
    return dataset_dicts

class newtrainer(DefaultTrainer):
    @classmethod
    # def build_evaluator(cls,cfg,dataset_name,output_folder=None):
    #     if output_folder is None:
    #         output_folder=os.path.join(cfg.OUTPUT_DIR,"validation")
    #     return COCOEvaluator(dataset_name,("bbox","segm"),True,output_folder)
    def build_train_loader(cls,cfg):
        if cfg.AUG_FLAG==1:
            mapper=DatasetMapper(cfg,is_train=True,augmentations=build_aug(cfg))
        else:
            mapper=DatasetMapper(cfg,is_train=True)
        
        return build_detection_train_loader(cfg,mapper=mapper)

# passing arguments
parser=argparse.ArgumentParser(description='PyTorch Example')
for key in args_internal_dict.keys():
    parser=parse_func_wrap(parser,key,args_internal_dict)

args=parser.parse_args()
classes=['root','AMF internal hypha','AMF external hypha','AMF arbuscule','AMF vesicle','AMF spore','others']
for direc in ['test']:
    DatasetCatalog.register("am_"+direc,lambda direc=direc: get_amseg_dicts_inference("../data/AM_classify2/"+direc))
    MetadataCatalog.get("am_"+direc).set(thing_classes=classes)#classes name list

# configuration parameters
cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+args.net_struct+".yaml"))
cfg.DATASETS.TEST=()
cfg.DATALOADER.NUM_WORKERS=2
cfg.SOLVER.IMS_PER_BATCH=args.batch_size
if args.gpu_use!=1:
    cfg.MODEL.DEVICE='cpu'

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128#Number of regions per image used to train RPN. faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES=len(classes)# (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT=args.freeze_at
cfg.SEED=args.seed
cfg.AUG_FLAG=args.aug_flag
#
# inference
cfg.MODEL.WEIGHTS=os.path.join("../pretrained/model_best.pth")# path to the model we just trained
#
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7# set a custom testing threshold
predictor=DefaultPredictor(cfg)

# test data set
am_metadata_test=MetadataCatalog.get("am_test")
dataset_dicts=get_amseg_dicts_inference("../data/AM_classify2/test")

classlist=[]
# Edit this loop to record different information
for d in dataset_dicts:
    im=cv2.imread(d["file_name"])
    outputs=predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    clasind=outputs['instances'].get('pred_classes')
    classlist.extend([classes[ele] for ele in clasind])
    # prediction
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_test,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('showimage.exp.pred'+str(d['image_id'])+'_test.jpg',out.get_image()[:, :, ::-1])

# vis: tensorboard --logdir ./dir
