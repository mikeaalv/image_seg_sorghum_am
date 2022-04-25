# this script inference based on trained models
# Mask-RCNN on detectron2:
# the folder structure should be:
#   data/test (images)
#   pretrained: the pretrained model model_best.pth
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
    augs=[T.ResizeShortestEdge(short_edge_length=(640,672,704,736,768,800),max_size=1333,sample_style='choice'),T.RandomBrightness(0.5,2.0),T.RandomCrop("relative_range",[0.5,0.5]),T.RandomFlip(),T.RandomRotation([0,360])]
    return augs

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def get_amseg_dicts_inference(img_dir):
    files=os.listdir(img_dir)
    # only load the images
    seleind=[re.search(r"\.jpg",x) is not None for x in files]
    files=list(compress(files,seleind))
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
    DatasetCatalog.register("am_"+direc,lambda direc=direc: get_amseg_dicts_inference("./data/"+direc))
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
cfg.MODEL.WEIGHTS=os.path.join("./pretrained/model_best.pth")# path to the model we just trained
#
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7# set a custom testing threshold
predictor=DefaultPredictor(cfg)

# test data set
am_metadata_test=MetadataCatalog.get("am_test")
dataset_dicts=get_amseg_dicts_inference("./data/test")#
#
nametab=pd.read_csv("./data/test/locatab.txt",delimiter=",")
#
classlist=[]
polygonlist=[]
hlist=[]
wlist=[]
namelist=[]
arealist=[]
idlist=[]
masklist=[]
confscorelist=[]
for fileind,d in enumerate(dataset_dicts):
    d["file_name"]
    im=cv2.imread(d["file_name"])
    # prediction
    outputs=predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_test,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    # matched names
    filenameloc=re.findall(r"\d+\.jpg",d["file_name"])
    rowind=np.where(nametab[['filenames']]==filenameloc)[0][0]
    oriname=nametab.iloc[rowind,0]
    #
    clasind=outputs['instances'].get('pred_classes')
    allmasks=outputs['instances'].get('pred_masks')
    allscores=outputs['instances'].get('scores')
    for segi in range(clasind.size()[0]):
        if not allmasks[segi,:,:].any():
            continue
        
        namelist.append(oriname)
        idlist.append(fileind)
        hlist.append(v.output.height)
        wlist.append(v.output.width)
        classlist.append(classes[clasind[segi]])
        #
        locmask=np.asarray(allmasks[segi,:,:])
        gmask=GenericMask(locmask,v.output.height,v.output.width)
        mergpolygon=gmask.polygons[0]
        all_points_x=mergpolygon[::2]
        all_points_y=mergpolygon[1::2]
        polygonlist.append(json.dumps({"all_points_x": (all_points_x.tolist()),"all_points_y": (all_points_y.tolist())}))
        #
        pgon=Polygon(zip(all_points_x,all_points_y))
        arealist.append(pgon.area)
        #
        locmasknp=np.array(locmask)
        packmask=np.packbits(locmasknp.view(np.uint8))
        masklist.append(packmask)
        confscorelist.append(allscores[segi].item())
    
    if fileind % 100==0:
        out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('showimage.exp.pred'+str(d['image_id'])+'_test.jpg',out.get_image()[:, :, ::-1])


# vis: tensorboard --logdir ./dir
rectab=pd.DataFrame({
    'filename': namelist,
    'id':idlist,
    'height': hlist,
    'weight': wlist,
    'annotations': classlist,
    'segmentation': polygonlist,
    'area': arealist,
    'confidenceScore': confscorelist})
rectab.to_csv("segmentation.txt",index=False)

# save the result table in ech folder
# dirlist=[re.sub(r"[^\/]+\.jpg","",x[0]) for x in rectab[['filename']].values.tolist() ]
# orifolds=np.unique(np.array(dirlist))
# for orifold in orifolds:
#     foldind=rectab['filename'].str.contains('('+re.escape(orifold)+')')
#     rectab_loc=rectab[foldind]
#     rectab_loc.to_csv(orifold+"segmentation.txt",index=False)

with open('masks.pickle','wb') as handle:
    pickle.dump(masklist,handle,protocol=pickle.HIGHEST_PROTOCOL)

# example all root area for one image
# with open('masks.pickle','rb') as handle:
#     masklist=pickle.load(handle)
# oriname='/work/aelab/AMF/AMF Imaging/0_Image_Collection/ZEISS Primo Star/Georgia/2021/Experiment001_Greenhouse_Colby/1_JPEG/Roots/Snap-1433.jpg'
# rowind=np.where((rectab[['filename']]==oriname).values & (rectab[['annotations']]=='root').values)[0]
# fullmask=np.full((rectab['height'][rowind[0]],rectab['weight'][rowind[0]]),False)
# for indhere in rowind:
#     locmasknpup=np.unpackbits(masklist[indhere]).reshape(fullmask.shape).view(np.bool)
#     # np.array_equal(locmasknpup,locmasknp)
#     fullmask=fullmask | locmasknpup
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.imshow(fullmask,aspect='auto',cmap=plt.cm.gray,interpolation='nearest')
# plt.savefig('foo.pdf')
