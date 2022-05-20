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
#
datapath='./'#./
restab=pd.read_csv(datapath+"segmentation.txt",delimiter=",")
with open(datapath+'masks.pickle','rb') as handle:
    masklist=pickle.load(handle)

classlist=restab['annotations'].unique()
strainlist=['N102','N108','N110','N162','N43','N66','N68','N6F3','N10','E37','L8','N116']#del 'EZY','E46'
foldlist=['N102-JPEG','N108-JPEG','N110-JPEG','N162-JPEG','N43-no match-JPEG','N66-no match-JPEG','N68-JPEG','N6F3-JPEG','N10-JPEG','E37-JPEG','L8-JPEG','N116-JPEG']
regionlist=['BOT','MID','TOP']
slidlist=['1','2','3']
#
classlist_col=[]
strainlist_col=[]
regionlist_col=[]
slidelist_col=[]
#
areasum_list_col=[]
numseg_list_col=[]
numimgexi_list_col=[]
#
for classele in classlist:
    clasind=restab['annotations']==classele
    for gind,genotype in enumerate(strainlist):
        genind=restab['filename'].str.contains('/'+foldlist[gind]+'/',regex=False)
        for reg in regionlist:
            regind=restab['filename'].str.contains('/'+reg+'/',regex=False)
            for slide in slidlist:
                classele+' '+genotype+' '+reg+' '+slide
                slideind=restab['filename'].str.contains('/'+slide+'/',regex=False)
                sumind=clasind & genind & regind & slideind
                subtab=restab[sumind]
                # areas sum
                rowind=np.where(sumind.values)[0]
                sumarea=0
                for fileele in subtab['filename'].unique():
                    fileind=np.where((subtab['filename']==fileele).values)[0]
                    ind_1file=fileind[0]
                    fullmask=np.full((subtab['height'].iloc[ind_1file],subtab['weight'].iloc[ind_1file]),False)
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
                strainlist_col.append(genotype)
                regionlist_col.append(reg)
                slidelist_col.append(slide)
    
stat_count_tab=pd.DataFrame({
    'class': classlist_col,
    'strain':strainlist_col,
    'region': regionlist_col,
    'slide': slidelist_col,
    'sum_area': areasum_list_col,
    'number_segments': numseg_list_col,
    'number_images': numimgexi_list_col})
zerocond=(stat_count_tab['sum_area']==0) & (stat_count_tab['number_segments']==0) & (stat_count_tab['number_images']==0)
stat_count_tab=stat_count_tab.drop(stat_count_tab[zerocond].index)
stat_count_tab.to_csv("stat_summary.txt",index=False)
#
# stat_count_tab[stat_count_tab['class']=='root']['number_images'].sum()
# len(restab['filename'].unique())
# stat_count_tab['number_segments'].sum()
# restab.shape
