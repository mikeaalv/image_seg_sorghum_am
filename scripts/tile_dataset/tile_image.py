# tile images into majority 512*512
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import uuid as uuid
import cv2
import re
import shutil
from operator import itemgetter
import math
import random

import shapely
import shapely.geometry
from shapely.geometry import Polygon,MultiPolygon
from shapely.validation import make_valid

def readImage(img_dir,filename):
    '''
    :param img_dir:
    :param filename:
    :return image:
    '''
    img_file=os.path.join(img_dir,filename)
    img=cv2.imread(img_file)
    height = img.shape[0]
    width = img.shape[1]
    # padd the image to be divisible by 512
    # padd the image
    pad_h=(512 - (height % 512)) % 512
    pad_w=(512 - (width % 512)) % 512
    # pad the image
    img=cv2.copyMakeBorder(img,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,value=[0,0,0])
    return img

def tileImage(img):
    '''
    :param img:
    :return list of coordinates:
    :rtype list:
    Objective: output a list of coordinates for the bounding boxes of the tiles
    '''
    tiles=[[]]
    for i in range(0,img.shape[0],512):
            for j in range(0,img.shape[1],512):
                #print(i,j)
                #tile=img[i:i+512,j:j+512]
                xmin=j
                ymin=i
                xmax=j+512
                ymax=i+512
                tiles[0].append([xmin,ymin,xmax,ymax])
    return tiles

def convertPoints(anno):
    points = [[]]
                    
    for x_coordinate in range(0,len(anno["all_points_x"]),1):
        points.append([anno["all_points_x"][x_coordinate],anno["all_points_y"][x_coordinate]])
    
    # remove empyt lists from the list of points
    points = [x for x in points if x]
    return points

def intersectBoundingBox(points,xmin,ymin,xmax,ymax):
    converted_points = []
    for ind, p in enumerate(points):
        p2=list(p)
        if p2[0]>=xmin and p2[0]<=xmax and p2[1]>=ymin and p2[1]<=ymax:
            p2[0]=p2[0]-xmin
            p2[1]=p2[1]-ymin
            converted_points.append(p2)
    #print(converted_points)
    return converted_points

def intersectmask(points,xmin,ymin,xmax,ymax):
    converted_points_list=[]
    if len(points)<3:
        converted_points_list.append([])
        return converted_points_list
    tilebox=Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
    polygonp=Polygon(points)
    polygonp=make_valid(polygonp)
    if polygonp.is_valid:
        polygon_inters=polygonp.intersection(tilebox)
        is_polygon=polygon_inters.geom_type=='Polygon'
        is_polygon_multi=polygon_inters.geom_type=='MultiPolygon'
        is_mutliothers=polygon_inters.geom_type=='GeometryCollection'
        if (not is_polygon) and (not is_polygon_multi) and (not is_mutliothers):
            converted_points_list.append([])
        else:
            if is_polygon_multi or is_mutliothers:
                polygon_inters_list=polygon_inters
            else: # elif is_polygon:
                polygon_inters_list=MultiPolygon([polygon_inters])
            
            for geom in polygon_inters_list.geoms:
                if geom.geom_type=='Point' or geom.geom_type=='LineString':
                    continue
                converted_points=[]
                poly_inter_points=geom.exterior.coords[:-1]
                poly_inter_points=[list(ele) for ele in poly_inter_points]
                for ind, p in enumerate(poly_inter_points):
                    p2=list(p)
                    p2[0]=p2[0]-xmin
                    p2[1]=p2[1]-ymin
                    converted_points.append(p2)
                converted_points_list.append(converted_points)
    else:
        converted_points_list.append([])
    return converted_points_list

def readAnnotation(img_dir):
    '''
    :param img_dir: image folder
    :return list of dictionaries: annotation table and file list
    '''
    anno_file=os.path.join(img_dir,"regiondata.csv")
    annotab=pd.read_csv(anno_file,delimiter=",")
    files=annotab['filename'].unique()
    return annotab, files

def IntersectSegmentations(img_dir,output_dir,tiles, img, annotab, file,classes):
    '''
    :param tiles:
    :paramtype list:
    :param img:
    :paramtype numpy array:
    :param annotab:
    :paramtype pandas dataframe:
    :param files:
    :paramtype list:
    :param classes:
    :paramtype list:
    :return dataset_dicts:
    :rtype list:
    Objective: iterate over each of the segmentations in the image and intersect them with the tile bounding boxes
    '''
    filename=os.path.join(img_dir,file)
    records=[]
    nonvalid_seg_counter=0
    # iterate over the tile coordinates
    for ind,tile in enumerate(tiles[0]):
        record = {}
        # get the coordinate over the tile image
        xmin=tile[0]
        ymin=tile[1]
        xmax=tile[2]
        ymax=tile[3]
        
        # make a tile id using the xmin,ymin,xmax,ymax and the filename
        tile_id=file[:-4] + '_'+ str(xmin)+"_"+str(ymin)+"_"+str(xmax)+"_"+str(ymax)
        # begin building the record by adding the information for the COCO dataset
        record["filename"] = tile_id + '.jpg'
        record["height"] = 512
        record["width"] = 512
        # make an empty list of objects for record annotation
        record["annotations"] = []
        subtab = annotab[annotab['filename'] == file]
        objs =[]
        for anno_i in range(subtab.shape[0]):
            # make a UID for each polygon
            uid = str(uuid.uuid4())
            tab_rec=subtab.iloc[anno_i]
            loadcldict=json.loads(tab_rec['region_attributes'])
            # get the catagory id
            category_id=classes.index(loadcldict['object'])
            # convert the category id to the class name by using the classes array
            className=classes[category_id]
            anno=json.loads(tab_rec["region_shape_attributes"])
            if len(anno)==0:
                continue
            #print(anno)
            points=convertPoints(anno)
            # quick check 1 (this was for speed purpose but the crossing patterns are more complex and might have no point within)
            # converted_points_in=intersectBoundingBox(points,xmin,ymin,xmax,ymax)
            converted_points_in=[1]
            if len(converted_points_in) >= 1:
                converted_points_list=intersectmask(points,xmin,ymin,xmax,ymax)
                
                for indp,converted_points in enumerate(converted_points_list):
                    if len(converted_points)>0:
                        Sxmin=min(converted_points,key=lambda x:x[0])[0]
                        Symin=min(converted_points,key=lambda x:x[1])[1]
                        Sxmax=max(converted_points,key=lambda x:x[0])[0]
                        Symax=max(converted_points,key=lambda x:x[1])[1]
                        converted_points=[item for sublist in converted_points for item in sublist]
                        Segbbox = [Sxmin,Symin,Sxmax,Symax]
                        obj = {
                            'filename': tile_id + '.jpg',
                            "image_id": tile_id,
                            'structure_id': uid,
                            'height': 512,
                            'width': 512,
                            "category_id": className,
                            "bbox": Segbbox,
                            "segmentation": converted_points,
                            "bbox_mode": 'BoxMode.XYXY_ABS',
                            "iscrowd":0,
                            }
                        #objs.append(obj)
                        record["annotations"].append(obj)
                    else:
                        print('2:'+str(ind)+'_'+str(anno_i)+'_'+className)
                        nonvalid_seg_counter=nonvalid_seg_counter+1
        if len(record['annotations']) > 0:
            #subset the image to the tile coordinates
            subimg=img[ymin:ymax,xmin:xmax]
            # write the tile image to the output directory
            cv2.imwrite(os.path.join(output_dir,tile_id+'.jpg'),subimg)
            records.append(record)
    print('nonvalid segmentation: '+str(nonvalid_seg_counter))
    return records

def writeRegionDataSet(dataset_dicts):
    '''
    :param dataset_dicts:
    :return:
    
    # Objective: Iterate over the dataset_dicts, each record will consist of filename, image_id, height, width, and annotations.
    We want to iterate over the annotations and write each annotation to a line in the tsv file where the header line of the file
    will be the keys of the annotations dict.
    '''
    import csv
    with open(output_dir+"regiondata.csv", "w") as csvfile:
        
        fieldnames = ['filename', 'image_id','structure_id','height', 'width', 'category_id' ,
        'bbox', 'segmentation', 'bbox_mode', 'iscrowd']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames,delimiter='\t')
        writer.writeheader()
        for indr,record in enumerate(dataset_dicts):
            for indo,obj in enumerate(record["annotations"]):
                writer.writerow(obj)
    pass

projdir='/scratch/yw44924/amf_tile/'
img_dir=projdir+'data/raw/'
output_dir=projdir+'tiling/'
preproc_dir=projdir+'preprocess/'
data_sep_dir=projdir+'datasep/'
qc_dir=projdir+'qc/'
os.makedirs(output_dir,exist_ok=True)
os.makedirs(preproc_dir,exist_ok=True)
os.makedirs(data_sep_dir,exist_ok=True)
os.makedirs(qc_dir,exist_ok=True)
os.chdir(output_dir)
annotab,files=readAnnotation(img_dir)
dataset_dicts=[]
classes=['root','AMF internal hypha','AMF external hypha','AMF arbuscule','AMF vesicle','AMF spore','others']
# clean up the table
annotab['region_attributes']=annotab['region_attributes'].apply(lambda y: np.nan if y=='{}' or y=='{"object":undefined}' else y)
annotab=annotab.dropna(subset=['region_attributes'])
annotab['region_attributes']=annotab['region_attributes'].apply(lambda y: '{"object":"root"}' if y=='{"object":"Root"}' else y)
rem=annotab['region_attributes']
files=annotab['filename'].unique()
# preprocess the raw data
for file_name in files:
    shutil.copy(img_dir+file_name,preproc_dir+file_name)

annotab.to_csv(preproc_dir+'regiondata.csv',index=False)
#
for indi,stained_image in enumerate(files):
    print(stained_image)
    img=readImage(img_dir,stained_image)
    tiles=tileImage(img)
    records=IntersectSegmentations(img_dir,output_dir,tiles,img,annotab,stained_image,classes)
    dataset_dicts.extend(records)


writeRegionDataSet(dataset_dicts)
dat=pd.read_csv('regiondata.csv',delimiter="\t")
dat_sub=dat[['filename','segmentation','category_id']]
filename=[]
file_size=[]
file_attributes=[]
region_count=[]
region_id=[]
region_shape_attributes=[]
region_attributes=[]
counttab=dat_sub['filename'].value_counts()
for index, row in dat_sub.iterrows():
    filename.append(row['filename'])
    file_size.append(os.stat(row['filename']).st_size)
    file_attributes.append({})
    region_count.append(counttab[row['filename']])
    region_id.append(index)
    points=json.loads(row['segmentation'])
    shapedic={'name':'polygon','all_points_x': points[0::2],'all_points_y': points[1::2]}
    region_shape_attributes.append(json.dumps(shapedic))
    region_attributes.append(row['category_id'])
    
dat_new=pd.DataFrame({
    'filename': filename,
    'file_size':file_size,
    'file_attributes': file_attributes,
    'region_count': region_count,
    'region_id': region_id,
    'region_shape_attributes': region_shape_attributes,
    'region_attributes': region_attributes})

dat_new.to_csv(output_dir+'regiondata_new.csv',index=False,sep='\t')

# separation into different groups
testperc=[0.1,0.1]#validation and test
#no repeats (there will be multiple annotations for the same image)
nonduplicated_ind=[not ele for ele in dat_new['filename'].duplicated(keep='first').tolist()]
datatab2=dat_new.loc[nonduplicated_ind]
totsampsize=datatab2.shape[0]
files=np.array(datatab2['filename'].tolist())
numsampvalid=math.floor(totsampsize*testperc[0])
numsamptest=math.floor(totsampsize*testperc[1])
sampleind=set(range(0,totsampsize))
testind=np.sort(np.array(random.sample(sampleind,numsamptest)))
testindset=set(testind)
validind=np.sort(np.array(random.sample(sampleind.difference(testindset),numsampvalid)))
validindset=set(validind)
trainind=np.sort(np.array(list(sampleind.difference(testindset.union(validindset)))))
indlist=[trainind,validind,testind]
setnames=['train','validate','test']
os.makedirs(data_sep_dir+'train',exist_ok=True)
os.makedirs(data_sep_dir+'test',exist_ok=True)
os.makedirs(data_sep_dir+'validate',exist_ok=True)

for i in range(len(indlist)):
    ind=indlist[i]
    setname=setnames[i]
    list_res=[]
    for file in files[ind]:
        sourcfile=output_dir+file
        if os.path.isfile(sourcfile):
            shutil.copy(sourcfile,data_sep_dir+setname+'/'+file)
            list_res.append(dat_new[dat_new['filename']==file])
        else:
            print('non existence file:'+file+'\n')
    
    resdf=pd.concat(list_res)
    resdf.to_csv(data_sep_dir+setname+'/'+'regiondata.csv',sep='\t',index=False)
