# tile images into majority 1024*1024
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

import shapely
import shapely.geometry
from shapely.geometry import Polygon,MultiPolygon

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
                    
    for x_coordinate in range(0,len(anno["all_points_x"]),2):
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
    if len(points)<=3:
        converted_points_list.append([])
        return converted_points_list
    tilebox=Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
    polygonp=Polygon(points)
    if polygonp.is_valid:
        polygon_inters=polygonp.intersection(tilebox)
        is_polygon=polygon_inters.geom_type=='Polygon'
        is_polygon_multi=polygon_inters.geom_type=='MultiPolygon'
        if (not is_polygon) and (not is_polygon_multi):
            converted_points_list.append([])
        else:
            if is_polygon_multi:
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
            # this is the problem line
            converted_points_in=intersectBoundingBox(points,xmin,ymin,xmax,ymax)
            if len(converted_points_in) > 1:
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
    We want to iterate over the annotations and write each annotation to a line in the csv file where the header line of the file
    will be the keys of the annotations dict.
    '''
    import csv
    with open(output_dir+"regiondata.csv", "w") as csvfile:
        
        fieldnames = ['filename', 'image_id','structure_id','height', 'width', 'category_id' ,
        'bbox', 'segmentation', 'bbox_mode', 'iscrowd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for indr,record in enumerate(dataset_dicts):
            for indo,obj in enumerate(record["annotations"]):
                writer.writerow(obj)
    pass

def processDataSetDicts(dataset_dicts):
    '''
    :param img_dir:
    :return formated COCO dataset:
    :rtype: pandas df
    Objective: Read in the dataset object from the regionsdata.csv. Process the csv into the format required by COCO.
    Each of the vlaue in the coc dataset correspond to the following keys:
    filename: the name of the image
    image_id: the id of the image
    height: the height of the image
    width: the width of the image
    annotations: the annotations of the image
    The keys in the dataset_dict for the tile are:
    filename: tile_id + _ + x_coord + _ + y_coord + .jpg
    name: the class of the tile, convert the class to the corresponding class id from the classes list
    bbox: the bounding box of the tile
    segmentation: the segmentation of the tile
    bbox_mode: the bounding box mode of the tile
    iscrowd: the iscrowd of the tile
    Iterate through the dataset dict and extract the values from the keyys to make a new dataset_dicts for the tiles.
    '''
    # iterate through the dataset_dict and extract the values that the COCO dataset needs
    COCO_dataset = []
    for indr,record in enumerate(dataset_dicts):
        #print(record)
        # if the record has an annotations column and it is not empty
        if 'annotations' in record and len(record['annotations']) > 0:
            # iterate through the annotations
            for inda,annotation in enumerate(record['annotations']):
                # append the keys to the COCO_dataset
                COCO_dataset.append({
                    'UID': annotation['id'],
                    'filename': annotation['filename'],
                    # output dir and image ID is the tild ID and the jpg extension
                    'image_id': annotation['tile_id']+'.jpg',
                    'height': 512,
                    'width': 512,
                    'segmentation': annotation['segmentation'],
                    'bbox': annotation['bbox'],
                    'category_id': annotation['category_id'],
                    'bbox_mode': annotation['bbox_mode'],
                    'iscrowd': annotation['iscrowd']
                })
    # convert the COCO_dataset to a pandas df
    #COCO_dataset = pd.DataFrame(COCO_dataset)
    return COCO_dataset

def addColumn(img_dir):
    '''
    :param img_dir
    :return: regiondata
    :rtype: pandas
    Objective: Read in the regiondata.csv from the img_dir. Add the vlaue val to the data_dir column if the filenamer contians the string "CNN2_Keyence_Nbenth_myc_8" or "CNN2_Keyence_Nbenth_myc_9", else add the value train to the column
    '''
    df = pd.read_csv(img_dir+'regiondata.csv')
    df['data_dir'] = df['filename'].apply(lambda x: 'val' if 'CNN2_Keyence_Nbenth_myc_8' in x or 'CNN2_Keyence_Nbenth_myc_9' in x else 'train')
    df.to_csv(img_dir+'regiondata.csv', index=False)
    return df

def readCOCO(img_dir):
    '''
    :param img_dir:
    :return: COCO_dataset
    :rtype: pandas
    Objective: Read in the regiondata.csv. Split the data from the filenames column into train and val according to the data_dir column in the csv.
    '''
    # list the unique .jpg file names with .jpg extension in the img_dir
    img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    # read the regiondata.csv as a pandas df
    df = pd.read_csv(img_dir+'regiondata.csv')
    # a train and val directory in the img_dir
    train_dir = img_dir+'train/'
    val_dir = img_dir+'val/'
    # iterate over the lines in df, move the images to the train or val directory, if the image has been moved, continue, else move the image
    # iterate over the image list and the rows of the dataframe. if the image in the image list matches the filename in the filename column in the dataframe
    # and the data_dir column is train, move the image to the train directory. if the data_dir column is val, move the image to the val directory. if the image
    # exists in the train or val directory already, continue.
    for index, row in df.iterrows():
        for indi,img in enumerate(img_list):
            if img == row['filename']:
                if row['data_dir'] == 'train':
                    if os.path.exists(train_dir+img):
                        continue
                    else:
                        shutil.move(img_dir+img, train_dir+img)
                elif row['data_dir'] == 'val':
                    if os.path.exists(val_dir+img):
                        continue
                    else:
                        shutil.move(img_dir+img, val_dir+img)
                else:
                    continue
            else:
                continue
    # subset the df to the train and the val according to the data_dir column
    train_df = df[df['data_dir'] == 'train']
    val_df = df[df['data_dir'] == 'val']
    # write a regiondata.csv for the train and val directories
    train_df.to_csv(train_dir+'regiondata.csv', index=False)
    val_df.to_csv(val_dir+'regiondata.csv', index=False)
    return df

# make coco form my csv file
def makeCOCOdataset(img_dir):
    '''
    : param img_dir:
    : return: COCO_dataset

    Objective: For the train and the val directories, we will read in the region.csv file. We will format a list of dictionaries in COCO dataset format. Where each dictionary has the following keys:
    filename, image_id, height, width, segmentation, bbox, category_id, bbox_mode, iscrowd.

    '''
    # read regiondata.csv as a pandas df
    df = pd.read_csv(img_dir+'regiondata.csv')
    # iterate over the lines of the df and create a list of dictionaries with keys of: filename, image_id, height, width, segmentation, bbox, category_id, bbox_mode, iscrowd
    COCO_dataset = []
    for index, row in df.iterrows():
        record = {}
        record['filename'] = str(row['filename'])
        record['image_id'] = row['image_id']
        record['height'] = row['height']
        record['width'] = row['width']
        record['segmentation'] = row['segmentation']
        record['bbox'] = row['bbox']
        record['category_id'] = row['category_id']
        record['bbox_mode'] = row['bbox_mode']
        record['iscrowd'] = row['iscrowd']
        COCO_dataset.append(record)
    return COCO_dataset

projdir='/scratch/yw44924/amf_tile/'
img_dir=projdir+'data/raw/'
output_dir=projdir+'tiling/'
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
for indi,stained_image in enumerate(files):
    print(stained_image)
    img=readImage(img_dir,stained_image)
    tiles=tileImage(img)
    records=IntersectSegmentations(img_dir,output_dir,tiles,img,annotab,stained_image,classes)
    dataset_dicts.extend(records)

writeRegionDataSet(dataset_dicts)
dat=pd.read_csv(img_dir+'regiondata.csv')
COCO_dataset=processDataSetDicts(dataset_dicts)
df=addColumn(output_dir)
df=readCOCO(output_dir)
COCO_dataset=makeCOCOdataset(output_dir+'train/')
