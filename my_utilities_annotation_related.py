import numpy as np
import os, glob
import cv2
import PIL.Image
import PIL.ImageDraw 

def extract_shapes(df, idx, unqID=False): 
    shapes = []  
    labels = []
    for i in idx:
        region = json.loads(df.region_shape_attributes[i])
        
        if unqID:
            attributes = 1
        else:
            attributes = int(df.region_id[i]) + 1
            
        shapes.append([tuple(xy) for xy in zip(region["all_points_x"], region["all_points_y"])])
        labels.append(attributes)
    return shapes, labels

def shape_to_mask(width, height, points): 
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = PIL.Image.fromarray(mask) 
    draw = PIL.ImageDraw.Draw(mask)  
    draw.polygon(xy=points, outline=1, fill=1)  
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(width, height, shapes, label_ids): 
    cls = np.zeros((height, width), dtype=np.int32) 
    for points, label_id in zip(shapes, label_ids):  
        mask = shape_to_mask(width, height, points)
        cls[mask] = label_id 
    return cls