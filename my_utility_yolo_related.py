import numpy as np
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Reads the yolo formatted bboxes and makes them as YOLO inferance output
def get_bboxes_from_file_as_yolo_pred(img_size, labels_path, padding=0, verbose=False):
    (height, width, _) = img_size
    
    bboxes = []
    
    with open(labels_path) as lp:
        for line in lp:            
            parsed = [float(x) for x in line.split(' ')]
            if verbose:
                print(parsed)                        
            
            bb_center_x = round(parsed[1] * width)
            bb_center_y = round(parsed[2] * height)
            bb_width = round(parsed[3] * width)
            bb_height = round(parsed[4] * height)
            
            bboxes.append((int(parsed[0]), 0.99, (round(bb_center_x - bb_width/2  + padding),
                                           round(bb_center_y - bb_height/2  + padding),
                                           round(bb_center_x + bb_width/2  + padding),
                                           round(bb_center_y + bb_height/2  + padding))))
      
    return bboxes

# colors used to visulize bouding boxes out of bound .....
GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]

#reads the yolo formatted bboxes and makes them into ai ready 
def get_bboxes_from_file(img_size, labels_path, padding):
    (height, width, _) = img_size
    
    labels = []
    bboxes = []
    
    with open(labels_path) as lp:
        for line in lp:
            parsed = [float(x) for x in line.split(' ')]
            labels.append(int(parsed[0]))
            
            bb_center_x = round(parsed[1] * width)
            bb_center_y = round(parsed[2] * height)
            bb_width = round(parsed[3] * width)
            bb_height = round(parsed[4] * height)
            
            bboxes.append(BoundingBox(
                x1=round(bb_center_x - bb_width/2  + padding),
                y1=round(bb_center_y - bb_height/2  + padding),
                x2=round(bb_center_x + bb_width/2  + padding),
                y2=round(bb_center_y + bb_height/2  + padding)))
      
    bboxes = BoundingBoxesOnImage(bboxes, shape=img_size)
    return bboxes, labels

def draw_bbs(image, bbs, border):
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = GREEN
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = RED
        image = bb.draw_on_image(image, size=3, color=color)

    return image