import numpy as np
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Reads the yolo formatted bboxes and makes them as YOLO inferance output
def get_bboxes_from_file_as_yolo_pred(img_size, labels_path, padding=0, verbose=False, class_names=None):
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
            
            if class_names is None:
                class_label=int(parsed[0])
            else:
                class_label=class_names[int(parsed[0])]
                
            bboxes.append((class_label, 0.99, (round(bb_center_x - bb_width/2  + padding),
                                           round(bb_center_y - bb_height/2  + padding),
                                           round(bb_center_x + bb_width/2  + padding),
                                           round(bb_center_y + bb_height/2  + padding))))
      
    return bboxes

<<<<<<< HEAD
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
=======
# Reads the yolo formatted bboxes (a list/tuple of one prediction) and makes them as YOLO inferance output
def get_bboxes_as_yolo_pred(img_size, yolo_anno, verbose=False):
    (height, width, _) = img_size
    
    bb_center_x = round(yolo_anno[1] * width)
    bb_center_y = round(yolo_anno[2] * height)
    bb_width = round(yolo_anno[3] * width)
    bb_height = round(yolo_anno[4] * height)

    bboxes = [(int(yolo_anno[0]), 0.99, (round(bb_center_x - bb_width/2),
                                   round(bb_center_y - bb_height/2),
                                   round(bb_center_x + bb_width/2),
                                   round(bb_center_y + bb_height/2)))]
      
    return bboxes

## To prepare YOLO annotation from YOLO predictions
# Reads the yolo prediction and convert it into YOLO annotation format
def get_bboxes_as_yolo_anno(img_size, yolo_pred, class_str_dict, verbose=False):
    (height, width, _) = img_size
    class_ = class_str_dict[yolo_pred[0]]
    
    t_box = yolo_pred[2]
    if verbose:
        print(t_box)
        
    cx, cy, w, h = convert_x1y1x2y2_to_cxcywh(img_size, t_box)
    if verbose:
        print(cx, cy, w, h)
        
    anno = (class_str_dict[yolo_pred[0]], cx, cy, w, h)
    
    return anno

def convert_x1y1x2y2_to_cxcywh(im_shape, box):
    dw = 1./im_shape[1]
    dh = 1./im_shape[0]
    cx = (box[0] + box[2])/2.0
    cy = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = cx*dw
    w = w*dw
    cy = cy*dh
    h = h*dh
    return (cx,cy,w,h)
>>>>>>> cc9312db33682156c3ec51552fd2064210a17fbe
