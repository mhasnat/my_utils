import numpy as np
import cv2

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