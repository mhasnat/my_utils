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