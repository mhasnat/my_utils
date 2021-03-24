#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

import utils_yolo_3 as dn
dn.import_darknet()
    
class Object_Detector(object):
    def __init__(self, dict_path):
        self.yolo_model_dict = dn.load_yolo(dict_path)

    def select_bboxes_by_name(self, allBox_detector_, class_name=None):
        selected_boxes = list()
        for t_box in allBox_detector_:
            #print(t_box[0])
            if t_box[0] == class_name:
                selected_boxes.append(t_box)
        
        return selected_boxes

    def detect_object(self, input_image, score_min=.45, isExtended=False, class_name=None, verbose=False):
        _info = []
        _detection = []

        try:
            ## Detect objects
            _, plate_detection = dn.detect_objects(self.yolo_model_dict["net"], self.yolo_model_dict["meta"], 
                                                   input_image, score_min=score_min)            
            
            if class_name is None:
                _detection = plate_detection
            else:
                _detection = self.select_bboxes_by_name(plate_detection, class_name=class_name)
            
            if verbose:
                print(plate_detection)
                
            if len(plate_detection) > 0:
                for t_plate in plate_detection:
                    t_plate_info = t_plate[2]
                    if isExtended:
                        # return an extended crop based coordinate
                        xmax = min(int(0.5 * (2 * t_plate_info[0] + t_plate_info[2])), input_image.shape[1])
                        xmin = max(int(0.5 * (2 * t_plate_info[0] - t_plate_info[2])), 0)
                        ymax = min(int(0.5 * (2 * t_plate_info[1] + t_plate_info[3])),input_image.shape[0])
                        ymin = max(int(0.5 * (2 * t_plate_info[1] - t_plate_info[3])), 0)
                    else:
                        # return exact crop based coordinate
                        xmax = int(min(t_plate_info[2], input_image.shape[1]))
                        xmin = int(max(t_plate_info[0], 0))
                        ymax = int(min(t_plate_info[3],input_image.shape[0]))
                        ymin = int(max(t_plate_info[1], 0))

                    #plate_cropped = input_image[ymin:ymax,xmin:xmax,:]
                    plate_info = {"x_top_left":None,"y_top_left":None,
                     "x_bottom_right":None,"y_bottom_right":None}
                    plate_info["x_top_left"] = xmin
                    plate_info["y_top_left"] = ymin
                    plate_info["x_bottom_right"] = xmax
                    plate_info["y_bottom_right"] = ymax

                    _info.append(plate_info)
        except:
            print("error @ object detection")

        return _info, _detection      