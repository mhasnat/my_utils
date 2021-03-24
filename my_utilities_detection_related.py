import numpy as np
import os, glob
import cv2

def get_my_detector(det_type='yv3'):
    from utils_machine import get_cuda_version
    cuda_version = get_cuda_version()
    
    if cuda_version=='10.1':
        if det_type=='yv3':
            import Object_Detector_2 as od
        else:
            import Object_Detector_3 as od
    else:
        if det_type=='yv3':
            import Object_Detector as od
        else:
            od = None
            print('Unable to load your specific detector in the machine with - '+cuda_version)

    return od

def filter_bbox_by_name(r, name_='Vehicle'):
    sel_det = []
    for b in r:
        if b[0] == name_:
            sel_det.append(b)
    return sel_det

def filter_bbox_size(r, ht, wd):
    sel_det = []
    for b in r:
        height_ = b[2][2]-b[2][0]
        width_ = b[2][3]-b[2][1]
        
        if height_<ht and width_<wd:
            #print((point_, ht))
            sel_det.append(b)
    return sel_det

def filter_bbox_ht(r, ht, isCent=False):
    sel_det = []
    for b in r:
        if isCent:
            point_ = b[2][1] + ((b[2][3]-b[2][1])/2)
        else:
            point_ = b[2][3]
        if point_ > ht:
            #print((point_, ht))
            sel_det.append(b)
    return sel_det

def filter_bbox_wd(r, wd, isCent=False):
    sel_det = []
    for b in r:
        if isCent:
            point_ = b[2][0] + ((b[2][2]-b[2][0])/2)
        else:
            point_ = b[2][2]
                          
        if point_ > wd:
            #print((point_, wd))
            sel_det.append(b)
    return sel_det

def filter_detections_by_region(tim_shape, det_, y_split=2, x_split=4):
    y_fix = int(tim_shape[0]/y_split)
    if x_split>0:
        x_fix = int(tim_shape[1]/x_split)
    det_ = filter_bbox_ht(det_, y_fix, isCent=True)
    if x_split>0:
        det_ = filter_bbox_wd(det_, x_fix, isCent=True)
    
    return det_

def get_filtered_detections(tim, det_, y_split=2, x_split=4, isCent_w=False, isCent_h=False):   
    y_fix = int(tim.shape[0]/y_split)    
    if x_split>0:
        x_fix = int(tim.shape[1]/x_split)
        
    det_ = filter_bbox_ht(det_, y_fix, isCent=isCent_h)    
    if x_split>0:
        det_ = filter_bbox_wd(det_, x_fix, isCent=isCent_w)
    
    return det_

def get_filtered_detections_from_file_name(t_file, det_, y_split=2, x_split=4):
    if t_file.split('.')[-1] == 'npy':
        tim = np.load(t_file)
    else:
        tim = cv2.imread(t_file)[:,:,::-1]
    tim = get_binned_image(tim)
    
    y_fix = int(tim.shape[0]/y_split)
    
    if x_split>0:
        x_fix = int(tim.shape[1]/x_split)
    det_ = filter_bbox_ht(det_, y_fix, isCent=True)
    
    if x_split>0:
        det_ = filter_bbox_wd(det_, x_fix, isCent=True)
    
    return tim, det_

def parse_detection_info(box_):
    t_id = box_[0]
    t_conf = box_[1]
    bb_ = np.asarray(box_[2], dtype=np.int32)
    t_box = np.asarray(bb_)

    return t_id, t_conf, t_box

def get_annotations_from_GT(tImg_anno, r, show_label=True):
    for b in r:
        (tx1, ty1, tx2, ty2) = (b[0], b[1], b[2], b[3])
        tImg_anno = cv2.rectangle(tImg_anno, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)        
        if show_label:
            cv2.putText(tImg_anno, str(t_id), (int(tx1), int(ty1-10)), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6,(0,0,255), 2)
    
    return tImg_anno

def get_annotations_from_detections_multi_objects(tImg_anno, r, color_dict=None, show_label=True, line_thikness=5):
    for b in r:
        t_id, t_conf, t_box = parse_detection_info(b)
        
        if color_dict is None:
            obj_color = (255, 0, 255)
        else:
            obj_color = color_dict[t_id]
            
        #print(t_conf)
        (tx1, ty1, tx2, ty2) = (t_box[0], t_box[1], t_box[2], t_box[3])

        if t_conf > 0.05:
            tImg_anno = cv2.rectangle(tImg_anno, (tx1, ty1), (tx2, ty2), obj_color, line_thikness)        
            if show_label:
                cv2.putText(tImg_anno, str(t_id), (int(tx1), int(ty1-10)), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6,(255,255,255), 2)
                
                cx = int(tx1 + (tx2-tx1)/2)
                cy = int(ty1 + (ty2-ty1)/2)
                cv2.circle(tImg_anno, (cx,cy), radius=3, color=(255, 0, 0), thickness=-1)
    
    return tImg_anno

def get_annotations_from_detections(tImg_anno, r, show_label=True):
    for b in r:
        t_id, t_conf, t_box = parse_detection_info(b)
        #print(t_conf)
        (tx1, ty1, tx2, ty2) = (t_box[0], t_box[1], t_box[2], t_box[3])

        if t_conf > 0.05:
            tImg_anno = cv2.rectangle(tImg_anno, (tx1, ty1), (tx2, ty2), (255, 0, 255), 5)        
            if show_label:
                cv2.putText(tImg_anno, str(t_id), (int(tx1), int(ty1-10)), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6,(0,0,255), 2)
                
                cx = int(tx1 + (tx2-tx1)/2)
                cy = int(ty1 + (ty2-ty1)/2)
                cv2.circle(tImg_anno, (cx,cy), radius=3, color=(255, 0, 0), thickness=-1)
    
    return tImg_anno

def get_croped_images_from_detections(tim, det_, padding=0):
    all_crops = []
    for t_ in det_:
        _,_,xy_ = parse_detection_info(t_)
        #print(xy_)
        xy_ = np.asarray(xy_)
        xy_[np.where(xy_<0)] = 0
        
        top = max(xy_[1]-padding, 0)
        bottom = min(xy_[3]+padding, tim.shape[0])
        left = max(xy_[0]-padding, 0)
        right = min(xy_[2]+padding, tim.shape[1])
        
        t_cr = tim[top:bottom, left:right, :]
        all_crops.append(t_cr)
    return all_crops

def get_crop_coordinate(response):
    return (response['x_top_left'], response['y_top_left'], response['x_bottom_right'], response['y_bottom_right'])