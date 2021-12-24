import os, glob
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import rmtree
import ast
from PIL import Image
import cv2
#from matplotlib import pyplot as plt
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

#######
import sys
#sys.path.insert(0,"/raid/data/nshvai_cyclope/deps/cyclope-oneshot/python/main/utils")
sys.path.insert(0,"/cyclope/share/notebooks/notebooks-old/utils/")

from utils_gpu import pick_gpu_lowest_memory
import os
print(str(pick_gpu_lowest_memory()))
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
str(pick_gpu_lowest_memory())


from utils_yolo import load_yolo, detect_yolo, import_darknet

def get_top_preds(all_recall, prediction_name_list, num_keep=5):
    srtIndx = np.argsort(np.asarray(all_recall) * -1)[:num_keep]
    pred_names = np.asarray(prediction_name_list)[srtIndx]
        
    return pred_names

def get_weights_to_keep(dict_path_list, all_recall, prediction_name_list, num_keep=5):
    # Create Dictionary of name vs weight path    
    pred_name_list = []
    weights_path_list = []
    for dict_path in dict_path_list:
        pred_name_list.append(dict_path["prediction_name"])
        weights_path_list.append(dict_path["weights"])
    pred_name_weight_path_dict = dict(zip(pred_name_list, weights_path_list))       
    
    #for t_ in pred_name_weight_path_dict.keys():
    #    print(t_)
    
    srtIndx = np.argsort(np.asarray(all_recall) * -1)[:num_keep]
    pred_names = np.asarray(prediction_name_list)[srtIndx]
    
    #print(pred_names)
    wt_paths_to_keep = [pred_name_weight_path_dict[t_] for t_ in pred_names]
    return wt_paths_to_keep, pred_names

def get_yolo_pred_filepaths(filepaths,yolo_model_dict,score_min=0.05):
    res_list = []
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
            tImg = cv2.imread(filepath)[:,:,::-1]
                
            r = detect_yolo(yolo_model_dict["net"], yolo_model_dict["meta"], 
                            tImg, version="alexeyab", score_min=score_min, verbose=False)
            for detection in r:
                class_name, score, [xmin, ymin, xmax, ymax] = detection
                class_name = str(class_name)
                
                res = {"filepath":filepath,
                       "filename":filename,
                        "class":class_name, 
                       "score":score, 
                       "xmin":xmin, 
                       "ymin":ymin, 
                       "xmax":xmax, 
                       "ymax":ymax}
                
                res_list.append(res)
                
    return res_list

def get_prediction_df(filepaths,dict_path, version="alexeyab",score_min=0.05):
    print(dict_path)
    yolo_model_dict = load_yolo(dict_path, version=version)
    all_res = get_yolo_pred_filepaths(filepaths,yolo_model_dict,score_min=score_min)
    prediction_df = pd.DataFrame(all_res)
    prediction_df["index_pred"] = prediction_df.index
    
    return prediction_df
#######


def box_iou_ratio(a, b):
    """
    :param a: Box A
    :param b: Box B
    :return: Ratio = AnB / AuB
    """
    w_intersection = max(0, (min(a[2], b[2]) - max(a[0], b[0])))
    h_intersection = max(0, (min(a[3], b[3]) - max(a[1], b[1])))
    s_intersection = w_intersection * h_intersection

    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intersection) / (s_a + s_b - s_intersection)

def get_iou_refined_boxes(t_box_all, iou_th=0.5):
    # Compute IoU among all boxes
    t_score = np.zeros((len(t_box_all), len(t_box_all)), dtype=np.float)
    for ti in range(len(t_box_all)):
        for tj in range(ti+1, len(t_box_all)):
            t_iou = box_iou_ratio(t_box_all[ti][2], t_box_all[tj][2])
            if t_iou>iou_th:
                t_score[ti, tj] = t_iou
                t_score[tj, ti] = t_iou                    
    t_score_sum = np.sum(t_score, axis=1)

    # Find valid 
    res_final = []
    valid_status = np.ones((len(t_score_sum)), dtype=bool)
    for jj in range(len(t_score_sum)):
        if t_score_sum[jj]>0:
            ol_indx = np.where(t_score[jj, :]>0)[0]

            for t_i in ol_indx:
                # check confidense and select accordingly
                if t_box_all[jj][1]>t_box_all[t_i][1]:
                    valid_status[t_i] = False
                    res_final.append(t_box_all[jj])
        else:
            res_final.append(t_box_all[jj])

    return res_final

def get_merged_prediction_truth(prediction_df,ground_truth_df):
    merged_df = prediction_df.merge(ground_truth_df, how="outer", on='filename',suffixes=('_pred', '_ground'))
    merged_df["iou"] = merged_df.apply(lambda row: box_iou_ratio(
    [row["xmin_ground"], row["ymin_ground"], row["xmax_ground"], row["ymax_ground"]],
    [row["xmin_pred"], row["ymin_pred"], row["xmax_pred"], row["ymax_pred"]]),
                                   axis=1)
    
    return merged_df

### Recall and precision calculation
def get_statistics_df_slices(merged_df):
    merged_df_assignment_recall = merged_df.sort_values(['obj_id','iou'],ascending=False).drop_duplicates(subset=["obj_id"]).sort_values(['index_pred','iou'],ascending=False).drop_duplicates(subset=["index_pred"])
    matched_df = merged_df_assignment_recall[(merged_df_assignment_recall['iou']>0.5) ]
    
    matched_df_correct_class = matched_df[matched_df['class_ground']==matched_df['class_pred']]
    matched_df_wrong_class = matched_df[matched_df['class_ground']!=matched_df['class_pred']]
    #unmatched_detections = prediction_df[~prediction_df['index_pred'].isin(matched_df.index_pred.values)]
    #unmatched_objects = ground_truth_df[(ground_truth_df['region_count']>0) &
    #                               (~ground_truth_df['obj_id'].isin(matched_df.obj_id.values))]
    
    return matched_df,matched_df_correct_class,matched_df_wrong_class

### Metrics
def get_metrics(ground_truth_objects_df,
                prediction_df,
                matched_df_correct_class,
                matched_df_wrong_class,
                class_names,
                score_threshold=0.45):
    nb_objects = ground_truth_objects_df[(ground_truth_objects_df['class'].isin(class_names))].shape[0]
    nb_objects_matched_correctly = matched_df_correct_class[(matched_df_correct_class['class_ground'].isin(class_names))
                                                           &(matched_df_correct_class['score']>=score_threshold)].shape[0]
    nb_predictions = prediction_df[(prediction_df['class'].isin(class_names))
                                   &(prediction_df['score']>=score_threshold)].shape[0]
    
    nb_FP = nb_predictions - nb_objects_matched_correctly
    nb_TP = nb_objects_matched_correctly
    nb_FN = nb_objects - nb_objects_matched_correctly
    nb_detected_wrong_class = matched_df_wrong_class[(matched_df_wrong_class['class_ground'].isin(class_names))].shape[0]
    
    if nb_objects == 0:
        recall = 1
    else:
        recall = 1.0*nb_TP/nb_objects
        
    if nb_predictions == 0:
        precision = 1
    else:
        precision = 1.0*nb_TP/nb_predictions
        
    eps = 0.0000000001
    f1 = 2*precision*recall/(precision + recall + eps)
    
    output_dict = {
    'class_names':class_names,
    'score_threshold':score_threshold,
    'nb_objects':nb_objects,
    'nb_objects_matched_correctly':nb_objects_matched_correctly,
    'nb_predictions':nb_predictions,
    'nb_FP':nb_FP,
    'nb_FN':nb_FN,
    'nb_detected_wrong_class':nb_detected_wrong_class,
    'recall':recall,
    'precision':precision,
        'f1':f1
    }
    
    return output_dict

### Evaluation
def evaluate(prediction_csv_path, ground_truth_df, ground_truth_objects_df, mapping):
    #prediction_df = read_predictions(prediction_csv_path)
    prediction_df = pd.read_csv(prediction_csv_path)
    prediction_name = os.path.basename(prediction_csv_path.split('.')[0])
    merged_df = get_merged_prediction_truth(prediction_df,ground_truth_df)
    
    matched_df,matched_df_correct_class,matched_df_wrong_class = get_statistics_df_slices(merged_df)
    possible_class_names = [[x] for x in mapping.values()]
    
    output_dict_list = []
    for class_names in possible_class_names:
        for score_threshold in np.linspace(0.1,1,18,endpoint=False):
            metrics_dict = get_metrics(ground_truth_objects_df,
                        prediction_df,
                        matched_df_correct_class,
                        matched_df_wrong_class,
                        class_names,
                        score_threshold=score_threshold)
            metrics_dict['prediction_name'] = prediction_name
            output_dict_list.append(metrics_dict)
    
    return output_dict_list

def map_class(list_class):
    if len(list_class)==1:
        return list_class[0]
    else:
        return 'All'    
    
def get_labels(filepath):
    label_filepath = '.'.join(filepath.split('.')[:-1]) + '.txt'
    with open(label_filepath) as f:
        labels = f.readlines()
        
    labels = [x.strip() for x in labels]
    
    rows = []
    for label in labels:
        class_idx, xc, yc, w, h = label.split(" ")
        class_idx = int(class_idx)
        xc = float(xc)
        yc = float(yc)
        w = float(w)
        h = float(h)
        
        row = {"filepath":filepath,
               "class_idx":class_idx,
              "xc":xc,
              "yc":yc,
              "w":w,
              "h":h,}
        
        rows.append(row)
        
    return rows

def get_color_image(img):
    if np.ndim(img) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def get_annotations_from_BoundingBox(tImg_anno, r):
    for b in r:
        (tx1, ty1, tx2, ty2) = (b.x1, b.y1, b.x2, b.y2)
        tImg_anno = cv2.rectangle(tImg_anno, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)        
    
    return tImg_anno

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