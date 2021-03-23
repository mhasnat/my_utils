#import os, glob, shutil, cv2, time
from collections import defaultdict
import torch
from torch import nn

from collections import OrderedDict
from torch.jit.annotations import Tuple, List, Dict, Optional

from voc_utils_incremental import get_voc

import transforms as T
from copy import deepcopy
import numpy as np

def get_losses_incremental_dbg(model_T, model_S, img_, tar_, return_details=False, 
                   return_distill_info=False, 
                   distill_info=None,
                          verbose=False):
    
    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in img_:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    ## Transform images
    images_, targets_ = model_S.transform(img_, tar_)
    
    ###################################################################
    ## Extract Features and Compute Features-based distillation loss ##
    ###################################################################
        
    # Features from the Student Model        
    features_S = model_S.backbone(images_.tensors)    
    
    
    #########################################
    ## Extract Region Proposals based loss ##
    #########################################
    # TODO : Update based on appropriate loss
    
    # Extract Region Proposals by student model
    # TODO: Not Necessary - Just set the proposals as None is sufficient
    if isinstance(features_S, torch.Tensor):
        features_S = OrderedDict([('0', features_S)])
    proposals_S, proposal_losses_S = model_S.rpn(images_, features_S, targets_, return_details=return_details)
    #print('Number of proposals (Student): ',len(proposals_S[0]))
        
    objectness_S = proposal_losses_S['objectness_details']
    prop_box_S = proposal_losses_S['pred_bbox_deltas_details']

    # The following will extract the classification and bounding box regression scores/values based on all 
    # proposals of the Student model. This is for the object detection loss.
    _, detector_losses_OD_S = model_S.roi_heads(features_S, proposals_S,
                                                images_.image_sizes, targets_,
                                                return_details=return_details,
                                                verbose_inc = False
                                                )    
    
    loss_dict_od = {
            "loss_od_objectness": proposal_losses_S['loss_objectness'],
            "loss_od_rpn_box_reg": proposal_losses_S['loss_rpn_box_reg'],
            "loss_od_classifier": detector_losses_OD_S['loss_classifier'],
            "loss_od_box_reg": detector_losses_OD_S['loss_box_reg']
        }
    
    return loss_dict_od

def get_dataset_incremental(name, image_set, transform, data_path, 
                CLASSES, incremental_classes=None, use_old_data=False):
    num_classes = len(CLASSES)    
    if incremental_classes is not None:
        num_classes += len(incremental_classes)    
        
    ds = get_voc(data_path, image_set=image_set, transforms=transform, 
               CLASSES=CLASSES, incremental_classes=incremental_classes, use_old_data=use_old_data)
    return ds, num_classes

def get_losses(model, img_, tar_, return_details=False):
    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in img_:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    ## Extract image features
    images_, targets_ = model.transform(img_, tar_)
    features_ = model.backbone(images_.tensors)

    ## Extract Region Proposals
    if isinstance(features_, torch.Tensor):
        features_ = OrderedDict([('0', features_)])
    proposals_, proposal_losses_ = model.rpn(images_, features_, targets_, return_details=return_details)

    ## Obtain Detections ..
    detections_, detector_losses = model.roi_heads(features_, proposals_, 
                                                   images_.image_sizes, targets_,
                                                  return_details=return_details)

    loss_doctionary = {}
    loss_doctionary.update(proposal_losses_)
    loss_doctionary.update(detector_losses)

    return loss_doctionary

def get_incremental_model(init_type, model, num_class_to_add):
    if init_type == 'random':
        model = get_incremental_model_v2(deepcopy(model), num_class_to_add=num_class_to_add)
    elif init_type == 'zero-forced':
        model = get_incremental_model_v1(deepcopy(model), num_class_to_add=num_class_to_add)
    else:
        print('Unknown initialization type for the incremental model ..')
    return model
    
def get_incremental_model_v2(model, num_class_to_add=1):
    # Modify the Linear classifiers : cls_score
    cls_score_old = model.roi_heads.box_predictor.cls_score
    cls_score_new = nn.Linear(in_features=cls_score_old.in_features, 
                           out_features=cls_score_old.out_features+num_class_to_add,
                           bias=True)
    cls_score_new.bias[:-num_class_to_add] = cls_score_old.bias
    cls_score_new.bias = nn.Parameter(cls_score_new.bias)
    
    cls_score_new.weight[:-num_class_to_add, ] = cls_score_old.weight
    cls_score_new.weight = nn.Parameter(cls_score_new.weight)
    
    model.roi_heads.box_predictor.cls_score = cls_score_new
    
    # Modify the Linear classifiers : bbox_pred
    bbox_pred_old = model.roi_heads.box_predictor.bbox_pred
    bbox_pred_new = nn.Linear(in_features=bbox_pred_old.in_features, 
                           out_features=bbox_pred_old.out_features+(num_class_to_add*4), # 4 - for 4 box related parameters
                           bias=True)
    bbox_pred_new.bias[:-(num_class_to_add*4)] = bbox_pred_old.bias
    bbox_pred_new.bias = nn.Parameter(bbox_pred_new.bias)
    bbox_pred_new.weight[:-(num_class_to_add*4), ] = bbox_pred_old.weight
    bbox_pred_new.weight = nn.Parameter(bbox_pred_new.weight)
    
    model.roi_heads.box_predictor.bbox_pred = bbox_pred_new
    
    return model

def get_incremental_model_v1(model, num_class_to_add=1):
    # Modify the Linear classifiers : cls_score
    cls_score_old = model.roi_heads.box_predictor.cls_score    
    cls_score_new = nn.Linear(in_features=cls_score_old.in_features, 
                       out_features=cls_score_old.out_features+num_class_to_add,
                       bias=True)
    
    cl_bias_old = cls_score_old.bias
    bias_shape = cl_bias_old.shape
    cl_bias_new = torch.zeros(bias_shape[0]+num_class_to_add)
    cl_bias_new = torch.nn.init.normal_(cl_bias_new, mean=0, std=0.1)
    cl_bias_new[:-num_class_to_add] = cl_bias_old
    cls_score_new.bias = nn.Parameter(cl_bias_new)
    
    # Modify - roi_heads.box_predictor.cls_score.weight
    cl_weight_old = cls_score_old.weight
    weight_shape = cl_weight_old.shape
    cl_weight_new = torch.zeros((weight_shape[0]+num_class_to_add, weight_shape[1]))
    cl_weight_new = torch.nn.init.normal_(cl_weight_new, mean=0, std=0.1)
    cl_weight_new[:-num_class_to_add, ] = cl_weight_old
    cls_score_new.weight = nn.Parameter(cl_weight_new)
    ##
    model.roi_heads.box_predictor.cls_score = cls_score_new
    
    
    # Modify the Linear classifiers : bbox_pred
    bbox_pred_old = model.roi_heads.box_predictor.bbox_pred
    bbox_pred_new = nn.Linear(in_features=bbox_pred_old.in_features, 
                           out_features=bbox_pred_old.out_features+(num_class_to_add*4), # 4 - for 4 box related parameters
                           bias=True)

    # Modify - roi_heads.box_predictor.bbox_pred.bias
    box_bias_old = bbox_pred_old.bias
    bias_shape = box_bias_old.shape
    box_bias_new = torch.zeros(bias_shape[0]+(num_class_to_add*4)) # 4 - for 4 box related parameters
    box_bias_new = torch.nn.init.normal_(box_bias_new, mean=0, std=1)
    box_bias_new[:-(num_class_to_add*4)] = box_bias_old
    bbox_pred_new.bias = nn.Parameter(box_bias_new)

    # Modify - roi_heads.box_predictor.bbox_pred.weight
    box_weight_old = bbox_pred_old.weight
    weight_shape = box_weight_old.shape
    box_weight_new = torch.zeros((weight_shape[0]+(num_class_to_add*4), weight_shape[1])) # 4 - for 4 box related parameters
    box_weight_new = torch.nn.init.normal_(box_weight_new, mean=0, std=1)
    box_weight_new[:-(num_class_to_add*4), ] = box_weight_old
    bbox_pred_new.weight = nn.Parameter(box_weight_new)
    ##
    model.roi_heads.box_predictor.bbox_pred = bbox_pred_new
    
    return model

def get_incremental_model_v0(model, num_class_to_add=1):
    # Modify - roi_heads.box_predictor.cls_score.bias
    cl_bias_old = model.roi_heads.box_predictor.cls_score.bias
    bias_shape = cl_bias_old.shape
    cl_bias_new = torch.zeros(bias_shape[0]+num_class_to_add)
    cl_bias_new = torch.nn.init.normal_(cl_bias_new, mean=0, std=1)
    cl_bias_new[:-num_class_to_add] = cl_bias_old
    model.roi_heads.box_predictor.cls_score.bias = nn.Parameter(cl_bias_new)

    # Modify - roi_heads.box_predictor.cls_score.weight
    cl_weight_old = model.roi_heads.box_predictor.cls_score.weight
    weight_shape = cl_weight_old.shape
    cl_weight_new = torch.zeros((weight_shape[0]+num_class_to_add, weight_shape[1]))
    cl_weight_new = torch.nn.init.normal_(cl_weight_new, mean=0, std=1)
    cl_weight_new[:-num_class_to_add, ] = cl_weight_old
    model.roi_heads.box_predictor.cls_score.weight = nn.Parameter(cl_weight_new)

    # Modify - roi_heads.box_predictor.bbox_pred.bias
    box_bias_old = model.roi_heads.box_predictor.bbox_pred.bias
    bias_shape = box_bias_old.shape
    box_bias_new = torch.zeros(bias_shape[0]+(num_class_to_add*4)) # 4 - for 4 box related parameters
    box_bias_new = torch.nn.init.normal_(box_bias_new, mean=0, std=1)
    box_bias_new[:-(num_class_to_add*4)] = box_bias_old
    model.roi_heads.box_predictor.bbox_pred.bias = nn.Parameter(box_bias_new)

    # Modify - roi_heads.box_predictor.cls_score.weight
    box_weight_old = model.roi_heads.box_predictor.bbox_pred.weight
    weight_shape = box_weight_old.shape
    box_weight_new = torch.zeros((weight_shape[0]+(num_class_to_add*4), weight_shape[1])) # 4 - for 4 box related parameters
    box_weight_new = torch.nn.init.normal_(box_weight_new, mean=0, std=1)
    box_weight_new[:-(num_class_to_add*4), ] = box_weight_old
    model.roi_heads.box_predictor.bbox_pred.weight = nn.Parameter(box_weight_new)
    
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_dataset(name, image_set, transform, data_path, CLASSES):
    # print(image_set)
    paths = {
        #"coco": (data_path, get_coco, 91),
        #"coco_kp": (data_path, get_coco_kp, 2),
        "voc": (data_path, get_voc, 21)
    }
    p, ds_fn, num_classes = paths[name]
    num_classes = len(CLASSES)
    
    ds = ds_fn(p, image_set=image_set, transforms=transform, CLASSES=CLASSES)
    return ds, num_classes

def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def modify_target(t_, device):
    t_mod = t_
    if len(t_mod['boxes']) == 0:
        t_mod = t_
        #print(t_)
        t_mod["boxes"] = torch.zeros((0, 4), dtype=torch.float32).to(device)
        t_mod["labels"] = torch.zeros((1, 1), dtype=torch.int64).to(device)
        t_mod["ishard"] = torch.ones((1, 1), dtype=torch.int64).to(device)
        #print(t_)
    return t_mod  

def filter_labels(target, CL_to_delete, CL_to_keep, device, verbose=False):
    target_ = deepcopy(target)
    if verbose:
        print(target_)
    cl_to_delete = deepcopy(CL_to_delete)
    cl_to_keep = deepcopy(CL_to_keep)
    
    cl_all = CL_to_delete + CL_to_keep

    # select which indcides to keep
    keep_index = [cl_all.index(t_) for t_ in cl_to_keep]

    classes = []
    boxes = []
    ishard = []

    for tt in range(len(target_["labels"])):
        t_label = target_["labels"].cpu().numpy()[tt]
        if t_label in keep_index:
            boxes.append(target_["boxes"].cpu().numpy()[tt])
            classes.append(t_label)            
            ishard.append(target_["ishard"].cpu().numpy()[tt])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    classes = torch.as_tensor(classes)
    ishard = torch.as_tensor(ishard)

    target_["boxes"] = boxes.to(device)
    target_["labels"] = classes.to(device)
    target_["ishard"] = ishard.to(device)
    if verbose:
        print(target_)
        
    return target_

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

def get_pseudo_labels(psl_constrain_type, op, targets, device, verbose=False):
    if psl_constrain_type == 'relaxed':
        ps_targets = get_pseudo_labels_relaxed(op, targets, device, min_conf=0.1, 
                                               append_with_target=True)
    elif psl_constrain_type == 'iv-2018':
        ps_targets = get_pseudo_labels_constrained_3(op, targets, device, min_conf=0.5, th_iou = 0.5,
                                                     append_with_target=True,
                                                     return_score_conf=True,
                                                     apply_constraints=True,
                                                     verbose=verbose
                                                    )
        
    elif psl_constrain_type == 'strong':
        ps_targets = get_pseudo_labels_constrained_2(op, targets, device, min_conf=0.5,
                                                     append_with_target=True,
                                                     return_score_conf=True,
                                                     apply_constraints=True,
                                                     th_very_high_iou=0.9,
                                                     th_medium_iou=0.2,
                                                     th_confidence=0.9,
                                                     verbose=verbose
                                                    )
    else:
        print('Unknown pseudo-labeling type for the incremental model ..')
        ps_targets = targets
        
    return ps_targets
                      
def get_pseudo_labels_relaxed(output_, target_, device, 
                      min_conf=0.5, 
                      append_with_target=False,
                      return_score_conf = False):
    t_conf = output_['scores'].detach().cpu().numpy()
    t_labels = output_['labels'].detach().cpu().numpy()
    t_boxes = output_['boxes'].detach().cpu().numpy()

    tindx = np.where(t_conf >= min_conf)[0]
    t_boxes = t_boxes[tindx]
    t_labels = t_labels[tindx]
    t_conf = t_conf[tindx]

    boxes = torch.as_tensor(t_boxes, dtype=torch.float32)
    classes = torch.as_tensor(t_labels, dtype=torch.long)
    ishard = torch.as_tensor(np.int32(t_conf), dtype=torch.long)
    score_conf = torch.as_tensor(t_conf, dtype=torch.float32)
    
    ps_label = deepcopy(target_)
    if append_with_target:
        ps_label['name'] = target_['name']
        ps_label["boxes"] = torch.cat((target_["boxes"], boxes.to(device)))
        ps_label["labels"] = torch.cat((target_["labels"].to(torch.long), classes.to(device)))    
        ps_label["ishard"] = torch.cat((target_["ishard"].to(torch.long), ishard.to(device)))             
    else:        
        ps_label['name'] = target_['name']
        ps_label["boxes"] = boxes.to(device)
        ps_label["labels"] = classes.to(device)
        ps_label["ishard"] = ishard.to(device)
        
        if return_score_conf:
            ps_label["score_conf"] = score_conf.to(device)
            
    return ps_label

def get_pseudo_labels_constrained(output_, target_, device, 
                                  min_conf=0.5,
                                  append_with_target=False,
                                  return_score_conf = False,
                                  apply_constraints = False,
                                  th_very_high_iou = 0.9,
                                  th_medium_iou = 0.6,
                                  th_confidence = 0.9,
                                  verbose=False
                                  ):
    ## information about pseudo-labels
    t_conf = output_['scores'].detach().cpu().numpy()
    t_labels = output_['labels'].detach().cpu().numpy()
    t_boxes = output_['boxes'].detach().cpu().numpy()

    ## pseudo-labels filtering based on score confidences
    tindx = np.where(t_conf >= min_conf)[0]
    t_boxes = t_boxes[tindx]
    t_labels = t_labels[tindx]
    t_conf = t_conf[tindx]

    if verbose:
        print('_confs_ :')
        print(t_conf)

    if apply_constraints:
        ## pseudo-labels filtering based on ground-truth constrains
        gt_boxes = target_['boxes'].detach().cpu().numpy()

        # compute iou's among the ground-truths and pseudo-boxes
        t_ious = np.ones((len(t_boxes), len(gt_boxes))) * -1
        for pp in range(len(t_boxes)):
            for qq in range(len(gt_boxes)):
                    t_ious[pp,qq] = box_iou_ratio(t_boxes[pp], gt_boxes[qq])
        
        if verbose:
            print('_ious_ :')
            print(t_ious)
        # sum of ious
        sum_ious = np.sum(t_ious, axis=1)

        # check validities - based on applied constrains
        is_valid = np.ones((len(t_boxes)), dtype=np.bool)
        for tt in range(len(t_boxes)):
            if sum_ious[tt] >= th_very_high_iou:
                is_valid = False
            elif (sum_ious[tt] >= th_medium_iou) and t_conf[tt]<=th_confidence:
                is_valid = False

        if verbose:
            print('validity of pseudo labels :')
            print(is_valid)

        # select only the valid boxes
        tindx = np.where(is_valid == True)[0]
        t_boxes = t_boxes[tindx]
        t_labels = t_labels[tindx]
        t_conf = t_conf[tindx]
    
    ## construct tensors for pseudo labels
    boxes = torch.as_tensor(t_boxes, dtype=torch.float32)
    classes = torch.as_tensor(t_labels, dtype=torch.long)
    ishard = torch.as_tensor(np.int32(t_conf), dtype=torch.long)
    score_conf = torch.as_tensor(t_conf, dtype=torch.float32)
    
    ps_label = deepcopy(target_)
    if append_with_target:
        ps_label['name'] = target_['name']
        ps_label["boxes"] = torch.cat((target_["boxes"], boxes.to(device)))
        ps_label["labels"] = torch.cat((target_["labels"].to(torch.long), classes.to(device)))    
        ps_label["ishard"] = torch.cat((target_["ishard"].to(torch.long), ishard.to(device)))                            
        
        if return_score_conf:
            target_score_conf = torch.ones((len(target_["ishard"]))).to(device)
            ps_label["score_conf"] = torch.cat((target_score_conf, score_conf.to(device)))
    else:        
        ps_label['name'] = target_['name']
        ps_label["boxes"] = boxes.to(device)
        ps_label["labels"] = classes.to(device)
        ps_label["ishard"] = ishard.to(device)
        
        if return_score_conf:
            ps_label["score_conf"] = score_conf.to(device)
            
    return ps_label

def get_pseudo_labels_constrained_2(output_, target_, device, 
                                  min_conf=0.5,
                                  append_with_target=False,
                                  return_score_conf = False,
                                  apply_constraints = False,
                                  th_very_high_iou = 0.9,
                                  th_medium_iou = 0.6,
                                  th_confidence = 0.9,
                                  verbose=False
                                  ):
    ## information about pseudo-labels
    t_conf = output_['scores'].detach().cpu().numpy()
    t_labels = output_['labels'].detach().cpu().numpy()
    t_boxes = output_['boxes'].detach().cpu().numpy()

    ## pseudo-labels filtering based on score confidences
    tindx = np.where(t_conf >= min_conf)[0]
    t_boxes = t_boxes[tindx]
    t_labels = t_labels[tindx]
    t_conf = t_conf[tindx]

    if verbose:
        print('_confs_ :')
        print(t_conf)

    if apply_constraints:
        ## pseudo-labels filtering based on ground-truth constrains
        gt_boxes = target_['boxes'].detach().cpu().numpy()

        # compute iou's among the ground-truths and pseudo-boxes
        t_ious = np.ones((len(t_boxes), len(gt_boxes))) * -1
        for pp in range(len(t_boxes)):
            for qq in range(len(gt_boxes)):
                    t_ious[pp,qq] = box_iou_ratio(t_boxes[pp], gt_boxes[qq])
        
        if verbose:
            print('_ious_ :')
            print(t_ious)
        # sum of ious
        sum_ious = np.sum(t_ious, axis=1)

        # check validities - based on applied constrains
        is_valid = np.ones((len(t_boxes)), dtype=np.bool)
        for tt in range(len(t_boxes)):
            if sum_ious[tt] >= th_very_high_iou:
                is_valid = False
            elif (sum_ious[tt] >= th_medium_iou) and t_conf[tt]<=th_confidence:
                is_valid = False

        if verbose:
            print('validity of pseudo labels :')
            print(is_valid)

        # select only the valid boxes
        tindx = np.where(is_valid == True)[0]
        t_boxes = t_boxes[tindx]
        t_labels = t_labels[tindx]
        t_conf = t_conf[tindx]
    
    ## construct tensors for pseudo labels
    boxes = torch.as_tensor(t_boxes, dtype=torch.float32)
    classes = torch.as_tensor(t_labels, dtype=torch.long)
    ishard = torch.as_tensor(np.int32(t_conf), dtype=torch.long)
    score_conf = torch.as_tensor(t_conf, dtype=torch.float32)
    
    ps_label = deepcopy(target_)
    if append_with_target:
        ps_label['name'] = target_['name']
        ps_label["boxes"] = torch.cat((target_["boxes"], boxes.to(device)))
        ps_label["labels"] = torch.cat((target_["labels"].to(torch.long), classes.to(device)))    
        ps_label["ishard"] = torch.cat((target_["ishard"].to(torch.long), ishard.to(device)))                            
        
        if return_score_conf:
            target_score_conf = torch.ones((len(target_["ishard"]))).to(device)
            ps_label["score_conf"] = torch.cat((target_score_conf, score_conf.to(device)))
    else:        
        ps_label['name'] = target_['name']
        ps_label["boxes"] = boxes.to(device)
        ps_label["labels"] = classes.to(device)
        ps_label["ishard"] = ishard.to(device)
        
        if return_score_conf:
            ps_label["score_conf"] = score_conf.to(device)
            
    return ps_label

def get_pseudo_labels_constrained_3(output_, target_, device, 
                                    min_conf=0.5, 
                                    th_iou = 0.5,
                                    append_with_target=False,
                                    return_score_conf = False,
                                    apply_constraints = False,
                                    verbose=False):
    ## information about pseudo-labels
    t_conf = output_['scores'].detach().cpu().numpy()
    t_labels = output_['labels'].detach().cpu().numpy()
    t_boxes = output_['boxes'].detach().cpu().numpy()

    ## pseudo-labels filtering based on score confidences
    tindx = np.where(t_conf >= min_conf)[0]
    t_boxes = t_boxes[tindx]
    t_labels = t_labels[tindx]
    t_conf = t_conf[tindx]

    if verbose:
        print('_confs_ :')
        print(t_conf)

    if apply_constraints:
        ## pseudo-labels filtering based on ground-truth constrains
        gt_boxes = target_['boxes'].detach().cpu().numpy()

        # compute iou's among the ground-truths and pseudo-boxes
        t_ious = np.ones((len(t_boxes), len(gt_boxes))) * -1
        for pp in range(len(t_boxes)):
            for qq in range(len(gt_boxes)):
                    t_ious[pp,qq] = box_iou_ratio(t_boxes[pp], gt_boxes[qq])
        
        if verbose:
            print('_ious_ :')
            print(t_ious)
            
        # sum of ious
        max_ious = np.max(t_ious, axis=1)

        # check validities - based on applied constrains - IoU > 0.5
        is_valid = np.ones((len(t_boxes)), dtype=np.bool)
        for tt in range(len(t_boxes)):
            if max_ious[tt] >= th_iou:
                is_valid = False

        if verbose:
            print('validity of pseudo labels :')
            print(is_valid)

        # select only the valid boxes
        tindx = np.where(is_valid == True)[0]
        t_boxes = t_boxes[tindx]
        t_labels = t_labels[tindx]
        t_conf = t_conf[tindx]
    
    ## construct tensors for pseudo labels
    boxes = torch.as_tensor(t_boxes, dtype=torch.float32)
    classes = torch.as_tensor(t_labels, dtype=torch.long)
    ishard = torch.as_tensor(np.int32(t_conf), dtype=torch.long)
    score_conf = torch.as_tensor(t_conf, dtype=torch.float32)
    
    ps_label = deepcopy(target_)
    if append_with_target:
        ps_label['name'] = target_['name']
        ps_label["boxes"] = torch.cat((target_["boxes"], boxes.to(device)))
        ps_label["labels"] = torch.cat((target_["labels"].to(torch.long), classes.to(device)))    
        ps_label["ishard"] = torch.cat((target_["ishard"].to(torch.long), ishard.to(device)))                            
        
        if return_score_conf:
            target_score_conf = torch.ones((len(target_["ishard"]))).to(device)
            ps_label["score_conf"] = torch.cat((target_score_conf, score_conf.to(device)))
    else:        
        ps_label['name'] = target_['name']
        ps_label["boxes"] = boxes.to(device)
        ps_label["labels"] = classes.to(device)
        ps_label["ishard"] = ishard.to(device)
        
        if return_score_conf:
            ps_label["score_conf"] = score_conf.to(device)
            
    return ps_label

def show_labels(targets_, CL):
    # Labels
    t_targ = targets_['labels'].cpu().numpy()
    t_labels = []
    for t_ in t_targ:
        t_labels.append(CL[t_])
    print(t_labels)    