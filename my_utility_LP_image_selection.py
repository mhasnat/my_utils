import cv2, math
import numpy as np
import sys
sys.path.append('/raid-dgx2/Hasnat/ImQual/')

from skimage import feature as ced
from skimage import transform

from BRISQUEE_features import *
from Hand_Crafted_Features import *
brisq = BRISQUE()
hcf = Hand_Crafted_Features()

#### Related to LP text  ###
def include_delimiter_to_lp_text(my_text, delim_):
    if get_pattern(my_text) in ['CCDDDCC','CCDDCC','DDCCCDD','DDCCDD',]:
        my_text_w_sp = my_text[:2] + delim_ + my_text[2:-2] + delim_ + my_text[-2:]
    elif get_pattern(my_text) == 'DDDDCCDD':
        my_text_w_sp = my_text[:4] + delim_ + my_text[4:-2] + delim_ + my_text[-2:]
    elif get_pattern(my_text) in ['DDDDCCDD', 'DDDDCCCDD']:
        my_text_w_sp = my_text[:4] + delim_ + my_text[4:-2] + delim_ + my_text[-2:]    
    elif get_pattern(my_text) in ['DDDCCCDD', 'DDDCCDD', 'CCCDDDCC']:        
        my_text_w_sp = my_text[:3] + delim_ + my_text[3:-2] + delim_ + my_text[-2:]

    return my_text_w_sp

def isFrench(my_text):
    french_LP_patterns = ['CCDDDCC','CCDDCC','DDCCCDD','DDCCDD','DDDCCCDD','DDDDCCCDD','DDDCCDD','DDDDCCDD','CCCDDDCC']
    is_fr_lp = get_pattern(my_text) in french_LP_patterns
    return is_fr_lp
        
def get_pattern(plate):
    lp_type =''
    for sample in plate:
        if sample.isdigit():
            lp_type += 'D'
        elif sample.isalpha():
            lp_type += 'C'
    return lp_type


#### Related to Image Quality ###
def get_iq_score_proba(img_orig, model, scaler, imRsz=(280, 120), 
                       feature_type='HC', verbose=False):
    _rsz_H, _rsz_W = (imRsz[1], imRsz[0])
    img_preprocessed = cv2.cvtColor(cv2.resize(img_orig, (_rsz_W, _rsz_H)), cv2.COLOR_BGR2RGB)

    # Extract IQ features
    if feature_type == 'HC':
        t_feature = get_hc_features_score(img_preprocessed)
    elif feature_type == 'HCBR':
        t_feature_brisque = brisq.get_feature(img_preprocessed)
        t_feature_hc = get_hc_features_score(img_preprocessed)
        t_feature = np.hstack((t_feature_brisque, t_feature_hc))        

    if verbose:
        print(t_feature)
        
    if np.isnan(t_feature).any():
        # means problem in features, so we return it as a bad image 
        print('Nan values in IQA features ..')
        prd = [np.array([0.0, 0.0, 0.0, 1.0])]
    else:
        try:
            # Predict IQ
            Xn = scaler.transform(np.expand_dims(np.asarray(t_feature), 0))
            prd = model.predict_proba(Xn)
        except:
            # means problem in prediction, so we return it as a bad image
            print('Problem in IQA prediction ..')
            prd = [np.array([0.0, 0.0, 0.0, 1.0])]

    return prd[0]

def get_hc_features_score(img_orig, imRsz=(280, 120), verbose=False):
    _rsz_H, _rsz_W = (imRsz[1], imRsz[0])
    img_preprocessed = cv2.cvtColor(cv2.resize(img_orig, (_rsz_W, _rsz_H)), cv2.COLOR_BGR2RGB)

    # Extract IQ features
    t_feature_hc = hcf.get_hand_crafted_features_combined(img_preprocessed)        
    return t_feature_hc

## Related to Image Geometry
def get_rotation_angle(image, th_angle=45):
    ## Estimate the edges and lines
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = ced.canny(gray_img, sigma=3)            
    lines = transform.probabilistic_hough_line(edges, line_length=3,line_gap=5)

    ## Calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = 0

    ## Calculate the angle between each line and the horizontal line:
    angle_all = []
    for line in lines:
        (x1,y1),(x2,y2) = line
        t_ang = np.rad2deg(math.atan2(y2*1.0 - y1*1.0, x2*1.0 - x1*1.0))
        if np.abs(t_ang) > 90:
            t_ang = 180-np.abs(t_ang)
        
        if np.abs(t_ang) <= th_angle:
            angle_all.append(t_ang)
        
    op_ang = np.mean(angle_all)
    
    return op_ang

def select_image_with_angle_iqa(all_scores, max_angle_to_normalize=20, verbose=False):
    all_scores = np.asarray(all_scores)
    all_scores[:, 0] = all_scores[:, 0] / max_angle_to_normalize
    all_scores_norm = np.abs(all_scores - np.array([0.0, 1.0]))
    if verbose:
        print(all_scores_norm)
    all_scores_comb = np.sum(all_scores_norm, axis=1)
    sel_indx = np.argmin(all_scores_comb)
    if verbose:
        print((all_scores_comb, sel_indx))
    return sel_indx