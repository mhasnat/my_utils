import numpy as np
import cv2

def check_difference_len(text_1, text_2):
    if abs(len(text_1)-len(text_2))>0:
        return False    
    else:
        return True
    
def check_difference_validity(t_list):
    # '- 0', '+ 0', '- O', '+ O', '- I', '+ I', '- 1', '+ 1', '- Q', '+ Q' \
    diff_list_accept = ['- B', '+ B', '- 8', '+ 8', '- 2', '+ 2', '- Z', '+ Z']

    output_list = [li for li in difflib.ndiff(t_list[0], t_list[1]) if li[0] != ' ']
    output_list = [t_list for t_list in output_list if t_list not in diff_list_accept]
    if len(output_list) == 0:
        return False
    else:
        return True  
    
def analyze_difference(t_list):
    diff_list_accept = ['- 0', '+ 0', '- O', '+ O', '- I', '+ I', '- 1', '+ 1', '- Q', '+ Q' \
                        '- B', '+ B', '- 8', '+ 8']

    output_list = [li for li in difflib.ndiff(t_list[0], t_list[1]) if li[0] != ' ']
    output_list = [t_list for t_list in output_list if t_list not in diff_list_accept]
    return output_list    

###########################
## Specific to wpod LPCR ##
###########################
from wpod_utils import nms
from label import Label, dknet_label_conversion

def get_matched_text(det_res, img_shape, ref_text):
    th_list = np.arange(0.1, 0.9, 0.1)
    all_text = []
    all_dist = []
    for ii, t_det_th in enumerate(th_list):
        t_text = get_lp_text_from_wpod_detections(det_res, img_shape, det_th=t_det_th)
        all_text.append(t_text)
        all_dist.append(textdistance.hamming(ref_text, t_text))

    final_text = all_text[np.argmin(all_dist)]
    final_dist = all_dist[np.argmin(all_dist)]
    final_th = th_list[np.argmin(all_dist)]

    return final_text, final_dist, final_th

def get_lp_text_from_wpod_detections(det_, im_size, det_th=None):
    if det_th is not None:
        det_ = [t_ for t_ in det_ if t_[1]>det_th]
        
    WH = np.array([im_size[1],im_size[0]],dtype=float)

    L  = []
    for r in det_:    
        r_ = change_det_info(r[2])
        center = np.array(r_[:2])/WH
        wh2 = (np.array(r_[2:])/WH)*.5
        L.append(Label(ord(r[0]),tl=center-wh2,br=center+wh2,prob=r[1]))

    ## Apply NMS    
    L_ = nms(L, .45)    

    L.sort(key=lambda x: x.tl()[0])
    lp_str = ''.join([chr(l.cl()) for l in L])

    return lp_str

def change_det_info(r):
    x1, y1, x2, y2 = r
    
    wd = (x2-x1)
    ht = (y2-y1)
    cx = x1 + (wd/2)
    cy = y1 + (ht/2)

    return (cx, cy, wd, ht)

def check_sample_usefulness(det_, im_shape, gt_anno):
    useful = False
    sel_det = []
    for th_val in np.arange(0.1, 0.9, 0.05):
        t_det = [t_ for t_ in det_ if t_[1]>th_val]
        text_ = get_lp_text_from_wpod_detections(t_det, im_shape)
        #print(text_, t_anno==text_)
        if gt_anno==text_:
            useful = True
            sel_det = t_det
            break
    return useful, t_det