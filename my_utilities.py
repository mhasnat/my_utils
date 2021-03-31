#!/usr/bin/env python
# coding: utf-8
import cv2
import pickle
import numpy as np
import os, glob, shutil
#from sklearn.metrics.pairwise import pairwise_distances

# Utility function: Dictionary
def update_dictionary(sigs_items_DB, item_key, item_value):
    if item_key in sigs_items_DB:
        # add to the value list
        sigs_items_DB.get(item_key).append(item_value)
    else:
        # update dictionary with this item
        sigs_items_DB.update( {item_key : [item_value]} )   
        
def show_dictionary_items(my_dictionary, num_keys=5):
    cnt = 0
    for key_ in my_dictionary.keys():
        print(key_ +' : ' + str(my_dictionary[key_]))
        cnt +=1

        if cnt>num_keys:
            break        

def get_dictionary_items(my_dictionary, num_items=5):
    dict_items = []
    cnt = 0
    for key_ in my_dictionary.keys():
        dict_items.append(key_)
        cnt +=1

        if cnt>num_items:
            break       
    return dict_items

def get_dictionary_items_by_range(my_dictionary, start_index, end_index):
    dict_items = []
    cnt = 0
    
    for key_ in my_dictionary.keys():
        if cnt>=start_index and cnt<=end_index:
            dict_items.append(key_)        

        if cnt>end_index:
            break       
            
        cnt +=1
    return dict_items
            
# Utility function: Distance based
#def get_pair_distance(t_features, dist_type='cosine'):
#    pdistArr = pairwise_distances(t_features , metric=dist_type)  # for LP
#    return pdistArr[np.triu_indices(len(t_features), k=1)]

# Utility function: save as pickel based
def save_data_with_pickel(file_name, info_list):
    with open(file_name, 'wb') as f:
        pickle.dump(info_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Info saved to :' + file_name)
        
def load_data_from_pickel(file_name):
    with open(file_name, 'rb') as f:
        all_info = pickle.load(f)
    return all_info