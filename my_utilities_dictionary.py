#!/usr/bin/env python
# coding: utf-8

#import cv2
#import numpy as np

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