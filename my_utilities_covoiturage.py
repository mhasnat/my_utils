import numpy as np
import os, glob
import cv2

def get_images_directories_extension(list_files, ext_='npy', dir_loc=-2):
    # List the extensions and directory names
    list_ext = np.asarray([t_.split('.')[-1] for t_ in list_files])
    list_dir = [t_.split('/')[dir_loc] for t_ in list_files]
    
    num_imgs = len(list_ext)
    print('Number of images :: ' + str(num_imgs))

    # List the raw images
    npy_indx = np.where(list_ext == ext_)[0]
    list_files_sel = np.asarray(list_files)[npy_indx]
    list_dirs_sel = np.asarray(list_dir)[npy_indx]
    
    num_imgs = len(list_files_sel)
    print('Number of Selected Images :: ' + str(num_imgs))

    # List the directories
    list_dirs_unique = list(set(list_dirs_sel))
    num_dir = len(list_dirs_unique)
    print('Number of Selected directories :: ' + str(num_dir))
    
    return list_files_sel, list_dirs_unique