#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
from copy import deepcopy
import os, glob, shutil
import matplotlib.pyplot as plt
from imutils import build_montages

gray_to_color = lambda a : cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)

def show_image_subplots(tImg_list, rc=(1,1), titles='', cmap=None, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    cnt = 1
    for tImg in tImg_list:
        plt.subplot(rc[0], rc[1], cnt)
        cnt += 1
        
        if np.ndim(tImg)==2:
            tImg = gray_to_color(tImg)
            
        if cmap is None:
            plt.imshow(tImg)
        else:
            plt.imshow(tImg, cmap=cmap)
        
        if titles=='':
            plt.title(cnt-1)
        else:
            plt.title(titles[cnt-2])
            
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def show_image(tImg, cmap=None, figsize=None, title=''):
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    if cmap is None:
        plt.imshow(tImg)
    else:
        plt.imshow(tImg, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    
def show_image_from_file(t_file, cmap=None, figsize=None, title=''):
    tImg = cv2.imread(t_file)[:,:,::-1]
    show_image(tImg, cmap=cmap, figsize=figsize, title=title)
    '''
    plt.imshow(tImg)
    '''

def show_montage_from_image_list(images, all_lp_texts, show_title=True, 
                 saveFig=None, showFig=True, im_shape=(280, 120), dpi=200):
    # construct the montages for the images
    maxNumImagePerRow = 4
    maxNumImagePerRow = min(4, len(images))
    montages = build_montages(images, im_shape, 
                              (maxNumImagePerRow, 
                               np.int(np.ceil(len(images)/maxNumImagePerRow))))[0]

    plt.figure(figsize=(12, 8))
    plt.imshow(montages)
    if show_title:
        print(all_lp_texts)
        plt.title(', '.join(set(all_lp_texts)), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    if saveFig is not None:
        plt.savefig(saveFig, dpi=dpi)
        
    if not showFig:
        plt.close()
    #plt.show()
    
def show_montage(all_file_names, all_lp_texts, show_title=True, 
                 saveFig=None, showFig=True, figsize=(12, 8), imresize=(280, 120), dpi=200):
    # initialize the list of images
    images = []

    # loop over the list of image paths
    for imagePath in all_file_names:
        # load the image and update the list of images
        image = cv2.imread(imagePath)
        images.append(image[:,:,::-1])

    # construct the montages for the images
    maxNumImagePerRow = 4
    maxNumImagePerRow = min(4, len(images))
    montages = build_montages(images, imresize, 
                              (maxNumImagePerRow, 
                               np.int(np.ceil(len(images)/maxNumImagePerRow))))[0]

    plt.figure(figsize=figsize)
    plt.imshow(montages)
    if show_title:
        print(all_lp_texts)
        plt.title(', '.join(set(all_lp_texts)), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    if saveFig is not None:
        plt.savefig(saveFig, dpi=dpi)
        
    if not showFig:
        plt.close()
    #plt.show()

## Image Segmentation
def get_seg_overlayed_image(img, seg, resize=True, show_img=False):
    class_colors = [(0,0,0), (12,76,125), (135,50,50),(13,50,135),(13,155,50) ]

    if seg.ndim==2:
        newSeg = np.dstack((seg,seg,seg))
    else:
        newSeg = seg
        
    for c in range(3):
        newSeg[:,:,0] = np.where(newSeg[:,:,0] == c, class_colors[c][0], newSeg[:,:,0])
        newSeg[:,:,1] = np.where(newSeg[:,:,1] == c, class_colors[c][1], newSeg[:,:,1])
        newSeg[:,:,2] = np.where(newSeg[:,:,2] == c, class_colors[c][2], newSeg[:,:,2])
    
    if resize:
        img = cv2.resize(img, (seg.shape[1],seg.shape[0]), interpolation = cv2.INTER_AREA)
    else:
        newSeg = cv2.resize(np.asarray(newSeg, dtype=np.uint8), (img.shape[1],img.shape[0]))
    
    dst = cv2.addWeighted(deepcopy(img), 0.7, newSeg, 0.5, 0, dtype=cv2.CV_32F)
    
    dst = np.asarray(dst, dtype=np.uint8)
    if show_img:
        show_image(dst)
        
    return dst

## Classifier Results Display
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
