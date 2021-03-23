#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np

def get_unicolor_ratio(img, thresh_ratio=0.95, numBins=64):    
    isUniColor = False
    unicolor_pixel_ratio = 0.1

    if np.ndim(img)>2:
        for jj in range(np.ndim(img)):
            hist,bins = np.histogram(img[:,:,jj].ravel(),numBins, [0,256])
            color_pixel_ratio = float(hist[np.argmax(hist)])/sum(hist) # check if a unique color exists too much in the image
            if color_pixel_ratio >= thresh_ratio:
                isUniColor = True
                unicolor_pixel_ratio = color_pixel_ratio
                break
            elif color_pixel_ratio >= unicolor_pixel_ratio:
                unicolor_pixel_ratio = color_pixel_ratio
    else:
        hist,bins = np.histogram(img.ravel(),numBins, [0,256])
        color_pixel_ratio = float(hist[np.argmax(hist)])/sum(hist) # check if a unique color exists too much in the image

        if color_pixel_ratio >= thresh_ratio:
            isUniColor = True
            unicolor_pixel_ratio = color_pixel_ratio

    return unicolor_pixel_ratio

def get_DOLP_normalized_image(im0, im45, im90, im135, normalize=True):
    im_stokes0 = im0.astype(np.float) + im90.astype(np.float)
    im_stokes1 = im0.astype(np.float) - im90.astype(np.float)
    im_stokes2 = im45.astype(np.float) - im135.astype(np.float)
    im_DOLP = np.divide(
        np.sqrt(np.square(im_stokes1) + np.square(im_stokes2)),
        im_stokes0,
        out=np.zeros_like(im_stokes0),
        where=im_stokes0 != 0.0,
    ).astype(np.float)
    im_DOLP = np.clip(im_DOLP, 0.0, 1.0)
    if normalize is True:
        # normalize from [0.0, 1.0] range to [0, 255] range (8 bit)
        im_DOLP = (im_DOLP * 255).astype(np.uint8)
    return im_DOLP 

def get_image_of_polarity_auto(t_file, pol_ang=0):
    if t_file.split('.')[-1] == 'npy':
        image_data = np.load(t_file)
    else:
        image_data = cv2.imread(t_file)[:,:,::-1]
    
    if np.ndim(image_data)>2:
        image_data = image_data[:,:,0]
    
    if image_data.shape[0] > 1024:        
        if pol_ang == 0:
            img = image_data[1::2, 1::2]
        elif pol_ang == 45:
            img = image_data[::2, 1::2]
        elif pol_ang == 90:
            img = image_data[::2, ::2]    
        elif pol_ang == 135:
            img = image_data[1::2, ::2]
    else:
        img = image_data

    return img

def get_image_of_polarity(t_file, pol_ang=0):
    image_data = np.load(t_file)
    if pol_ang == 0:
        img = image_data[1::2, 1::2]
    elif pol_ang == 45:
        img = image_data[::2, 1::2]
    elif pol_ang == 90:
        img = image_data[::2, ::2]    
    elif pol_ang == 135:
        img = image_data[1::2, ::2]
    
    return img

def get_image_0_45_polarity(t_file, op='min'):
    image_data = np.load(t_file)
    im45 = image_data[::2, 1::2]
    im0 = image_data[1::2, 1::2]
    
    if op == 'min':
        img = np.minimum.reduce([im45, im0])
    if op == 'max':
        img = np.maximum.reduce([im45, im0])
    if op == 'avg':    
        img = np.asarray(np.mean([im45, im0], axis=0), dtype=np.uint8)
    return img

def get_glare_reduced_image(image_data):
    im90 = image_data[::2, ::2]
    im45 = image_data[::2, 1::2]
    im135 = image_data[1::2, ::2]
    im0 = image_data[1::2, 1::2]
    glare_reduced_image = np.minimum.reduce([im90, im45, im135, im0])
    return glare_reduced_image

def get_binned_image(image_data, max_px=True, return_stacked=False):
    l1 = image_data[::2, ::2] # im90
    l2 = image_data[1::2, ::2] # im45
    l3 = image_data[::2, 1::2] # im135
    l4 = image_data[1::2, 1::2] # im0

    L = np.stack([l1, l2, l3, l4])
    # print(L.shape)
    if max_px:
        img = L.max(axis=0)
    else:
        img = L.min(axis=0)
    
    if return_stacked:
        return L, img
    else:
        return img

def construct_polarized_to_RGB(image_data):
    p_90 = image_data[::2, ::2] # im90
    p_45 = image_data[1::2, ::2] # im45
    #p_135 = image_data[::2, 1::2] # im135
    p_0 = image_data[1::2, 1::2] # im0

    p_rgb = np.stack([p_0, p_45, p_90], axis=2)
    
    return p_rgb
    
def read_image_auto_binning(t_file, to_RGB=True):
    if t_file.split('.')[-1] == 'npy':
        tim = np.load(t_file)
    else:
        tim = cv2.imread(t_file)[:,:,::-1]
    
    if np.ndim(tim)>2:
        tim = tim[:,:,0]
    
    if tim.shape[0] > 1024:
        if to_RGB:
            tim = construct_polarized_to_RGB(tim)
        else:
            tim = get_binned_image(tim)
    
    return tim

def read_image(t_file, apply_Binning=False):
    if t_file.split('.')[-1] == 'npy':
        tim = np.load(t_file)
    else:
        tim = cv2.imread(t_file)[:,:,::-1]
    
    if np.ndim(tim)>2:
        tim = tim[:,:,::-1]
    
    if apply_Binning:
        tim = get_binned_image(tim)
    
    return tim