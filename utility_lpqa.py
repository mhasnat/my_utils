import cv2
import numpy as np
import os, glob, shutil
from collections import namedtuple

def create_index_from_file(dir):
    index = faiss.read_index(dir)
    return index


BLevel = namedtuple("BLevel", ['brange', 'bval'])
_blevels = [
    BLevel(brange=range(0, 24), bval=0),
    BLevel(brange=range(23, 47), bval=1),
    BLevel(brange=range(46, 70), bval=2),
    BLevel(brange=range(69, 93), bval=3),
    BLevel(brange=range(92, 116), bval=4),
    BLevel(brange=range(115, 140), bval=5),
    BLevel(brange=range(139, 163), bval=6),
    BLevel(brange=range(162, 186), bval=7),
    BLevel(brange=range(185, 209), bval=8),
    BLevel(brange=range(208, 232), bval=9),
    BLevel(brange=range(231, 256), bval=10),
]

BlurLevel = namedtuple("BlurLevel", ['brange', 'bval'])
_blurlevels = [
    BlurLevel(brange=range(0, 10), bval=0),
    BlurLevel(brange=range(10, 30), bval=1),
    BlurLevel(brange=range(30, 50), bval=2),
    BlurLevel(brange=range(50, 85), bval=3),
    BlurLevel(brange=range(85, 100), bval=4),
    BlurLevel(brange=range(100, 150), bval=5),
    BlurLevel(brange=range(150, 200), bval=6),
    BlurLevel(brange=range(200, 400), bval=7),
    BlurLevel(brange=range(400, 800), bval=8),
    BlurLevel(brange=range(800, 1200), bval=9),
    BlurLevel(brange=range(1200, 1000000), bval=10),
]

def detect_level(h_val):
    h_val = int(h_val)
    for blevel in _blevels:
        if h_val in blevel.brange:
            return blevel.bval
    raise ValueError("Brightness Level Out of Range")

def get_img_avg_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return int(np.average(v.flatten()))           

def detect_blur_level(b_val):
    b_val = int(b_val)
    for blevel in _blurlevels:
        if b_val in blevel.brange:
            return blevel.bval
    raise ValueError("Blur Level Out of Range")
    
def blur_detect(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def grade_image(img_name):
    img = cv2.imread(img_name)
    g1 = detect_level(get_img_avg_brightness(img))
    g2 = detect_blur_level(blur_detect(img))
    return round((g1+3*g2)/4, 2)