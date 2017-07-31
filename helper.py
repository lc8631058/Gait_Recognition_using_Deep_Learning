import os

import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
import os
from numpy import genfromtxt
import tensorflow as tf
from PIL import Image
import pickle

def generate_data_list(path, subjects):
    """
    生成每个subject的所有图片
    """
    for subject in subjects:
        subject_path = os.path.join(path, subject)
        images = os.listdir(subject_path)
        yield (images, subject)
		
def generate_degree_list(gallery_path, images):
    """
    生成每个subject的四个不同的degree的图片, 列在一起 
    """
    list_55 = []
    list_65 = []
    list_75 = []
    list_85 = []
    
    for image in images[0]:
        image_path = os.path.join(gallery_path, images[1], image)
        if image[-6:] == '55.png':
            list_55.append(imread(image_path))
        elif image[-6:] == '65.png':
            list_65.append(imread(image_path))
        elif image[-6:] == '75.png':
            list_75.append(imread(image_path))
        elif image[-6:] == '85.png':
            list_85.append(imread(image_path))
    list_total = []
    
    for list_i in [list_55, list_65, list_75, list_85]:
        if len(list_i) != 0:
            list_total.append(list_i)
    
    return list_total

def generate_degree_normalized_list(gallery_path, images):
    """
    生成每个subject的四个不同的degree的图片, 列在一起 
    """
    list_55 = []
    list_65 = []
    list_75 = []
    list_85 = []
    
    for image in images[0]:
        image_path = os.path.join(gallery_path, images[1], image)
        if image[-6:] == '55.png':
            list_55.append((imread(image_path)/255).round(decimals=3))
        elif image[-6:] == '65.png':
            list_65.append((imread(image_path)/255).round(decimals=3))
        elif image[-6:] == '75.png':
            list_75.append((imread(image_path)/255).round(decimals=3))
        elif image[-6:] == '85.png':
            list_85.append((imread(image_path)/255).round(decimals=3))
    list_total = []
    
    for list_i in [list_55, list_65, list_75, list_85]:
        if len(list_i) != 0:
            list_total.append(list_i)
    
    return list_total
	
def generate_subject_list(subjects, gallery_path, subject_iteration):
    """
    生成一个list, 包含所有subjects的images 
    """
    subject_list = []
    for subject in subjects:
        list_total = generate_degree_list(gallery_path, next(subject_iteration))
        subject_list.append(list_total)
    return subject_list

	
def generate_subject_normalized_list(subjects, gallery_path, subject_iteration):
    """
    生成一个list, 包含所有subjects的images 
    """
    subject_list = []
    for subject in subjects:
        list_total = generate_degree_normalized_list(gallery_path, next(subject_iteration))
        subject_list.append(list_total)
    return subject_list