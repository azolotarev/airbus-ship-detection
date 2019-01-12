import numpy as np
import pandas as pd
import os
import shutil
import math
import pickle
from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

def get_train_data():
    return pd.read_csv('train_ship_segmentations.csv')

train = get_train_data()

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def find_bbox(mask):
    """
    input: mask
    return: coordinates of bounding box
    """
    y, x = np.nonzero(rle_decode(mask))
    coor = []

    idx = np.argmax(x)
    maxy = y[idx]
    # for y_idx in range(idx, idx+len(y[idx:])):
    #     if y[y_idx] > maxy and x[y_idx] == x.max():
    #         maxy = y[y_idx]

    idx = np.argmin(x)
    miny = y[idx]
    # for y_idx in range(idx, idx+len(y[idx:])):
    #     if y[y_idx] < miny and x[y_idx] == x.min():
    #         miny = y[y_idx]

    idx = np.argmax(y)
    minx = x[idx]
    # for x_idx in range(idx, idx+len(x[idx:])):
    #     if x[x_idx] < minx and y[x_idx] == y.max():
    #         minx = x[x_idx]

    idx = np.argmin(y)
    maxx = x[idx]
    # for x_idx in range(idx, idx+len(x[idx:])):
    #     if x[x_idx] > maxx and y[x_idx] == y.min():
    #         maxx = x[x_idx]
    coor.append([x.max(), maxy])
    coor.append([minx, y.max()])
    coor.append([x.min(), miny])
    coor.append([maxx, y.min()])
    return coor

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_bboxes():
    """
    Plots original image, mask and image + mask
    """
    boxes = {}
    masks = train[train['EncodedPixels'].notnull()]
    count = len(masks['ImageId'])
    
    for ImageId in masks['ImageId'][:100]:
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        box = []
        if type(img_masks[0]) == str:
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle_decode(mask)
                y, x = np.nonzero(rle_decode(mask))
                coor = find_bbox(mask)
                box.append(coor)
            boxes[ImageId] = box
        print(len(boxes))
    save_obj(boxes, 'dict')
    print('Done!')

get_bboxes()