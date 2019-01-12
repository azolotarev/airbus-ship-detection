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

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dbox = load_obj('dict')
#print(dbox['000155de5.jpg'])

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def show_train_data():
    """
    Plots original image, mask and image + mask
    """
    masks = train
    for ImageId in os.listdir('train_positives/train/class1'):
        img = imread('train_positives/train/class1/' + ImageId)
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        if type(img_masks[0]) == str:
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle_decode(mask)
            fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[2].axis('off')
            axarr[1].imshow(all_masks)
            axarr[2].imshow(img)
            axarr[2].imshow(all_masks, alpha=0.4)
            if ImageId in dbox:
                for coor in dbox[ImageId]:
                    print(coor)
                    cpointx = (coor[0][0] + coor[2][0])//2
                    cpointy = (coor[0][1] + coor[2][1])//2
                    boxh = math.ceil(distance(coor[0], coor[3]))
                    boxw =  math.ceil(distance(coor[0], coor[2]))
                    radians = math.atan2(coor[1][1] - coor[3][1], coor[1][0] - coor[3][0])
                    degree = math.degrees(radians)
                    #rect = patches.Rectangle(tuple(coor[1]), boxw, boxh, round(degree-180))
                    poly = patches.Polygon(coor)
                    axarr[1].plot(cpointx, cpointy, 'ro')
                    axarr[1].plot(coor[0][0], coor[0][1], 'go')
                    axarr[1].plot(coor[1][0], coor[1][1], 'bo')
                    axarr[1].plot(coor[2][0], coor[2][1], 'gx')
                    axarr[1].plot(coor[3][0], coor[3][1], 'bx')
                    axarr[1].add_patch(poly)
            plt.tight_layout(h_pad=0.1, w_pad=0.1)
            #plt.savefig('mask.png')
            plt.show()
        else:
            print('NaN')

show_train_data()

"""
Train
Input: X - image, Y - masks. Metric: IoU

Prediction
Input: X - image. Output: Y - masks. Metric: IoU

First, I have to understand how to draw a rotatable BBox. How to specify it's region and angle of rotation.
Then I should think how to encode that data to create a submission.

For now output of the NN should be bboxes. Then I'll create an encoded submission file.

Bug: the boxes are a bit fucked for small masks

TODO: convert rboxes to RLE

"""

