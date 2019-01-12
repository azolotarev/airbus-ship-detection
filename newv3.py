import numpy as np
import pandas as pd
import os
import shutil
import math
import pickle
import cv2
import tensorflow as tf
from skimage.data import imread
from skimage.io import imsave
from skimage.morphology import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, img_to_array

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

def get_masks():
    """
    """
    masks = train[train['EncodedPixels'].notnull()]
    count = len(masks['ImageId'])
    
    for ImageId in masks['ImageId']:
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        box = []
        if type(img_masks[0]) == str:
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle_decode(mask)
            imsave('data/masks/' + ImageId, all_masks)
    print('Done!')

#get_masks()

def move_images():
    masks = train
    images = os.listdir('data/train/class1')
    for ImageId in os.listdir('data/masks/class1'):
        if ImageId in images:
            pass
        else:
            shutil.move('data/masks/class1/' + ImageId,
                        'data/fmask/' + ImageId)

#move_images()

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def model1(input_shape, lr=1e-5):
    X_input = Input(input_shape)

    X = Conv2D(8, (7, 7), activation='relu', padding='same') (X_input)
    X = Conv2D(16, (7, 7), activation='relu', padding='same') (X)
    #X = Conv2D(32, (3, 3), activation='relu', padding='same') (X)
    #X = Conv2D(64, (3, 3), activation='relu', padding='same') (X)
    X = Conv2D(25, (7, 7), activation='relu', padding='same') (X)
    X = Conv2D(16, (2, 2), activation='relu', padding='same') (X)
    X = Conv2D(8, (2, 2), activation='relu', padding='same') (X)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (X)

    model = Model(inputs=X_input, outputs=outputs)
    model.compile(loss=IoU,
                  optimizer=Adam(lr=lr, decay=1e-7),
                  metrics=[my_iou_metric])
    #model.summary()

    return model

input_shape = (256, 256, 3)

#model1(input_shape)

def train_model(input_shape):
    batch_size = 64 #was 16

    #data_gen_args = dict(featurewise_std_normalization=True)
    #featurewise_center=True,
    #rotation_range=90.,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zoom_range=0.2)
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    #print(data_gen_args)

    seed = 1
    # image_datagen.fit(images, augment=False, seed=seed)
    # mask_datagen.fit(masks, augment=False, seed=seed)

    image_generator = image_datagen.flow_from_directory(
    'data/train',
    target_size=input_shape[:2],
    class_mode=None,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    target_size=input_shape[:2],
    color_mode='grayscale',
    class_mode=None,
    seed=seed)

    train_generator = zip(image_generator, mask_generator)

    model = model1(input_shape)
    model.load_weights('checkpoint.h5')

    checkpoint = ModelCheckpoint('checkpoint.h5', monitor='loss',
                                verbose=1, save_best_only=True,
                                save_weights_only=True, mode='min')

    tensbrd = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001)


    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model.fit_generator(
            train_generator,
            steps_per_epoch=23203 // batch_size, #train set divided by batch size
            epochs=100,
            use_multiprocessing=False,
            initial_epoch=40, #change this to continue learning (current epoch is 0)
            callbacks=[checkpoint, tensbrd, reduce_lr])

    model.save_weights('model.h5')

train_model(input_shape)

def test_model():
    """
    Plots predicted mask, ground-truth mask and orig image + mask
    """
    model = model1(input_shape)
    model.load_weights('checkpoint.h5')
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(
        'data/test',
        shuffle=False,
        color_mode='rgb',
        batch_size=1,
        class_mode=None)
    filenames = test_generator.filenames
    pred_arr = model.predict_generator(test_generator, verbose=1)
    masks = train
    idx = 0
    for ImageId in filenames:
        ImageId = ImageId[6:]
        img = imread('data/test/class/' + ImageId)
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        if type(img_masks[0]) == str:
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle_decode(mask)
            pred = pred_arr[idx].reshape((256, 256))
            fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
            all_masks = cv2.resize(all_masks, (256, 256))
            #print(all_masks.shape, pred.shape)
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[2].axis('off')
            axarr[0].imshow(pred)
            axarr[1].imshow(all_masks)
            axarr[2].imshow(img)
            axarr[2].imshow(all_masks, alpha=0.4)
            plt.tight_layout(h_pad=0.1, w_pad=0.1)
            plt.show()
            fig.savefig(ImageId)
            idx += 1
        else:
            print('NaN')

#test_model()

# def eval_model():
#     model = model1(input_shape)
#     model.load_weights('checkpoint.h5')
#     datagen = ImageDataGenerator()
#     test_generator = datagen.flow_from_directory(
#         'data/dev',
#         shuffle=False,
#         color_mode='rgb',
#         batch_size=1,
#         class_mode=None)
#     filenames = test_generator.filenames
#     arr = model.predict_generator(test_generator, verbose=1)
#     np.savetxt('arr.csv', arr.flatten(), delimiter=',')

# eval_model()

# def get_accuracy():
#     datagen = ImageDataGenerator()
#     test_generator = datagen.flow_from_directory(
#         'data/dev',
#         shuffle=False,
#         color_mode='rgb',
#         batch_size=1,
#         class_mode=None)
#     filenames = test_generator.filenames

#     arr = np.loadtxt('arr.csv', delimiter=',')
#     count = 0
#     pred_labels = arr
#     inc_images = {}
#     data = pd.read_csv('train_ship_segmentations.csv', index_col='ImageId')
#     for idx in range(len(pred_labels)):
#         name = filenames[idx][5:]
#         pred_label = pred_labels[idx]
#         true_label = data.loc[name]['EncodedPixels']
#         if str(true_label) == 'nan' and pred_label > 0.5: #count negatives
#             count += 1
#         elif str(true_label) != 'nan' and pred_label <= 0.5: #count positives
#             count += 1
#         else:
#             inc_images[name] = round(pred_label)
#     print('Accuracy:', count/len(pred_labels))
#     sum_1 = sum(value == 1 for value in inc_images.values())
#     sum_0 = sum(value == 0 for value in inc_images.values())
#     print('Incorrectly labeled images:', sum_1, "with label '1' and", sum_0, "with label '0'")
#     #show_incorrect(list(inc_images.keys())) #uncomment this to show incorrectly labeled images

# get_accuracy()

"""
TODO:
change metric and loss. Model works but it doesn't know how it's doing.
I should find more appropriate loss function.
"""