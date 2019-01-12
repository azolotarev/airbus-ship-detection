import numpy as np
import pandas as pd
import os
import shutil
import math
import pickle
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

"""
Train
Input: X - image, Y - masks. Metric: IoU

Prediction
Input: X - image. Output: Y - masks. Metric: IoU

"""
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

class MeanIoU(object):
    """
    Custom metric for model.compile
    model.compile(metrics=[miou_metric.mean_iou])
    Epoch 1/5
    1000/1000 [==] - 0s 214us/step - loss: 1.6345 - mean_iou: 0.1151
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_mean_iou, [y_true, y_pred], tf.float32)

    def np_mean_iou(self, y_true, y_pred):
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick from torchnet for bincounting 2 arrays together
        # https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0

        return np.mean(iou).astype(np.float32)

input_shape = (256, 256, 3)

num_classes = 1

miou_metric = MeanIoU(num_classes)
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

def model1(input_shape):
    X_input = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (X_input)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=X_input, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[my_iou_metric])
    model.summary()

    return model

def train_model(input_shape):
    batch_size = 1

    data_gen_args = dict(featurewise_std_normalization=True)
    #featurewise_center=True,
    #rotation_range=90.,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    print(data_gen_args)

    seed = 1
    # image_datagen.fit(images, augment=False, seed=seed)
    # mask_datagen.fit(masks, augment=False, seed=seed)

    image_generator = image_datagen.flow_from_directory(
    'data/train',
    #target_size=(128,128),
    class_mode=None,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    #target_size=(128,128),
    color_mode='grayscale',
    class_mode=None,
    seed=seed)

    train_generator = zip(image_generator, mask_generator)

    model = model1(input_shape)
    #model.load_weights('checkpoint.h5')

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
            epochs=1,
            use_multiprocessing=False,
            initial_epoch=0, #change this to continue learning (current epoch is 0)
            callbacks=[checkpoint, tensbrd, reduce_lr])

    model.save_weights('model.h5')

train_model(input_shape)

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