import numpy as np
import pandas as pd
import os
import shutil
from skimage.data import imread
import matplotlib.pyplot as plt
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

def show_train_data():
    """
    Plots original image, mask and image + mask
    """
    masks = train
    for ImageId in os.listdir('train_positives/dev/class1'):
        img = imread('train_positives/dev/class1/' + ImageId)
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
            axarr[0].imshow(img)
            axarr[1].imshow(all_masks)
            axarr[2].imshow(img)
            axarr[2].imshow(all_masks, alpha=0.4)
            plt.tight_layout(h_pad=0.1, w_pad=0.1)
            #savefig(ImageId)
            plt.show()
        else:
            print('NaN')

#show_train_data()

def move_images():
    masks = train
    for ImageId in os.listdir('train/'):
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        if type(img_masks[0]) == str: #move positives
            pass
            # shutil.move('train/' + ImageId,
            #             'train_positives/' + ImageId)
        else: #move negatives
            shutil.move('train/' + ImageId,
                        'train_negatives/' + ImageId)

#move_images()

input_shape = (256, 256, 1) #grayscale image

def model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(400))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def train_model():
    batch_size = 15

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
            'train_positives/train',
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = datagen.flow_from_directory(
            'train_positives/dev',
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary')

    model = model1()
    model.load_weights('checkpoint.h5')

    checkpoint = ModelCheckpoint('checkpoint.h5', monitor='loss',
                                verbose=1, save_best_only=True,
                                save_weights_only=True, mode='min')
    tensbrd = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.00001)


    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model.fit_generator(
            train_generator,
            steps_per_epoch=46406 // batch_size, #train set divided by batch size
            epochs=80,
            validation_data=validation_generator,
            validation_steps=11734 // batch_size, #dev set divided by batch size
            use_multiprocessing=False,
            initial_epoch=78, #change this to continue learning (current epoch is 78)
            callbacks=[checkpoint, tensbrd, reduce_lr])

    model.save_weights('model.h5')

#train_model()

def show_incorrect(inc_images):
    masks = train
    for ImageId in inc_images:
        img = imread('tgen/test/' + ImageId)
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
            axarr[0].imshow(img)
            axarr[1].imshow(all_masks)
            axarr[2].imshow(img)
            axarr[2].imshow(all_masks, alpha=0.4)
            plt.tight_layout(h_pad=0.1, w_pad=0.1)
            plt.show()
        else:
            print('NaN')

def eval_model():
    model = model1()
    model.load_weights('first_model/checkpoint.h5')
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        'tgen/',
        shuffle=False,
        color_mode='grayscale',
        batch_size=1,
        class_mode=None)
    filenames = test_generator.filenames
    arr = model.predict_generator(test_generator, verbose=1)
    np.savetxt('arr.csv', arr, delimiter=',')

#eval_model()

def get_accuracy():
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        'tgen/',
        shuffle=False,
        color_mode='grayscale',
        batch_size=1,
        class_mode=None)
    filenames = test_generator.filenames

    arr = np.loadtxt('arr.csv', delimiter=',')
    count = 0
    pred_labels = arr
    inc_images = {}
    data = pd.read_csv('train_ship_segmentations.csv', index_col='ImageId')
    for idx in range(len(pred_labels)):
        name = filenames[idx][5:]
        pred_label = pred_labels[idx]
        true_label = data.loc[name]['EncodedPixels']
        if str(true_label) == 'nan' and pred_label > 0.5: #count negatives
            count += 1
        elif str(true_label) != 'nan' and pred_label <= 0.5: #count positives
            count += 1
        else:
            inc_images[name] = round(pred_label)
    print('Accuracy:', count/len(pred_labels))
    sum_1 = sum(value == 1 for value in inc_images.values())
    sum_0 = sum(value == 0 for value in inc_images.values())
    print('Incorrectly labeled images:', sum_1, "with label '1' and", sum_0, "with label '0'")
    #show_incorrect(list(inc_images.keys())) #uncomment this to show incorrectly labeled images

#get_accuracy()

"""
Remember that first model gained only 0.005 val_acc after ~12 epochs
Next:

1. Get bounding boxes
2. (on images with ships) boundbox the ships and mask them
3. encode masks
4. create submission

Use rotating bounding boxes, or
Use outline objects?

Outline object will give higher IoU if I understand the evaluation metric correctly.

Do I need model1? Wouldn't model2 recognize ships by design?
That's why I should've built the model that outputs the result directly and THEN decide should I add additional model or
change the architecture or smth else.

No object outlining.
IoU means "how similar two bounding boxes are".
I.e. I should reverse-engineer their bboxing model.

Rotating bounding boxes it is.

Ideas:
Rotate the image? Seems like a no-go
Train model on the full-res image (I need those juicy points)
"""

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