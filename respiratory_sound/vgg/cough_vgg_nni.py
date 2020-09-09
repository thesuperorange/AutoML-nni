import tensorflow.keras as keras
import logging
import nni
import argparse

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import metrics
#import tensorflow as tf
#import keras
from sklearn.utils import class_weight
from collections import Counter

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use GPU training
LOG = logging.getLogger('cough_mobile_keras')

def get_params():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=False, default=4, help="batch size")
    ap.add_argument("-l", "--lr", required=False, default=1e-3, help="learning rate")
    ap.add_argument("-e", "--epoch", required=False,default=10,  help="epoch number")
    args, _ = ap.parse_known_args()
    return args

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
      
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_accuracy"])
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    train_loc = 'output_img2/train/'
    test_loc = 'output_img2/val/'

    tuner_params = nni.get_next_parameter()
    params = vars(get_params())
    params.update(tuner_params)
    
    
    
    output_folder_name = 'output_vgg/'+str (calendar.timegm(time.gmtime()))
    if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)
    with open(output_folder_name+'/params.json', 'w') as outfile:
        json.dump(params, outfile)


    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory=train_loc, target_size=(224,224),batch_size=params['batch'])
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory=test_loc, target_size=(224,224),batch_size=params['batch'])

    diagnosis_csv = 'patient_diagnosis.csv'
    diagnosis = pd.read_csv(diagnosis_csv, names=['pId', 'diagnosis'])

    categories = diagnosis['diagnosis'].unique()

    
    vgg16 = VGG16(weights='imagenet')
    vgg16.summary()

    x = vgg16.get_layer('fc2').output
    prediction = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=vgg16.input, outputs=prediction)
    

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-20:]:
        layer.trainable = True
        print("Layer '%s' is trainable" % layer.name)

    opt = Adam(lr=params['lr'])
    model.compile(optimizer=opt, loss=categorical_crossentropy,
                  metrics=['accuracy', 'mae'])
    model.summary()

    checkpoint = ModelCheckpoint(output_folder_name+"/mobilenetv2_base_res.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto')
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    counter = Counter(traindata.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    #class_weights
    print(traindata.batch_size)
    hist = model.fit(traindata, steps_per_epoch=traindata.samples // traindata.batch_size, validation_data=testdata,
                     class_weight=class_weights, validation_steps=testdata.samples // testdata.batch_size,
                     epochs=params['epoch'], callbacks=[SendMetrics(), early])

    _, acc = model.evaluate(testdata, verbose=0)

    LOG.debug('Final result is: %d', acc)
    nni.report_final_result(acc)
