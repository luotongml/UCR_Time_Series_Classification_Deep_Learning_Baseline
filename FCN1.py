#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""
 
from __future__ import print_function
 
from keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, Activation
from keras.layers.pooling import GlobalAveragePooling1D
import numpy as np
import pandas as pd
import dataio

import keras 
from keras.callbacks import ReduceLROnPlateau
      
def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  
nb_epochs = 2000
nb_epochs = 20


#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist  = ['Adiac']
for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    print(type(x_train))
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0]/10, 16)
    
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
    
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)

    train = np.concatenate((x_train, Y_train), axis = 1)
    #train.reshape(train.shape + (1,))


    X = pd.DataFrame(train)
    #X.reshape(X.shape + (1,))
    #print(X.columns.values)
    x_test = (x_test - x_train_mean)/(x_train_std)
    #x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))


    iter = dataio.SeriesIterator(X, labels=range(X.shape[1]-nb_classes, X.shape[1]), window=1, shuffle=False)



    #x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)

    model = Sequential()
    model.add(Conv1D(128, 8, border_mode='same', input_shape= x_train.shape[1:]+(1,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(256, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(128,3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001)
    steps = x_train.shape[0] / batch_size
    hist = model.fit_generator(dataio.batch(iter, batch_size=batch_size),steps_per_epoch=steps, epochs=nb_epochs, validation_data=(x_test, Y_test), callbacks=[reduce_lr], workers=1)
    #model.fit_generator()
    #hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
     #         verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print(log.loc[log['loss'].idxmin]['loss'])
    print(log.loc[log['loss'].idxmin]['val_acc'])
