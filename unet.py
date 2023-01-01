# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:16:54 2022

@author: HARSHANA
"""

import tensorflow as tf
IMG_width = 128
IMG_height = 128
IMG_chanel = 3

#build the chanel

#input layer
inputs = tf.keras.layers.Input((IMG_width, IMG_height, IMG_chanel))

s= tf.keras.layers.Lambda(lambda x:x/255)(inputs)#convert integer rgb values into floats within the range 0-1

c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)#drop out to generalize
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPool2D((2,2))(c1)


