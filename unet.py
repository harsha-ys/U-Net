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
input = tf.keras.layers.Input((IMG_width, IMG_height, IMG_chanel))