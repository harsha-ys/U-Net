# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:16:54 2022

@author: HARSHANA
"""

import arrange_dataset as a

import tensorflow as tf
IMG_width = 256
IMG_height = 256
IMG_chanel = 3


#import unet as un
import numpy as np
import random


#from skimage.io import imread, imshow
#from skimage.transform import resize
#import matplotlib.pyplot as plt

print("hari")
a.cd()
print("a")
X_train = a.X
Y_train = a.Y

#build the chanel

#input layer
def getModel():
    inputs = tf.keras.layers.Input((256, 256, 3))

    s= tf.keras.layers.Lambda(lambda x:x/255)(inputs)#convert integer rgb values into floats within the range 0-1


    #contradiction path(Encoder path)
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)#drop out to generalize
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)
        
    c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)#drop out to generalize
    c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPool2D((2,2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)#drop out to generalize
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPool2D((2,2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)#drop out to generalize
    c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPool2D((2,2))(c4)
        
    c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)#drop out to generalize
    c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    #p5 = tf.keras.layers.MaxPool2D((2,2))(c1)
        
    #expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2,2), padding="same")(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2,2), padding="same")(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2,2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9) 
        
    model= tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

"""
#modelcheckpoint
checkpoint =  tf.keras.callbacks.ModelCheckpoint('UNET_model.h5', monitor='val_loss', verbose=1)

callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='var_loss', patience=2),
                tf.keras.callbacks.TensorBoard(log_dir='logs'),
                checkpoint
            ] 
"""

"""

results = model.fit(X_train, Y_train, batch_size=16, validation_split=0.1, epochs= 100, callbacks=callbacks)

#Expansive path (Decoder path)

model.save('mitochondria_test.hdf5')

print("model have saved")

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
#plt.show()
imshow(np.squeeze(Y_train[ix]))
#plt.show()
imshow(np.squeeze(preds_train_t[ix]))
#plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
#plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
#plt.show()
imshow(np.squeeze(preds_val_t[ix]))
#plt.show()
"""




