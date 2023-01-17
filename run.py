# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:50:06 2023

@author: HARSHANA
"""

import arrange_dataset as a
import unet as un
import tensorflow as tf
import numpy as np
import random
import cv2
#from keras.utils import multi_gpu_model
#from tensorflow.keras.utils import multi_gpu_model
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

#from skimage.io import imread, imshow
#from skimage.transform import resize
#import matplotlib.pyplot as plt

print("hari")
a.cd()
print("a")
X_train = a.X
Y_train = a.Y


model = un.getModel()




#modelcheckpoint
checkpoint =  tf.keras.callbacks.ModelCheckpoint('UNET_model_cpu.h5', verbose=1, save_best_only=True)

callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                tf.keras.callbacks.TensorBoard(log_dir='logs'),
                checkpoint
            ] 
"""
checkpoint1 =  tf.keras.callbacks.ModelCheckpoint('UNET_model_gpu.h5', monitor='val_loss', verbose=1)

callbacks1 = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
                tf.keras.callbacks.TensorBoard(log_dir='logs'),
                checkpoint1
            ] 
"""
"""
gpu_model = multi_gpu_model(model, gpus=2)
results = gpu_model.fit(X_train, Y_train, batch_size=16, validation_split=0.1, epochs= 100, callbacks=callbacks1)
gpu_model.save('tomato_gpu.hdf5')
"""

results = model.fit(X_train, Y_train, batch_size=16, validation_split=0.1, epochs= 100, callbacks=callbacks)
model.save('tomato_cpu.hdf5')

print("model have saved")

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_train_t = (preds_train > 0.8).astype(np.uint8)
preds_val_t = (preds_val > 0.8).astype(np.uint8)


from PIL import Image as im


"""
X1 = np.zeros((1, 256, 256, 3), dtype=np.uint8)
image1 = cv2.imread(r"D:\maskimages\input\A1_20221130_113011.jpg")
image2 = cv2.resize( image1, ( 256, 256 ))
X1[0] = image2
pre = model.predict(X_train[0], verbose=1)
preds_t = (pre > 0.5).astype(np.uint8)
array4 = np.squeeze(preds_t[0])
data4 = im.fromarray(array4)
data4.save(r"D:\maskimages\test4.png")
"""


array = np.squeeze(X_train[11])
data = im.fromarray(array)
data.save(r"D:\maskimages\test1.png")

array2 = np.squeeze(Y_train[11])
data2 = im.fromarray(array2)
data2.save(r"D:\maskimages\test2.png")


array3 = np.squeeze(preds_train_t[11])
data3 = im.fromarray(array3)
data3.save(r"D:\maskimages\test3.png")


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
#imshow(X_train[ix])
cv2.imwrite("D:\maskimages\test1.png", X_train[ix])
#plt.show()
#imshow(np.squeeze(Y_train[ix]))
cv2.imwrite("D:\maskimages\test12.png", np.squeeze(Y_train[ix]))
#plt.show()
#imshow(np.squeeze(preds_train_t[ix]))
cv2.imwrite("D:\maskimages\test123.png", np.squeeze(preds_train_t[ix]))
#plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
#imshow(X_train[int(X_train.shape[0]*0.9):][ix])
#plt.show()
#imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
#plt.show()
#imshow(np.squeeze(preds_val_t[ix]))
#plt.show()


