# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:35:33 2023

@author: HARSHANA
"""
import os
import glob
from PIL import Image
import shutil as cpy
import cv2
import numpy as np
#mport matplotlib.pyplot as plt
#mport matplotlib.image as mpimg
from numpy import asarray

class arrange_datase:
    
    def __init__(self, rootPathForImages = None, rootPathForMasks = None, imageType=None):
        
        imageSet_ = [ f for f in glob.glob(rootPathForImages + "\*."+imageType) ]
        maskSet_ = [ f for f in glob.glob(rootPathForMasks + "\*") ]
        
        self.imageSet = sorted(imageSet_)
        self.maskSet = sorted(maskSet_)
    
    def createAFolder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def createTheFolderStructureForAnImageAndMask(self, rootPath, folderName):
        
        folderPath = rootPath + "\\" + folderName
        self.createAFolder(folderPath)
        
        imagePath = folderPath + '\image'
        maskPath  = folderPath + '\mask'
        
        self.createAFolder(imagePath)
        self.createAFolder(maskPath)
                    
    def createTheDataset(self, rootPath):
        i=1  
        for image in self.imageSet :
            
            imageName = os.path.basename(image)
            for mask in self.maskSet :
                test = os.path.basename(mask)
                fileNameMask = os.path.splitext(os.path.basename(mask))[0]
                extension = os.path.splitext(os.path.basename(mask))[1]
                if os.path.splitext(imageName)[0] == fileNameMask:
                    self.createTheFolderStructureForAnImageAndMask(rootPath, "image"+str(i))
                    
                    cpy.copy(image, rootPath+"\image"+str(i)+"\image") 
                    cpy.copy(mask, rootPath+"\image"+str(i)+"\mask")
                    p=rootPath+"\image"+str(i)+"\mask\\" + test
                    p.encode('unicode_escape')
                    raw_s = r'{}'.format(p)
                    self.createMask(p)
                    
                    #self.imageSet.remove(image)
                    self.maskSet.remove(mask)
                    
                    i=i+1
                    break
    
    def createMask(self, imagePath):
        
        image = cv2.imread(imagePath)
        
        #cat = Image.open(r"C:\Users\HARSHANA\Downloads\A1_20221130_113011-PhotoRoom.png")
        #image2 = cv2.resize( image, ( 400, 400 ))
        L_B = np.array([0, 0, 0])
        U_B = np.array([254, 254, 254])
        mask = cv2.inRange(image, L_B, U_B)
        #cv2.imshow("test", mask)
        
        #cv2.waitKey(0)
        dirName = os.path.dirname(imagePath)
        extension = os.path.splitext(os.path.basename(imagePath))[1]
        p = dirName+"\\"+"bw" + extension
        cv2.imwrite(p, mask)
        
        

        
        
if __name__ == "__main__":
    
    object1 =  arrange_datase('D:\maskimages\input', 'D:\maskimages\output', 'jpg')
    #arrange_datase.createMask("D:\maskimages\images\image1\image\A10_20221130_114159.jpg")
    print("v")
    
    object1.createTheDataset('D:\maskimages\images')
    

    
        