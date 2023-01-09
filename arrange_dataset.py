# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:35:33 2023

@author: HARSHANA
"""
import os
import glob
from PIL import Image
import shutil as cpy

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
                if os.path.splitext(imageName)[0] == os.path.splitext(os.path.basename(mask))[0]:
                    self.createTheFolderStructureForAnImageAndMask(rootPath, "image"+str(i))
                    
                    cpy.copy(image, rootPath+"\image"+str(i)+"\image")
                    cpy.copy(mask, rootPath+"\image"+str(i)+"\mask")
                    
                    #self.imageSet.remove(image)
                    self.maskSet.remove(mask)
                    
                    i=i+1
                    break

        
        
if __name__ == "__main__":
    
    object1 =  arrange_datase('D:\maskimages\input', 'D:\maskimages\output', 'jpg')
    
    object1.createTheDataset('D:\maskimages\images')
    

    
        