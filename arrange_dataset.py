# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:35:33 2023

@author: HARSHANA
"""
import os
import glob
from PIL import Image

class arrange_datase:
    
    def __init__(self, rootPathForImages = None, rootPathForMasks = None, imageType=None):
        
        self.imageSet = [ f for f in glob.glob(rootPathForImages + "*."+imageType) ]
        self.MaskSet = [ f for f in glob.glob(rootPathForMasks + "*."+imageType) ]
    
    def createAFolder(path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def createTheFolderStructureForAnImageAndMask(path):
        
    
    def createTheDataset():
        
        