#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 23:32:20 2017

@author: srikanthnarayanan
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import glob
import os
# Arrays to store object and image points from the chessboard images


test_path = glob.glob(os.path.join('test_images', 'test*.jpg'))
test_images = []
undist_images = []
for impath in test_path:
    img = cv2.imread(impath)
    test_images.append(img.copy())
    
for img in test_images:
    plt.imshow(img)