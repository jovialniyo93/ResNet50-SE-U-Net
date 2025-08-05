# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:41:38 2023

@author: kamus
"""

import numpy as np
import cv2
import pandas as pd
import glob
import pickle
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import measure, color, io
import os
import pandas as pd
from skimage import data, io
from skimage import data, io

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed=24
batch_size= 4

model = tf.keras.models.load_model("Unetmodel.h5", compile=False)

img_gen_args = dict(rescale = 1/255.,   
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

mask_gen_args = dict(rescale = 1/255.,  
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 



image_data_generator = ImageDataGenerator(**img_gen_args)
mask_data_generator = ImageDataGenerator(**mask_gen_args)
test_img_generator = image_data_generator.flow_from_directory("new_data/test_images/test/", 
                                                              seed=seed, 
                                                              batch_size=100, 
                                                              target_size=(576, 576),
                                                              class_mode=None) #Default batch size 32, if not specified here

test_mask_generator = mask_data_generator.flow_from_directory("new_data/test_masks/test/", 
                                                              seed=seed, 
                                                              batch_size=100,
                                                              target_size=(576, 576), 
                                                              color_mode = 'grayscale',   #Read masks in grayscale
                                                              class_mode=None)  #Default batch size 32, if not specified here

t = test_img_generator.next()
m = test_mask_generator.next()
for i in range(0,3):
    image = t[i]
    mask = m[i]
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()
#####Watershed 
def process_images(prediction):
  #img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/output3.png")  #Read as color (3 channels)
  img_grey = prediction
  #img_grey = prediction[:,:,0]
  
# change the unet result to binary image
# Threshold image to binary using OTSU. All thresholded pixels will be set to 255
  ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plt.imshow(thresh, cmap = 'gray')
# Morphological operations to remove small noise - opening
# To remove holes we can use closing
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#plt.imshow(opening, cmap = 'gray')

#Watershed should find this area (Sure_Background) for us. 
  sure_bg = cv2.dilate(opening,kernel,iterations=10)
  #plt.imshow(sure_bg, cmap = 'gray')

# extract sure foreground area using distance transform and thresholding
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

# threshold the dist transform by starting at 1/2 its max value.
  ret2, sure_fg = cv2.threshold(dist_transform, 0.48 *dist_transform.max(),255,0)

# Unknown  region will calculate according to bkground - foreground
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)


# For sure regions,  foreground and background will be consider and kabel with positive numbers.
# Unknown regions will be labeled 0. 
  ret3, markers = cv2.connectedComponents(sure_fg)

# add 10 to all labels so that sure background will be consider as 10
  markers = markers+10

# Now, mark the region of unknown with zero
  markers[unknown==255] = 0
  prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
#plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.
 
  markers = cv2.watershed(prediction, markers)
  #plt.imshow(markers, cmap='gray')

#boundary region will be marked -1
#color boundaries in yellow. 
  prediction[markers == -1] = [0,255,255]  

  processed = color.label2rgb(markers, bg_label=0)

  props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
  df = pd.DataFrame(props)
  df = df[df.mean_intensity > 100]  
  #print(dataf.head())
  return df, processed

