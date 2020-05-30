#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:37:54 2018

@author: Purnendu  Mishra
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage import io,color

from keras import backend as K
from keras.utils import Sequence
from keras.preprocessing import image

def load_image(path, color_space = None, target_size = None):
    """Loads an image as an numpy array
    
    Arguments:
        path: Path to image file
        target_size: Either, None (default to original size)
            or tuple of ints '(image height, image width)'
    """
    img = io.imread(path)
    
    if target_size:
        img = cv2.resize(img, target_size, interpolation = cv2.INTER_CUBIC)
        
    if color_space is not None:
        if color_space == 'hsv':
            img = color.rgb2hsv(img)
            
        elif color_space == 'ycbcr':
            img = color.rgb2ycbcr(img)
            
        elif color_space == 'lab':
            img = color.rgb2lab(img)
            
        elif color_space == 'luv':
            img = color.rgb2luv(img)
                    
    return img 

class DataAugmentor(object):
    
    def __init__(self,
                rotation_range   = 0.,
                zoom_range       = 0.,
                horizontal_flip  = False,
                vertical_flip    = False,
                rescale          = None,
                data_format      = None,
                normalize = False,
                mean = None,
                std = None 
                ):
        
        if data_format is None:
            data_format = K.image_data_format()
            
        self.data_format = data_format
        
        if self.data_format == 'channels_last':
            self.row_axis = 0
            self.col_axis = 1
            self.channel_axis = 2
        
        self.rotation_range  = rotation_range
        self.zoom_range      = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.normalize       = normalize
        
        self.rescale = rescale
        self.mean = mean
        self.std = mean
        
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
            
        elif len(zoom_range) == 2:
            self.zoom_range =[zoom_range[0], zoom_range[1]]
            
        else:
            raise ValueError("""zoom_range should be a float or
                             a tuple or lis of two floats. 
                             'Receved args:'""", zoom_range)
        
    
    def random_transforms(self, samples, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            

            
            
        if len(samples) != 2:
            x = samples
            y = None
            
        else:
            x = samples[0]
            y = samples[1]
        
                    
        if self.rotation_range:
            theta = int(180 * np.random.uniform(-self.rotation_range,
                                                self.rotation_range))
            
            (h, w) = x.shape[:2]
            (cx, cy) = [w//2, h//2]
            
            M = cv2.getRotationMatrix2D((cx,cy), -theta, 1.0)
            x = cv2.warpAffine(x , M, (w,h))
            y = cv2.warpAffine(y,  M, (w,h))
            
            
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)

    
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = np.flipud(x)
                y = np.flipud(y)
           
        
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
            
        x = image.random_zoom(x, (zx,zy), channel_axis=self.channel_axis)
        y = np.expand_dims(y, axis=2)
        y = image.random_zoom(y, (zx,zy), channel_axis=self.channel_axis)
        
        return (x,y)
    
    def flow_from_directory(self,
                            root = None, 
                            csv_file = None,
                            target_size=(224,224),
                            color_space = None,
                            batch_size =8,
                            shuffle=False,
                            data_format=None,
                            seed=None):
        return Dataloader(
                    root,
                    csv_file,
                    self,
                    target_size = target_size,
                    color_space = color_space,
                    batch_size  = batch_size,
                    shuffle     = shuffle,
                    data_format = self.data_format,
                    seed        = seed
                     )
    
    def standardize(self,x):
        """Apply the normalization configuration to a batch of inputs.
            Arguments:
                x: batch of inputs to be normalized.
            Returns:
                The inputs, normalized.
        """       
        x = x.astype('float32')
        
        if self.rescale:
            x *= self.rescale
            
            
        if self.normalize:
            if self.mean is not None:
                x -= self.mean
#            else:
#                x -= np.mean(x, axis=self.channel_axis, keepdims=True)
                
                
            if self.std is not None:  
                x /= self.std
                
#            else:
#                x /= (np.std(x, axis=self.channel_axis, keepdims=True) + 1e-7)

        return x


class Dataloader(Sequence):
    ''' Data generator for keras to be used with model.fit_generator
    
    #Arguments:
        csv_file:
        batch_size: Integere, size of a batch
        shuffle: Boolean, whether to shuffle the data between epochs
        target_size = Either 'None' (default to original size) or
            tuple or int '(image height, image width)'
            
    '''
    
    def __init__(self,
                 root     = None,
                 csv_file = None,
                 image_data_generator=None,
                 batch_size   = None, 
                 shuffle      = True,
                 target_size  = None,
                 color_space  = None, 
                 data_format  = 'channel_last',
#                  nb_classes   = 2,
                 seed = None):
        
#         super(Dataloader, self).__init__(self)
        
        if data_format is None:
            data_format = K.image_data_format()
        
        self.root               = root 
        self.files              = pd.read_csv(csv_file, header = None)
        self.image_data_generator =  image_data_generator
        self.batch_size         = batch_size
        self.shuffle            = shuffle
#         self.classes            = nb_classes
        self.target_size        = tuple(target_size)
        self.color_space        = color_space
        self.data_format        = data_format
        self.seed               = seed
        
        if data_format == 'channels_last':
            self.row_axis        = 1
            self.col_axis        = 2
            self.channel_axis    = 3
            self.image_shape = self.target_size + (3,)
            self.label_shape = self.target_size + (1,)
        
        
        self.on_epoch_end()
        
#         print self.files.head()
        
    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))
    
    
    def __getitem__(self, idx):
        

        # total number of samples in the dataset
        n = len(self.files)
        
        if n > idx * self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = n - self.batch_size
        
        file_names = self.f.iloc[idx * current_batch_size : (idx + 1) * current_batch_size, 0]
        
        
    
        batch_x = []
        
        batch_y = []
        
#         print batch_x.shape
        
        for m, files in enumerate(file_names):
            
            image_path = os.path.join(self.root, 'images',files) + '.jpg'
            label_path = os.path.join(self.root, 'labels', files) + '.bmp'
            
            x = load_image(image_path, target_size=self.target_size, color_space=self.color_space)
            y = load_image(label_path, target_size=self.target_size, color_space=None)
            

            x,y = self.image_data_generator.random_transforms((x,y),seed=self.seed)
    
            if len(y.shape) != 3:
                y = np.expand_dims(y, axis = 2)   
            
            # All pixels in labels should have value zero(non-skin) and one(skin)
            y[y != 0] = 1.0
            
            
            # bring pixel values between zero and one
            x = self.image_data_generator.standardize(x)
            
            batch_x.append(x)
            batch_y.append(y)
            
        batch_x = np.array(batch_x, dtype = np.float32)
        batch_y = np.array(batch_y, dtype = np.float32)
            
        return (batch_x, batch_y)
    
    def on_epoch_end(self):
        'Shuffle the at the end of every epoch'
        self.f = self.files.copy()
       
        if self.shuffle == True:
            self.f = self.files.sample(frac=1).reset_index(drop=True)
