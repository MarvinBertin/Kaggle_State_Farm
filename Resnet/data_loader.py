import os, sys
import pickle
import numpy as np
import random
import six
from six.moves import urllib, range
import copy
import tarfile
import logging
from sklearn.cross_validation import LabelShuffleSplit


from ...utils import logger, get_rng
from ...utils.fs import download
from ..base import DataFlow

__all__ = ['KaggleStrat2']

class KaggleStrat2(DataFlow):

    def __init__(self, train_or_test, shuffle=True, dir=None,random_state=1):

        assert train_or_test in ['train', 'test']
            
        DATASET_PATH = os.environ.get('DATASET_PATH', '/home/ubuntu/distracted-drivers-tf/dataset/data_large20.pkl')

        print('Loading dataset {}...'.format(DATASET_PATH))
        with open(DATASET_PATH, 'rb') as f:
            X_train_raw, y_train_raw, self.X_test, self.X_test_ids, driver_ids = pickle.load(f)
            
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        
        for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=1, test_size=0.2, random_state=random_state):
    
            x = X_train_raw #.reshape(X_train_raw.shape[0],3, 24, 32)
            y = y_train_raw
            y = np.argmax(y,axis=1)

            x = x/np.float32(255)
            self.X_test = self.X_test/np.float32(255)
            
            self.pixel_mean = np.mean(np.vstack((self.X_test,x)),axis=0)

            x -= self.pixel_mean
        
            self.X_test -= self.pixel_mean

            pickle.dump( self.pixel_mean, open( "/home/ubuntu/tensorpack/examples/ResNet/pixel_mean.p", "wb" ) )

            X_train = x[train_index,:,:,:]
            Y_train = y[train_index]

            X_test = x[valid_index,:,:,:]
            Y_test = y[valid_index]
            ret = []

            if train_or_test == 'train':
                #####
                for i in range(len(X_train)):
                               img = X_train[i]
                               ret.append([img,Y_train[i]])

            else:
                for i in range(len(X_test)):
                               img = X_test[i]
                               ret.append([img,Y_test[i]])
            #####
                           
  
        self.train_or_test = train_or_test

        self.data = ret
        self.shuffle = shuffle
        self.rng = get_rng(self)

    def reset_state(self):
        self.rng = get_rng(self)

    def size(self):
        return len(self.data) 
    
    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.data[k]

    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0,1))